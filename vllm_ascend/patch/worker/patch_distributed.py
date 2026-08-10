#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import lru_cache
from typing import Any, List, Optional, Union

import os
import torch
import vllm
from collections import namedtuple
from torch.distributed import Backend
from vllm.distributed.parallel_state import (GroupCoordinator,
                                             _get_unique_name, _register_group)
from vllm.logger import logger

from vllm_ascend.distributed.communicator import NPUCommunicator
from vllm_ascend.utils import create_hccl_pg_options


CHACHA20_NAIVE = "chacha20-naive"
AES_NAIVE = "aes-naive"
AES_VEC = "aes-vec"
SUPPORTED_ENCRYPTION_ALGORITHMS = (CHACHA20_NAIVE, AES_NAIVE,
                                   AES_VEC)


@lru_cache(maxsize=1)
def get_encryption_algorithm() -> Optional[str]:
    """Parse VLLM_ENC_ENABLE once for each worker process.

    The variable is intentionally parsed lazily because this module can be
    imported before the worker process finishes setting up its environment.
    """
    configured_value = os.getenv("VLLM_ENC_ENABLE")
    if configured_value is None:
        return None

    algorithm = configured_value.strip().lower()
    if algorithm in SUPPORTED_ENCRYPTION_ALGORITHMS:
        return algorithm

    logger.warning(
        "Unsupported VLLM_ENC_ENABLE=%r. Communication encryption is "
        "disabled. Supported values are: %s.", configured_value,
        ", ".join(SUPPORTED_ENCRYPTION_ALGORITHMS))
    return None


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])
def _split_tensor_dict(
    tensor_dict: dict[str, Union[torch.Tensor, Any]]
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size())))
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


class GroupCoordinatorPatch(GroupCoordinator):

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None
        hccl_pg_options = create_hccl_pg_options(group_name)

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                pg_options=hccl_pg_options)

            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group
        self.device = torch.npu.current_device()

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import \
            MessageQueue
        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6)

        self.use_custom_op_call = False
        self.use_cpu_custom_send_recv = False

        self.encryption_algorithm = get_encryption_algorithm()
        self.is_enc = self.encryption_algorithm is not None

        # Do not reserve key-stream buffers when encryption is disabled or the
        # configured algorithm is invalid.
        self.pool_size_collective = 1024 * 1024 * 1024
        self.pool_size_p2p = 256 * 1024 * 1024
        self.key_stream_for_align: Optional[torch.Tensor] = None
        self.key_stream_for_unalign: Optional[torch.Tensor] = None
        self.key_stream_for_send: Optional[torch.Tensor] = None
        self.key_stream_for_recv: Optional[torch.Tensor] = None
        if self.is_enc:
            self.key_stream_for_align = torch.rand(
                self.pool_size_collective,
                dtype=torch.int8,
                device=self.device)
            self.key_stream_for_unalign = torch.rand(
                self.pool_size_collective,
                dtype=torch.int8,
                device=self.device)
            self.key_stream_for_send = torch.rand(
                self.pool_size_p2p, dtype=torch.int8, device=self.device)
            self.key_stream_for_recv = torch.rand(
                self.pool_size_p2p, dtype=torch.int8, device=self.device)

    def _crypt(self,
               input_: torch.Tensor,
               output: torch.Tensor,
               is_encrypt: bool,
               tp_size: int = 1) -> None:
        if not self.is_enc:
            return
        assert self.key_stream_for_align is not None

        if self.encryption_algorithm == CHACHA20_NAIVE:
            torch.ops._C_ascend.chacha20_naive_encrypt_do(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_NAIVE:
            torch.ops._C_ascend.aes_naive_encrypt_do(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_VEC:
            torch.ops._C_ascend.aes_vec_encrypt_do(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)

    def _crypt_batch(self,
                     input_: torch.Tensor,
                     output: torch.Tensor,
                     is_encrypt: bool,
                     tp_size: int = 1) -> None:
        if not self.is_enc:
            return
        assert self.key_stream_for_align is not None

        if self.encryption_algorithm == CHACHA20_NAIVE:
            torch.ops._C_ascend.chacha20_naive_encrypt_do_batch(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_NAIVE:
            torch.ops._C_ascend.aes_naive_encrypt_do_batch(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_VEC:
            torch.ops._C_ascend.aes_vec_encrypt_do_batch(
                self.key_stream_for_align, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)

    def _crypt_unalign(self,
                       input_: torch.Tensor,
                       output: torch.Tensor,
                       is_encrypt: bool,
                       tp_size: int = 1) -> None:
        if not self.is_enc:
            return
        assert self.key_stream_for_unalign is not None

        if self.encryption_algorithm == CHACHA20_NAIVE:
            torch.ops._C_ascend.chacha20_naive_encrypt_do_unalign(
                self.key_stream_for_unalign, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_NAIVE:
            torch.ops._C_ascend.aes_naive_encrypt_do_unalign(
                self.key_stream_for_unalign, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_VEC:
            torch.ops._C_ascend.aes_vec_encrypt_do_unalign(
                self.key_stream_for_unalign, input_, output,
                self.pool_size_collective, is_encrypt, tp_size)

    def _crypt_send(self,
                    input_: torch.Tensor,
                    output: torch.Tensor,
                    is_encrypt: bool,
                    tp_size: int = 1) -> None:
        if not self.is_enc:
            return
        assert self.key_stream_for_send is not None

        if self.encryption_algorithm == CHACHA20_NAIVE:
            torch.ops._C_ascend.chacha20_naive_encrypt_do_send(
                self.key_stream_for_send, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_NAIVE:
            torch.ops._C_ascend.aes_naive_encrypt_do_send(
                self.key_stream_for_send, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_VEC:
            torch.ops._C_ascend.aes_vec_encrypt_do_send(
                self.key_stream_for_send, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)

    def _crypt_recv(self,
                    input_: torch.Tensor,
                    output: torch.Tensor,
                    is_encrypt: bool,
                    tp_size: int = 1) -> None:
        if not self.is_enc:
            return
        assert self.key_stream_for_recv is not None

        if self.encryption_algorithm == CHACHA20_NAIVE:
            torch.ops._C_ascend.chacha20_naive_encrypt_do_recv(
                self.key_stream_for_recv, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_NAIVE:
            torch.ops._C_ascend.aes_naive_encrypt_do_recv(
                self.key_stream_for_recv, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)
        elif self.encryption_algorithm == AES_VEC:
            torch.ops._C_ascend.aes_vec_encrypt_do_recv(
                self.key_stream_for_recv, input_, output,
                self.pool_size_p2p, is_encrypt, tp_size)
    
    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1,
                   scatter_sizes: Optional[List[int]] = None,
                   gather_sizes: Optional[List[int]] = None) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim,
                                                   gather_dim, scatter_sizes,
                                                   gather_sizes)
    
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        if self.device_communicator is None:
            raise ValueError("No device communicator found")

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        if self.is_enc:
            self._crypt_unalign(input_, input_, True, 1)
            torch.distributed.all_gather_into_tensor(output_tensor,
                                                     input_,
                                                     group=self.device_group)
            self._crypt_unalign(output_tensor, output_tensor, False,
                                self.world_size)
        else:
            torch.distributed.all_gather_into_tensor(output_tensor,
                                                     input_,
                                                     group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])

        return output_tensor
    
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.is_enc:
            self._crypt(input_, input_, True, 1)
        output_ =  self.device_communicator.all_gather(input_, 0)
        single_batch = output_.size(0) // self.world_size
        new_shape = (self.world_size, single_batch, *output_.shape[1:])
        reshaped = output_.view(*new_shape)
        if self.is_enc:
            self._crypt_batch(reshaped, reshaped, False, self.world_size)
        reduced_sum = reshaped.sum(dim=0)
        reduced_sum = reduced_sum.view(input_.shape)
        return reduced_sum

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        all_gather_size = (1 if all_gather_group is None else
                        all_gather_group.world_size)
        all_gather_rank = (0 if all_gather_group is None else
                        all_gather_group.rank_in_group)

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            self.device_communicator.send_tensor_dict(  # type: ignore
                tensor_dict, dst)
            return None

        metadata_list: list[tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [
            k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)
        ]
        assert len(tensor_keys) == len(tensor_list)

        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue

            # send-allgather: send only a slice, then do allgather.
            use_all_gather = (all_gather_group is not None
                            and tensor.numel() % all_gather_size == 0)
            use_all_gather = all_gather_tensors.get(key, use_all_gather) \
                if all_gather_tensors else use_all_gather
            if use_all_gather:
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(tensor,
                                    dst=self.ranks[dst],
                                    group=metadata_group)
            else:
                # use group for GPU tensors
                if self.is_enc:
                    self._crypt_send(tensor, tensor, True, 1)
                torch.distributed.send(tensor,
                                       dst=self.ranks[dst],
                                       group=group)
        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        all_gather_size = (1 if all_gather_group is None else
                        all_gather_group.world_size)
        all_gather_rank = (0 if all_gather_group is None else
                        all_gather_group.rank_in_group)

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            return self.device_communicator.recv_tensor_dict(  # type: ignore
                src)

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                    dtype=value.dtype,
                                    device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue

                # send-allgather: send only a slice, then do allgather.
                use_all_gather = (all_gather_group is not None
                                and tensor.numel() % all_gather_size == 0)
                use_all_gather = all_gather_tensors.get(key, use_all_gather) \
                    if all_gather_tensors else use_all_gather

                if use_all_gather:
                    orig_shape = tensor.shape
                    tensor = tensor.reshape(all_gather_size,
                                            -1)[all_gather_rank]

                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(tensor,
                                        src=self.ranks[src],
                                        group=metadata_group)
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor,
                                           src=self.ranks[src],
                                           group=group)
                    if self.is_enc:
                        self._crypt_recv(tensor, tensor, True, 1)
                if use_all_gather:
                    # do the allgather
                    tensor = all_gather_group.all_gather(  # type: ignore
                        tensor, dim=0)
                    tensor = tensor.reshape(orig_shape)

                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
