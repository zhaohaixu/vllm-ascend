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

from typing import List, Optional, Union

import os
import torch
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import (GroupCoordinator,
                                             _get_unique_name, _register_group)

from vllm_ascend.distributed.communicator import NPUCommunicator
from vllm_ascend.utils import create_hccl_pg_options


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

        self.element_count = 1024 * 1024 * 1024
        self.key_stream = torch.rand(self.element_count, dtype=torch.int8, device=self.device)
        self.key_stream_for_unalign = torch.rand(self.element_count, dtype=torch.int8, device=self.device)
        self.is_enc = os.getenv("VLLM_ENC_ENABLE")
        self.counter = 0

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
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do_unalign(self.key_stream_for_unalign, input_, input_, self.element_count, True, 1)
            torch.distributed.all_gather_into_tensor(output_tensor,
                                                     input_,
                                                     group=self.device_group)
            torch.ops._C_ascend.chacha20_encrypt_do_unalign(self.key_stream_for_unalign, output_tensor, output_tensor, self.element_count, False, self.world_size)
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

        # chacha20_encrypt_do!!
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, True, 1)
            # self.current_pos = torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, self.current_pos, True, 1)
        output_ =  self.device_communicator.all_gather(input_, 0)
        single_batch = output_.size(0) // self.world_size
        new_shape = (self.world_size, single_batch, *output_.shape[1:])
        reshaped = output_.view(*new_shape)
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do_batch(self.key_stream, reshaped, reshaped, self.element_count, False, self.world_size)
            # self.current_pos = torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, reshaped, reshaped, self.element_count, self.current_pos, False, self.world_size)
        reduced_sum = reshaped.sum(dim=0)
        reduced_sum = reduced_sum.view(input_.shape)
        return reduced_sum
    
    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, True, 1)
            # self.current_pos = torch.ops._C.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, self.current_pos, True, 1)
        output = self.device_communicator.all_gather(input_, dim)
        # output = self.device_communicator.gather(input_, dst, dim)
        if self.rank_in_group == dst:
            if self.is_enc is not None:
                torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, output, output, self.element_count, False, self.world_size)
                # self.current_pos = torch.ops._C.chacha20_encrypt_do(self.key_stream, output, output, self.element_count, self.current_pos, False, self.world_size)
            return output
        else:
            if self.is_enc is not None:
                torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, False, 1)
                # self.current_pos = torch.ops._C.chacha20_encrypt_do(self.key_stream, input_, input_, self.element_count, self.current_pos, False, 1)
            return input_

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, tensor, tensor, self.element_count, True, 1)
            # self.current_pos = torch.ops._C.chacha20_encrypt_do(self.key_stream, tensor, tensor, self.element_count, self.current_pos, True, 1)
        self.device_communicator.send(tensor, dst)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        output_ = self.device_communicator.recv(size, dtype, src)
        if self.is_enc is not None:
            torch.ops._C_ascend.chacha20_encrypt_do(self.key_stream, output_, output_, self.element_count, False, 1)
            # self.current_pos = torch.ops._C.chacha20_encrypt_do(self.key_stream, output_, output_, self.element_count, self.current_pos, False, 1)
        return output_


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
