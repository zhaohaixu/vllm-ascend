#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch_npu
import torchair._contrib.custom_torch_ops  # type: ignore  # noqa: F401
from torch.nn.functional import scaled_dot_product_attention
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           CommonMetadataBuilder,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.ops.cache import concat_and_cache_mla
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16,
                               enable_custom_op, is_310p, nd_to_nz_2d)
from vllm_ascend.worker.model_runner import (
    ModelInputForNPUBuilder, ModelInputForNPUWithSamplingMetadata)

_ALLOWED_NUM_QUERIES_PER_KV = [32, 64, 128]


class AscendAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if is_310p():
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        else:
            return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_builder_cls() -> Type["AscendMetadataBuilder"]:
        return AscendMetadataBuilder

    @classmethod
    def make_metadata_builder(cls, *args, **kwargs) -> "AscendMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)


class AscendMLAAttentionBackend(AscendAttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["AscendMLAAttentionBackendImpl"]:
        return AscendMLAAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AscendMetadata(AttentionMetadata):
    """Metadata for Ascendbackend.
        * modified from XFormersbackend
    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # Avoid mypy error
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor

    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    chunked_prefill_enabled: bool

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: Optional[torch.Tensor]

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # The query lengths of the input sequences
    query_lens: Optional[List[int]] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["AscendMetadata"] = None
    _cached_decode_metadata: Optional["AscendMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Mask for normal situation
    attn_mask: Optional[torch.Tensor] = None

    # Mask for prefix caching
    compress_mask: Optional[torch.Tensor] = None

    # Mask for chunked prefill
    chunk_mask: Optional[torch.Tensor] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure.
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))

        # Compute some attn_metadata fields which default to None.
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        query_lens = (None if self.query_lens is None else
                      self.query_lens[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure.
        self._cached_prefill_metadata = AscendMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            chunked_prefill_enabled=self.chunked_prefill_enabled,
            block_tables=block_tables,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure.
            return self._cached_decode_metadata

        # Compute some attn_metadata fields which default to None.
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[self.num_prefills:])
        query_lens = (None if self.query_lens is None else
                      self.query_lens[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        # Construct & cache decode-phase attention metadata structure.
        self._cached_decode_metadata = AscendMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            chunked_prefill_enabled=self.chunked_prefill_enabled,
            block_tables=block_tables,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False)
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForNPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)
        if enable_custom_op():
            #advance a step on NPU for existing inputs for a multi-step runner if custom ops is enabled
            torch.ops._C.advance_step_flashattn_ascendc(
                num_seqs=num_seqs,
                num_queries=num_queries,
                block_size=block_size,
                input_tokens=model_input.input_tokens,
                sampled_token_ids=sampled_token_ids,
                input_positions=model_input.input_positions,
                seq_lens=self.seq_lens_tensor,
                slot_mapping=self.slot_mapping,
                block_tables=self.block_tables)
        else:
            # use traditional Pytorch method for updating these tensors.
            # update input_tokens
            sampled_token_ids_list = sampled_token_ids[:
                                                       num_queries].squeeze(  # type: ignore
                                                           -1)
            model_input.input_tokens[:
                                     num_queries] = sampled_token_ids_list  # type: ignore

            # get seq_lens and input_positions
            seq_lens = self.seq_lens_tensor[:num_queries]
            next_seq_lens = seq_lens + 1
            next_input_pos = next_seq_lens - 1

            # update seq_lens and input_positions
            self.seq_lens_tensor[:num_queries] = next_seq_lens
            model_input.input_positions[:
                                        num_queries] = next_input_pos  # type: ignore

            # 计算 block index 和 offset
            block_idx = next_input_pos // block_size
            block_offset = next_input_pos % block_size

            current_block_table = self.block_tables.gather(
                1, block_idx.unsqueeze(-1)).squeeze(-1)
            slot_num = current_block_table * block_size + block_offset

            # update slot_mapping
            self.slot_mapping[:num_queries] = slot_num


class AscendMetadataBuilder(CommonMetadataBuilder[AscendMetadata]):

    _attn_mask_builder = None  # noqa

    def __init__(self, input_builder: "ModelInputForNPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

        self.attn_mask = None
        self.compress_mask = None
        self.chunk_mask = None
        if AscendMetadataBuilder._attn_mask_builder is None:
            AscendMetadataBuilder._attn_mask_builder = AttentionMaskBuilder(
                128, self.input_builder.runner.model_config.dtype)

    def _add_seq_group(
            self, inter_data: ModelInputForNPUBuilder.InterDataForSeqGroup,
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table: List[int] = []
            prefix_cache_hit = any([
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ])
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                if block_tables is not None:
                    block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables  # [:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        graph_pad_size: int,
    ):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        dtype = self.runner.model_config.dtype
        use_npu_graph = graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        max_seq_len = max(max_prefill_seq_len, max_decode_seq_len)
        num_decode_tokens = self.num_decode_tokens

        if self.num_prefills == 0 and use_npu_graph:
            num_seqs = len(seq_lens)
            self.slot_mapping.extend([PAD_SLOT_ID] * graph_pad_size)
            self.block_tables.extend([[]] * graph_pad_size)
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int32,
                device=device,
            )

        if self.num_prefills > 0:
            if block_tables is None or block_tables.numel() == 0:
                # normal mask
                self.attn_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                    max_prefill_seq_len, dtype, device)
                if is_310p():
                    mask_nz = nd_to_nz_2d(self.attn_mask)
                    mask_nz = torch_npu.npu_format_cast(
                        mask_nz.contiguous(), ACL_FORMAT_FRACTAL_NZ)
                    self.attn_mask = mask_nz
            elif self.num_decode_tokens == 0 and not self.input_builder.chunked_prefill_enabled:
                # compress mask for prefix cache
                self.compress_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                    128, dtype, device)
            else:
                # chunk_mask for chunk prefill
                attn_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                    max_seq_len, dtype, device)
                if attn_mask.numel() > 1 and attn_mask[0][1] > 0:
                    # Do not use in-place multiplication to avoid modifying `attn_mask_cache`!
                    attn_mask = attn_mask * -10000
                chunk_mask_list = []
                for i, seq_len in enumerate(seq_lens):
                    context_len = self.context_lens[i]
                    chunk_mask_list.append(attn_mask[context_len:seq_len])
                self.chunk_mask = torch.cat(chunk_mask_list, 0)
        else:
            self.attn_mask = None
            self.compress_mask = None
            self.chunk_mask = None

        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.int32,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return AscendMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            block_tables=block_tables,
            attn_mask=self.attn_mask,
            compress_mask=self.compress_mask,
            chunk_mask=self.chunk_mask,
            chunked_prefill_enabled=self.input_builder.chunked_prefill_enabled,
        )


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.seq_len_cpu_tensor = None
        self.query_len_cpu_tensor = None
        self.key_cache = None
        self.value_cache = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
                   num_tokens = batch_size * seq_len
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads, head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        # View q k v to BSH.
        num_tokens = query.shape[0]
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()
        attn_type = self.attn_type

        output = torch.empty(num_tokens,
                             self.num_heads,
                             self.head_size,
                             dtype=query.dtype,
                             device=query.device)

        if kv_cache.numel() > 0:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

        if hasattr(layer, 'quant_method'):
            isPrefill = True if attn_metadata.num_prefills > 0 else False
            if isPrefill:
                assert attn_metadata.prefill_metadata is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.prefill_metadata.seq_lens).astype(
                        np.int32))
            else:
                assert attn_metadata.decode_metadata is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.decode_metadata.seq_lens).astype(
                        np.int32))
            block_tables = attn_metadata.decode_metadata.block_tables if attn_metadata.decode_metadata else None
            # Details of kv_cache arrangement in attention quantization
            # are implemented by quant_method.
            layer.quant_method.apply(
                layer,
                query,
                key,
                value,
                self.key_cache,
                self.value_cache,
                self.scale,
                block_tables,
                isPrefill,
                attn_metadata,
                output,
                seq_lens_tensor_cpu=self.seq_lens_tensor_cpu)
        else:
            if self.key_cache is not None:
                torch_npu._npu_reshape_and_cache(key=key,
                                                 value=value,
                                                 key_cache=self.key_cache,
                                                 value_cache=self.value_cache,
                                                 slot_indices=slots)

            if attn_metadata.num_prefills > 0:
                # Prefix cache disabled  and  chunk prefill disabled  or  no prefix cache hit
                if (attn_metadata.block_tables is None
                        or attn_metadata.block_tables.numel() == 0):
                    if attn_type == AttentionType.ENCODER_ONLY:
                        # TODO: change to use torch_npu encoder attention op, instead
                        # of torch sdpa
                        query = query.movedim(0, query.dim() - 2)
                        key = key.movedim(0, key.dim() - 2)
                        value = value.movedim(0, value.dim() - 2)

                        causal_attn = (attn_type == AttentionType.DECODER)
                        if attn_metadata.seq_lens is not None:
                            seq_lens_q = seq_lens_kv = attn_metadata.seq_lens
                        attn_masks = [None] * len(seq_lens_q)
                        start_q, start_kv = 0, 0
                        for seq_len_q, seq_len_kv, mask in zip(
                                seq_lens_q, seq_lens_kv, attn_masks):
                            end_q = start_q + seq_len_q
                            end_kv = start_kv + seq_len_kv
                            sub_out = scaled_dot_product_attention(
                                query[None, :, start_q:end_q, :],
                                key[None, :, start_kv:end_kv, :],
                                value[None, :, start_kv:end_kv, :],
                                attn_mask=mask,
                                dropout_p=0.0,
                                is_causal=causal_attn and mask is None,
                                scale=self.scale).squeeze(0).movedim(
                                    query.dim() - 2, 0)
                            output[start_q:end_q, :, :] = sub_out
                            start_q, start_kv = end_q, end_kv
                    else:
                        assert attn_metadata.attn_mask is not None
                        mask = attn_metadata.attn_mask
                        assert attn_metadata.prefill_metadata is not None
                        self.seq_lens_tensor_cpu = torch.from_numpy(
                            np.array(attn_metadata.prefill_metadata.seq_lens).
                            astype(np.int32))
                        if is_310p():
                            # align q k v output tensors
                            query = aligned_16(query)
                            key = aligned_16(key)
                            value = aligned_16(value)
                            output = aligned_16(output)

                            # do reformat in case of broadcasted tensors
                            mask = mask.repeat(
                                self.seq_lens_tensor_cpu.size(0), 1, 1, 1)
                            mask = torch_npu.npu_format_cast(
                                mask.contiguous(), ACL_FORMAT_FRACTAL_NZ)
                        torch_npu._npu_flash_attention(
                            query=query,
                            key=key,
                            value=value,
                            mask=mask,
                            seq_len=self.seq_lens_tensor_cpu,
                            scale_value=self.scale,
                            num_heads=self.num_heads,
                            num_kv_heads=self.num_kv_heads,
                            out=output)
                        output = output[:num_tokens, :, :]
                # Prefix cache only and cache hit
                elif attn_metadata.num_decode_tokens == 0 and not attn_metadata.chunked_prefill_enabled:
                    assert kv_cache is not None
                    assert attn_metadata.prefill_metadata is not None
                    self.seq_lens_tensor_cpu = torch.from_numpy(
                        np.array(
                            attn_metadata.prefill_metadata.seq_lens).astype(
                                np.int32))
                    self.query_lens_tensor_cpu = torch.from_numpy(
                        np.array(
                            attn_metadata.prefill_metadata.query_lens).astype(
                                np.int32))
                    block_tables = attn_metadata.prefill_metadata.block_tables
                    assert attn_metadata.compress_mask is not None
                    compress_mask = attn_metadata.compress_mask
                    torch_npu._npu_flash_attention_qlens(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        block_table=block_tables,
                        mask=compress_mask,
                        seq_len=self.query_lens_tensor_cpu,
                        context_lens=self.seq_lens_tensor_cpu,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)
                # Splitfuse
                else:
                    assert kv_cache is not None
                    self.seq_lens_tensor_cpu = torch.from_numpy(
                        np.array(attn_metadata.seq_lens).astype(np.int32))
                    self.query_lens_tensor_cpu = torch.from_numpy(
                        np.array(attn_metadata.query_lens).astype(np.int32))
                    block_tables = attn_metadata.block_tables
                    assert attn_metadata.chunk_mask is not None
                    chunk_mask = attn_metadata.chunk_mask
                    torch_npu._npu_paged_attention_splitfuse(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        block_table=block_tables,
                        context_lens=self.seq_lens_tensor_cpu,
                        mask=chunk_mask,
                        seq_len=self.query_lens_tensor_cpu,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)
            # Decode only
            else:
                assert self.key_cache is not None
                assert self.value_cache is not None
                assert attn_metadata.decode_metadata is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.decode_metadata.seq_lens).astype(
                        np.int32))
                if is_310p():
                    # # seq_lens_tensor needs to be transferred to the device for 310P
                    self.seq_lens_tensor_cpu = self.seq_lens_tensor_cpu.to(
                        device=self.key_cache.device)
                block_tables = attn_metadata.decode_metadata.block_tables
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=block_tables,
                    context_lens=self.seq_lens_tensor_cpu,
                    out=output)

        return output.view(num_tokens, self.hidden_size)


class AscendMLAAttentionBackendImpl(MLAAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.seq_len_cpu_tensor = None

        # MLA Args
        self.q_lora_rank = extra_impl_args['q_lora_rank']
        self.kv_lora_rank = extra_impl_args['kv_lora_rank']
        self.qk_nope_head_dim = extra_impl_args['qk_nope_head_dim']
        self.qk_rope_head_dim = extra_impl_args['qk_rope_head_dim']
        self.qk_head_dim = extra_impl_args['qk_head_dim']
        self.v_head_dim = extra_impl_args['v_head_dim']
        self.rotary_emb = extra_impl_args['rotary_emb']
        self.q_proj = extra_impl_args['q_proj']
        self.kv_b_proj = extra_impl_args['kv_b_proj']
        self.o_proj = extra_impl_args['o_proj']
        self.kv_a_proj_with_mqa = extra_impl_args.get('kv_a_proj_with_mqa',
                                                      None)
        self.kv_a_layernorm = extra_impl_args.get('kv_a_layernorm', None)
        self.k_pe_cache = None
        self.k_nope_cache = None
        self.w_kc = None
        self.w_vc = None

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled


    def exec_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
    ):
        B = hidden_states.shape[0]
        N = self.num_kv_heads
        S = 1
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv = kv.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)

        k_pe, k_nope, _, _ = torch.ops.npu_inference.npu_kv_rmsnorm_rope_cache(
            kv,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA",
        )

        return k_pe, k_nope

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, num_heads, head_size]
            cos: [num_tokens, head_size // 2]
            sin: [num_tokens, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        """
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch.ops.npu_inference.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.w_kc is None or self.w_vc is None:
            kv_b_proj_weight = self.kv_b_proj.weight.reshape(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank)
            self.w_kc = kv_b_proj_weight[:, :self.
                                         qk_nope_head_dim, :].contiguous()
            self.w_vc = kv_b_proj_weight[:,
                                         self.qk_nope_head_dim:, :].transpose(
                                             1, 2).contiguous()

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,
        hidden_states_or_kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            hidden_states_or_q_c: shape = [num_tokens, num_heads * head_size]
                                           num_tokens = batch_size * seq_len
            hidden_states_or_kv_c_normed: shape = [num_tokens, num_kv_heads * head_size]
            k_pe: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [1, num_blocks, block_size,
                               num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        if attn_metadata is None:
            # for profile run
            return hidden_states_or_q_c

        num_tokens = hidden_states_or_q_c.shape[0]
        q = self.q_proj(hidden_states_or_q_c)[0].view(-1, self.num_heads,
                                                      self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        if k_pe is None and attn_metadata.decode_metadata:
            seq_len = self.rotary_emb.max_position_embeddings

            cos = self.rotary_emb.cos_cached[:seq_len].to(dtype=q_pe.dtype)
            sin = self.rotary_emb.sin_cached[:seq_len].to(dtype=q_pe.dtype)
            cos = cos[attn_metadata.input_positions]
            sin = sin[attn_metadata.input_positions]
            cos = cos[:, None, None, :]
            sin = sin[:, None, None, :]

            q_pe = self.rope_single(q_pe, cos, sin)
            k_pe, k_nope = self.exec_kv(hidden_states_or_kv_c_normed, cos, sin,
                                        kv_cache, attn_metadata.slot_mapping)
        else:
            if k_pe is None:
                # NOTE: k_pe is None when graph mode enabled
                kv_c, k_pe = self.kv_a_proj_with_mqa(
                    hidden_states_or_kv_c_normed)[0].split(
                        [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            else:
                kv_c_normed = hidden_states_or_kv_c_normed
            k_pe = k_pe.view(num_tokens, self.num_kv_heads, -1)
            if self.rotary_emb.__class__.__name__ == 'RotaryEmbedding':
                # NOTE: When scaling not specified
                ori_q_pe_shape, ori_k_pe_shape = q_pe.shape, k_pe.shape
                q_pe = q_pe.reshape(num_tokens, -1)
                k_pe = k_pe.reshape(num_tokens, -1)
                q_pe, k_pe = self.rotary_emb(attn_metadata.input_positions,
                                             q_pe, k_pe)
                q_pe = q_pe.view(ori_q_pe_shape)
                k_pe = k_pe.view(ori_k_pe_shape)
            else:
                q_pe, k_pe = self.rotary_emb(attn_metadata.input_positions,
                                             q_pe, k_pe)

        if attn_metadata.num_prefills > 0:
            kv = self.kv_b_proj(kv_c_normed)[0].view(num_tokens,
                                                     self.num_heads, -1)
            k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim],
                                     dim=-1)
        else:
            q_nope_t = torch.transpose(q_nope, 0, 1)
            q_nope_out = torch.bmm(q_nope_t, self.w_kc)
            q_nope = torch.transpose(q_nope_out, 0, 1)

        query = torch.cat([q_nope, q_pe], dim=-1).view(num_tokens,
                                                       self.num_heads, -1)

        # TODO: Replace the env with more flexible expressions
        if self.torchair_graph_enabled:
            if len(kv_cache) > 0 and kv_cache[0].numel(
            ) > 0 and attn_metadata.num_prefills > 0:
                slots = attn_metadata.slot_mapping
                # NOTE: Separate the kv cache in advance to avoid OOM or other issues
                torch_npu._npu_reshape_and_cache(key=kv_c_normed.view(
                    num_tokens, self.num_kv_heads, -1),
                                                 value=k_pe,
                                                 key_cache=kv_cache[0],
                                                 value_cache=kv_cache[1],
                                                 slot_indices=slots)
        elif kv_cache.numel() > 0:
            # TODO replace this naive implement with fusion kernel
            concat_and_cache_mla(kv_c_normed, k_pe, kv_cache,
                                 attn_metadata.slot_mapping)

        if attn_metadata.num_prefills > 0:
            attn_output = torch.empty(num_tokens,
                                      self.num_heads,
                                      self.v_head_dim,
                                      dtype=query.dtype,
                                      device=query.device)
            if (attn_metadata.block_tables is None
                    or attn_metadata.block_tables.numel() == 0):
                assert attn_metadata.attn_mask is not None
                assert attn_metadata.prefill_metadata is not None
                assert attn_metadata.prefill_metadata.seq_lens is not None
                mask = attn_metadata.attn_mask
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.prefill_metadata.seq_lens).astype(
                        np.int32))
                k_pe = k_pe.repeat(1, self.num_heads, 1)
                key = torch.cat(
                    [k_nope.view(num_tokens, self.num_heads, -1), k_pe], dim=2)
                torch_npu._npu_flash_attention(
                    query=query,
                    key=key,
                    value=value,
                    mask=mask,
                    seq_len=self.seq_lens_tensor_cpu,
                    scale_value=self.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_heads,
                    out=attn_output)
            else:
                # TODO: Will support prefix cache and chunked prefill soon.
                raise RuntimeError(
                    "Prefix cache and chunked prefill are currently not supported."
                )
        elif attn_metadata.decode_metadata:
            assert kv_cache is not None
            if self.torchair_graph_enabled:
                # shape of query for npu graph mode should be:
                # [bs, num_heads_per_rank, seq_len, dim]
                q_nope = q_nope.view(num_tokens, self.num_heads, 1, -1)
                q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
                # shape of knope/k_pe for npu graph mode should be:
                # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
                block_size = kv_cache[0].shape[1]
                k_nope = k_nope.view(-1, self.num_kv_heads, block_size,
                                     self.kv_lora_rank)
                k_pe = k_pe.view(-1, self.num_kv_heads, block_size,
                                 self.qk_rope_head_dim)
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope,
                    k_nope,
                    query_rope=q_pe,
                    key_rope=k_pe,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BNSD",
                    atten_mask=attn_metadata.attn_mask,
                    scale=self.scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=attn_metadata.block_tables,
                    block_size=block_size,
                    actual_seq_lengths_kv=attn_metadata.seq_lens,
                )
                attn_output = attn_output.view(num_tokens, -1,
                                               self.kv_lora_rank).transpose(
                                                   0, 1)
                attn_output = torch.bmm(attn_output, self.w_vc).transpose(0, 1)
            else:
                # if torch.empty is used here, the preemptive scheduling case of
                # test_mtp_correctness.py will fail to run.
                attn_output = torch.randn(
                    [num_tokens, self.num_heads, self.kv_lora_rank],
                    dtype=query.dtype,
                    device=query.device)
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.decode_metadata.seq_lens).astype(
                        np.int32))
                block_tables = attn_metadata.decode_metadata.block_tables
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=block_tables,
                    context_lens=self.seq_lens_tensor_cpu,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output)
                attn_output_t = torch.transpose(attn_output, 0, 1)
                attn_output_t = torch.bmm(attn_output_t, self.w_vc)
                attn_output = torch.transpose(attn_output_t, 0, 1)

        output, _ = self.o_proj(attn_output.reshape(num_tokens, -1))

        return output
