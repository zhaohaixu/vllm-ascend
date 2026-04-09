/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "ops.h"
#include "utils.h"
#include "mla_preprocess/op_host/mla_preprocess.h"

namespace vllm_ascend {

aclrtStream enc_stream;
int64_t current_pos = -1;
int64_t current_pos_unalign = -1;

int32_t nounce_counter = []() {
    std::random_device rd;
    std::mt19937 gen(rd());    // Mersenne Twister
    std::uniform_int_distribution<int32_t> dist;
    return dist(gen);
}();

constexpr int AES_BLOCK_SIZE = 16;
constexpr int MAX_BLOCKS_PER_CALL = 4096;
constexpr int AES128_RK_BYTES = 16 * (10 + 1);
constexpr int AES128_RK_PAD_BYTES = 192;

uint8_t key[16] = {
    0x60, 0x3d, 0xeb, 0x10,
    0x15, 0xca, 0x71, 0xbe,
    0x2b, 0x73, 0xae, 0xf0,
    0x85, 0x7d, 0x77, 0x81
};


void expandKey128(const uint8_t inKey[16], uint8_t outExpanded[176]) {
    static const uint8_t sbox[256] = {
        // 256-byte AES S-box
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };
    static const uint8_t Rcon[11] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36};

    std::memcpy(outExpanded, inKey, 16);
    uint32_t bytesGenerated = 16;
    uint8_t rconIter = 1;
    uint8_t t[4];

    while (bytesGenerated < 64) {
        // Last 4 bytes as temp
        for (int i = 0; i < 4; ++i) t[i] = outExpanded[bytesGenerated - 4 + i];
        if (bytesGenerated % 16 == 0) {
            // RotWord
            uint8_t tmp = t[0]; t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = tmp;
            // SubWord
            for (int i = 0; i < 4; ++i) t[i] = sbox[t[i]];
            // Rcon
            t[0] ^= Rcon[rconIter++];
        }
        for (int i = 0; i < 4; ++i) {
            outExpanded[bytesGenerated] = outExpanded[bytesGenerated - 16] ^ t[i];
            bytesGenerated++;
        }
    }

    bytesGenerated = 16;
    rconIter = 1;

    while (bytesGenerated < 176) {
        // Last 4 bytes as temp
        for (int i = 0; i < 4; ++i) t[i] = outExpanded[bytesGenerated - 4 + i];
        if (bytesGenerated % 16 == 0) {
            // RotWord
            uint8_t tmp = t[0]; t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = tmp;
            // SubWord
            for (int i = 0; i < 4; ++i) t[i] = sbox[t[i]];
            // Rcon
            t[0] ^= Rcon[rconIter++];
        }
        for (int i = 0; i < 4; ++i) {
            outExpanded[bytesGenerated] = outExpanded[bytesGenerated - 16] ^ t[i];
            bytesGenerated++;
        }
    }
}

void padRoundKeys192(const uint8_t rk176[AES128_RK_BYTES], uint8_t rk192[AES128_RK_PAD_BYTES]) {
    std::memcpy(rk192, rk176, AES128_RK_BYTES);
    std::memset(rk192 + AES128_RK_BYTES, 0, AES128_RK_PAD_BYTES - AES128_RK_BYTES);
}

AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor &positions, at::Tensor &query, at::Tensor &key,
    int64_t head_size, at::Tensor &cos_sin_cache,  bool is_neox)
{
    int32_t deviceId = 0;
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) &&
              key.size(0) == positions.size(0) &&
              query.size(1) == positions.size(1) &&
              key.size(1) == positions.size(1),
          "query, key and positions must have the same batch_size and seq_len");
    }
    TORCH_CHECK(head_size % 32 == 0, "rotary_embedding: headSize should be divisible by 32");
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);
    TORCH_CHECK(is_neox == true, "rotary_embedding: neox=false is not supported as custom kernel in vllm-ascend");

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    TORCH_CHECK(num_heads % num_kv_heads == 0);
    at::Tensor query_dst = at::empty({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty({num_tokens, num_kv_heads, head_size}, key.options());

    int rot_dim = cos_sin_cache.size(1);
    int seq_dim_idx = positions_ndim - 1;
    int64_t *position_ids_ptr = positions.data_ptr<int64_t>();
    void *query_dst_ptr = query_dst.data_ptr();
    void *key_dst_ptr = key_dst.data_ptr();
    void *query_ptr = query.data_ptr();
    void *key_ptr = key.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int64_t dst_query_stride = query_dst.stride(0);
    int64_t dst_key_stride = key_dst.stride(0);
    at::ScalarType scalar_type = query.scalar_type();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("rotary_embedding");
    cmd.SetCustomHandler([scalar_type, is_neox, num_tokens, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr,
                          query_ptr, key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride,
                          dst_query_stride, dst_key_stride, num_heads, num_kv_heads, head_size]() -> int {
        auto dtype_num = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (num_tokens + aiv_num - 1) / aiv_num;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, dst_query_stride,
                                dst_key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aiv_num);
        return 0;
    });
    cmd.Run();
    return {query_dst, key_dst};
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1, const at::Tensor &wuq,
    const at::Tensor &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const at::Tensor &quant_scale0, const at::Tensor &quant_offset0, const at::Tensor &bias0,
    const at::Tensor &quant_scale1, const at::Tensor &quant_offset1, const at::Tensor &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode, at::Tensor &q_out0,
    at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1)
{
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    
    auto [workspace_tensor, tiling, block_dim] = mlapo::mla_preprocess_tiling(
        hiddenState,
        wuk,
        cache_mode,
        quant_mode
    );

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = quant_scale0.data_ptr();
    void *quant_offset0_ptr = quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = beta1.data_ptr();
    void *quant_scale1_ptr = quant_scale1.data_ptr();
    void *quant_offset1_ptr = quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *sin_ptr = sin.data_ptr();
    void *cos_ptr = cos.data_ptr();
    void *kv_cache_ptr = kv_cache.data_ptr();
    void *slotmapping_ptr = slotmapping.data_ptr();
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = descale0.data_ptr();
    void *descale1_ptr = descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *kv_cache_out0_ptr = kv_cache_out0.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *kv_cache_out1_ptr = kv_cache_out1.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("mla_preprocess");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                          gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr,
                          kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                          qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, workspace_ptr,
                          tiling_ptr, block_dim]() -> int {
        mla_preprocess_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                            gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr,
                            kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                            qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, workspace_ptr,
                            tiling_ptr, block_dim);
        return 0;
    });
    cmd.Run();
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1);
}

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index)
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py#L161-L198
    Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |
    Parameters:
        org_vocab_start_index //base embeddings start
        org_vocab_end_index //base embeddings end
        num_org_vocab_padding //base embeddings padding
        added_vocab_start_index //LoRA embeddings start
        added_vocab_end_index //LoRA embeddings end
    */
{
    // Input validation
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(org_vocab_start_index >= 0, "org_vocab_start_index must be non-negative");
    TORCH_CHECK(org_vocab_end_index >= org_vocab_start_index, "org_vocab_end_index must be greater than org_vocab_start_index");
    TORCH_CHECK(num_org_vocab_padding >= 0, "num_org_vocab_padding must be non-negative");
    TORCH_CHECK(added_vocab_start_index >= org_vocab_end_index, "added_vocab_start_index must be greater than org_vocab_end_index");
    TORCH_CHECK(added_vocab_end_index >= added_vocab_start_index, "added_vocab_end_index must be greater than added_vocab_start_index");

    // Get total number of elements
    int64_t size = input.numel();

    // Create output tensors
    at::Tensor masked_input = at::empty_like(input);
	at::Tensor mask = at::empty_like(input).to(at::kBool);

    // Get data pointers
    void *input_ptr = input.data_ptr();
    void *masked_input_ptr = masked_input.data_ptr();
    void *mask_ptr = mask.data_ptr();

    // Get current stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Get scalar type
    at::ScalarType scalar_type = input.scalar_type();

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("get_masked_input_and_mask");
    cmd.SetCustomHandler([scalar_type, size, stream,
                         input_ptr, masked_input_ptr, mask_ptr,
                         org_vocab_start_index, org_vocab_end_index,
                         num_org_vocab_padding, added_vocab_start_index,
                         added_vocab_end_index]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (size + aiv_num - 1) / aiv_num;

        // Call implementation
        get_masked_input_and_mask_impl(
            stream,
            input_ptr,
            masked_input_ptr,
            mask_ptr,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            size,
            loop_cnt,
            aiv_num);

        return 0;
    });
    cmd.Run();
    return {masked_input, mask};
}

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, input_hidden_token,
                          lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, num_tokens_per_core,
                         input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size, lora_rank,
                          slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size,
                         num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                 at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size,
                          seq_len_ptr, seq_len_size, y_ptr,
                          batch_size, input_hidden_token, lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size,
                         y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void chacha20_encrypt_do(
    at::Tensor &key_stream, 
    at::Tensor &input, 
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos ==  -1) {
        current_pos = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos + data_size / tp_size > element_count) {
        at::Tensor state = at::rand({64}, key_stream.options());
        void* state_ptr = state.data_ptr();
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_generate_mask");
        cmd.SetCustomHandler([threadnum, state_ptr, base_ptr, element_count]() -> int {
            chacha20_encrypt_generate_mask_impl(threadnum, enc_stream, state_ptr, base_ptr, element_count);
            return 0;
        });
        cmd.Run();
        current_pos = 0;
    }

    void* key_stream_ptr = base_ptr + current_pos;
    for (int i = 0; i < tp_size; i++) {
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_do");
        void* input_ptr_ = (void*)(input_ptr + i * (data_size / tp_size));
        void* output_ptr_ = (void*)(output_ptr + i * (data_size / tp_size));
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size / tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos += data_size / tp_size;
    }

    return;
}

void chacha20_encrypt_do_batch(
    at::Tensor &key_stream, 
    at::Tensor &input, 
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos ==  -1) {
        current_pos = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos + data_size / tp_size > element_count) {
        at::Tensor state = at::rand({64}, key_stream.options());
        void* state_ptr = state.data_ptr();
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_generate_mask");
        cmd.SetCustomHandler([threadnum, state_ptr, base_ptr, element_count]() -> int {
            chacha20_encrypt_generate_mask_impl(threadnum, enc_stream, state_ptr, base_ptr, element_count);
            return 0;
        });
        cmd.Run();
        current_pos = 0;
    }

    void* key_stream_ptr = base_ptr + current_pos;
    if (data_size > 1) {
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_do_batch");
        void* input_ptr_ = (void*)input_ptr;
        void* output_ptr_ = (void*)output_ptr;
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_batch_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos += data_size / tp_size;
    }

    return;
}

void chacha20_encrypt_do_unalign(
    at::Tensor &key_stream, 
    at::Tensor &input, 
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos_unalign ==  -1) {
        current_pos_unalign = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;
    uint32_t localSizePadding = (data_size / tp_size + 31) / 32 * 32;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos_unalign + localSizePadding > element_count) {
        at::Tensor state = at::rand({64}, key_stream.options());
        void* state_ptr = state.data_ptr();
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_generate_mask");
        cmd.SetCustomHandler([threadnum, state_ptr, base_ptr, element_count]() -> int {
            chacha20_encrypt_generate_mask_impl(threadnum, enc_stream, state_ptr, base_ptr, element_count);
            return 0;
        });
        cmd.Run();
        current_pos_unalign = 0;
    }

    void* key_stream_ptr = base_ptr + current_pos_unalign;
    for (int i = 0; i < tp_size; i++) {
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_do");
        void* input_ptr_ = (void*)(input_ptr + i * (data_size / tp_size));
        void* output_ptr_ = (void*)(output_ptr + i * (data_size / tp_size));
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_unalign_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size / tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos_unalign += localSizePadding;
    }

    return;
}

void aes_ctr_encrypt_do_batch(
    at::Tensor &key_stream, 
    at::Tensor &input, 
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos ==  -1) {
        current_pos = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos + data_size / tp_size > element_count) {
        printf("generating key stream...\n");
        uint32_t totalBlocks = static_cast<uint32_t>((element_count + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE);
        uint32_t kernelBlocks = (totalBlocks + MAX_BLOCKS_PER_CALL - 1) / MAX_BLOCKS_PER_CALL;

        uint8_t rk176[AES128_RK_BYTES];
        expandKey128(key, rk176);
        uint8_t rk192[AES128_RK_PAD_BYTES];
        padRoundKeys192(rk176, rk192);
        void* deviceRoundKeys = nullptr;
        aclrtMalloc(&deviceRoundKeys, AES128_RK_PAD_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(deviceRoundKeys, AES128_RK_PAD_BYTES, rk192, AES128_RK_PAD_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

        at_npu::native::OpCommand cmd;
        cmd.Name("aes_ctr_encrypt_generate_mask");
        cmd.SetCustomHandler([kernelBlocks, deviceRoundKeys, base_ptr, element_count]() -> int {
            aes128_ecb_encrypt_do_impl(kernelBlocks, enc_stream, deviceRoundKeys, base_ptr, base_ptr, static_cast<uint32_t>(element_count));
            return 0;
        });
        cmd.Run();
        current_pos = 0;

        aclrtFree(deviceRoundKeys);
    }

    void* key_stream_ptr = base_ptr + current_pos;
    if (data_size > 1) {
        at_npu::native::OpCommand cmd;
        cmd.Name("xor_do_batch");
        void* input_ptr_ = (void*)input_ptr;
        void* output_ptr_ = (void*)output_ptr;
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_batch_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos += data_size / tp_size;
    }

    return;
}

void aes_ctr_encrypt_do(
    at::Tensor &key_stream,
    at::Tensor &input,
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos ==  -1) {
        current_pos = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos + data_size / tp_size > element_count) {
        printf("generating key stream...\n");
        uint32_t totalBlocks = static_cast<uint32_t>((element_count + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE);
        uint32_t kernelBlocks = (totalBlocks + MAX_BLOCKS_PER_CALL - 1) / MAX_BLOCKS_PER_CALL;

        uint8_t rk176[AES128_RK_BYTES];
        expandKey128(key, rk176);
        uint8_t rk192[AES128_RK_PAD_BYTES];
        padRoundKeys192(rk176, rk192);
        void* deviceRoundKeys = nullptr;
        aclrtMalloc(&deviceRoundKeys, AES128_RK_PAD_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(deviceRoundKeys, AES128_RK_PAD_BYTES, rk192, AES128_RK_PAD_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

        at_npu::native::OpCommand cmd;
        cmd.Name("aes_ctr_encrypt_generate_mask");
        cmd.SetCustomHandler([kernelBlocks, deviceRoundKeys, base_ptr, element_count]() -> int {
            aes128_ecb_encrypt_do_impl(kernelBlocks, enc_stream, deviceRoundKeys, base_ptr, base_ptr, static_cast<uint32_t>(element_count));
            return 0;
        });
        cmd.Run();
        current_pos = 0;

        aclrtFree(deviceRoundKeys);
    }

    void* key_stream_ptr = base_ptr + current_pos;
    for (int i = 0; i < tp_size; i++) {
        at_npu::native::OpCommand cmd;
        cmd.Name("xor_do");
        void* input_ptr_ = (void*)(input_ptr + i * (data_size / tp_size));
        void* output_ptr_ = (void*)(output_ptr + i * (data_size / tp_size));
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size / tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos += data_size / tp_size;
    }

    return;
}

void aes_ctr_encrypt_do_unalign(
    at::Tensor &key_stream, 
    at::Tensor &input, 
    at::Tensor &output,
    int64_t element_count,
    bool is_enc,
    int64_t tp_size = 1
){
    if (current_pos_unalign ==  -1) {
        current_pos_unalign = element_count;
    }

    char* input_ptr = (char*)input.data_ptr();
    char* output_ptr = (char*)output.data_ptr();
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(key_stream.data_ptr());
    int64_t data_size = input.nbytes();
    int64_t threadnum = element_count / 64 / 2048;
    uint32_t localSizePadding = (data_size / tp_size + 31) / 32 * 32;

    uint32_t maxValue = 4096;

    // bool exists = std::find(intVector.begin(), intVector.end(), data_size) != intVector.end();

    if (!enc_stream) {
        enc_stream = c10_npu::getCurrentNPUStream().stream();
    }

    if (current_pos_unalign + localSizePadding > element_count) {
        uint32_t totalBlocks = static_cast<uint32_t>((element_count + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE);
        uint32_t kernelBlocks = (totalBlocks + MAX_BLOCKS_PER_CALL - 1) / MAX_BLOCKS_PER_CALL;

        uint8_t rk176[AES128_RK_BYTES];
        expandKey128(key, rk176);
        uint8_t rk192[AES128_RK_PAD_BYTES];
        padRoundKeys192(rk176, rk192);
        void* deviceRoundKeys = nullptr;
        aclrtMalloc(&deviceRoundKeys, AES128_RK_PAD_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(deviceRoundKeys, AES128_RK_PAD_BYTES, rk192, AES128_RK_PAD_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

        at_npu::native::OpCommand cmd;
        cmd.Name("aes_ctr_encrypt_generate_mask");
        cmd.SetCustomHandler([kernelBlocks, deviceRoundKeys, base_ptr, element_count]() -> int {
            aes128_ecb_encrypt_do_impl(kernelBlocks, enc_stream, deviceRoundKeys, base_ptr, base_ptr, static_cast<uint32_t>(element_count));
            return 0;
        });
        cmd.Run();
        current_pos_unalign = 0;
    }

    void* key_stream_ptr = base_ptr + current_pos_unalign;
    for (int i = 0; i < tp_size; i++) {
        at_npu::native::OpCommand cmd;
        cmd.Name("chacha20_encrypt_do");
        void* input_ptr_ = (void*)(input_ptr + i * (data_size / tp_size));
        void* output_ptr_ = (void*)(output_ptr + i * (data_size / tp_size));
        cmd.SetCustomHandler([key_stream_ptr, input_ptr_, output_ptr_, data_size, tp_size, maxValue]() -> int {
            chacha20_encrypt_do_unalign_impl(enc_stream, key_stream_ptr, input_ptr_, output_ptr_, data_size / tp_size, maxValue);
            return 0;
        });
        cmd.Run();
    }

    if (!is_enc) {
        current_pos_unalign += localSizePadding;
    }

    return;
}

} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    // vLLM-Ascend custom ops
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_ascend::weak_ref_tensor);

    // Rotary embedding
    // Apply GPT-NeoX style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> (Tensor query, Tensor key)");
    ops.impl("rotary_embedding", torch::kPrivateUse1, &vllm_ascend::rotary_embedding);

    ops.def(
        "get_masked_input_and_mask(Tensor input, "
        "                         int org_vocab_start_index, "
        "                         int org_vocab_end_index, "
        "                         int num_org_vocab_padding, "
        "                         int added_vocab_start_index, "
        "                         int added_vocab_end_index) -> (Tensor masked_input, Tensor mask)");
    ops.impl("get_masked_input_and_mask", torch::kPrivateUse1, &vllm_ascend::get_masked_input_and_mask);

    ops.def("bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y, float scale) -> ()");
    ops.impl("bgmv_shrink", torch::kPrivateUse1, &vllm_ascend::bgmv_shrink);

    ops.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("bgmv_expand", torch::kPrivateUse1, &vllm_ascend::bgmv_expand);

    ops.def("sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y, float scale) -> ()");
    ops.impl("sgmv_shrink", torch::kPrivateUse1, &vllm_ascend::sgmv_shrink);

    ops.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("sgmv_expand", torch::kPrivateUse1, &vllm_ascend::sgmv_expand);

    ops.def(
        "mla_preprocess(Tensor hiddenState, Tensor wdqkv,"
        "               Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1,"
        "               Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
        "               Tensor kv_cache_rope, Tensor slotmapping, Tensor quant_scale0,"
        "               Tensor quant_offset0, Tensor bias0, Tensor quant_scale1, Tensor quant_offset1,"
        "               Tensor bias1, Tensor? ctkv_scale, Tensor? q_nope_scale, str? cache_mode,"
        "               str? quant_mode, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! q_out1,"
        "               Tensor! kv_cache_out1) -> (Tensor q_out0, Tensor kv_cache_out0,"
        "                                          Tensor q_out1, Tensor kv_cache_out1)"
    );
    ops.impl("mla_preprocess", torch::kPrivateUse1, &vllm_ascend::mla_preprocess);

    ops.def(
        "chacha20_encrypt_do(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("chacha20_encrypt_do", torch::kPrivateUse1, &vllm_ascend::chacha20_encrypt_do);

    ops.def(
        "chacha20_encrypt_do_batch(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("chacha20_encrypt_do_batch", torch::kPrivateUse1, &vllm_ascend::chacha20_encrypt_do_batch);

    ops.def(
        "chacha20_encrypt_do_unalign(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("chacha20_encrypt_do_unalign", torch::kPrivateUse1, &vllm_ascend::chacha20_encrypt_do_unalign);

    ops.def(
        "aes_ctr_encrypt_do_batch(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("aes_ctr_encrypt_do_batch", torch::kPrivateUse1, &vllm_ascend::aes_ctr_encrypt_do_batch);

    ops.def(
        "aes_ctr_encrypt_do(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("aes_ctr_encrypt_do", torch::kPrivateUse1, &vllm_ascend::aes_ctr_encrypt_do);

    ops.def(
        "aes_ctr_encrypt_do_unalign(Tensor! keystream, Tensor! input, Tensor! output, int element_count, bool is_enc, int tp_size) -> ()");
    ops.impl("aes_ctr_encrypt_do_unalign", torch::kPrivateUse1, &vllm_ascend::aes_ctr_encrypt_do_unalign);
}
