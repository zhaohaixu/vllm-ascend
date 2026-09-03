/**
 * @file ascen.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

//#define M_BATCH_SIZE 4000
#define STATE_BATCH_SIZE 640

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, size_t m_batch_size) {
        xGm.SetGlobalBuffer((__gm__ uint16_t *)x + AscendC::GetBlockIdx() * 8 * STATE_BATCH_SIZE, 8 * STATE_BATCH_SIZE);
        yGm.SetGlobalBuffer((__gm__ uint16_t *)y + 16 * STATE_BATCH_SIZE *  AscendC::GetBlockIdx(), 8 * STATE_BATCH_SIZE + 8 * STATE_BATCH_SIZE);
        zGm.SetGlobalBuffer((__gm__ uint16_t *)z + AscendC::GetBlockIdx() * 8 * STATE_BATCH_SIZE, 8 * STATE_BATCH_SIZE);
        pipe.InitBuffer(inQueueX, 1, 8 * STATE_BATCH_SIZE * sizeof(uint16_t));
        pipe.InitBuffer(inQueueY, 1, 8 * STATE_BATCH_SIZE * sizeof(uint16_t) * 2);
        pipe.InitBuffer(outQueueZ, 1, 8 * STATE_BATCH_SIZE * sizeof(uint16_t));
        pipe.InitBuffer(updateBuffer, 8 * STATE_BATCH_SIZE * sizeof(uint16_t) * 4);
        pipe.InitBuffer(statesBuffer, 8 * STATE_BATCH_SIZE * sizeof(uint16_t) * 5);
        pipe.InitBuffer(initBuffer, 8 * STATE_BATCH_SIZE * sizeof(uint16_t) * 2);
    }
    __aicore__ inline void Process(int m_batch_size) {
        for (int i = 0; i < m_batch_size; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        
    }

private:
    __aicore__ inline void CopyIn(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.AllocTensor<uint16_t>(); // iv + key
        AscendC::DataCopy(xLocal, xGm[8 * STATE_BATCH_SIZE * progress], 8 * STATE_BATCH_SIZE);
        AscendC::DataCopy(yLocal, yGm, 8 * STATE_BATCH_SIZE * 2);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.AllocTensor<uint16_t>();
        AscendC::LocalTensor<uint16_t> states = statesBuffer.Get<uint16_t>();

        if (progress == 0) {
        Initialization(states, yLocal, yLocal[8 * STATE_BATCH_SIZE]);
        }
        Encryption(states, xLocal, zLocal);
        
        //Finalization(states, yLocal, zLocal);

        outQueueZ.EnQue<uint16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

    }
    __aicore__ inline void CopyOut(int progress) {
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.DeQue<uint16_t>();
        AscendC::DataCopy(zGm[8 * STATE_BATCH_SIZE * progress], zLocal, 8 * STATE_BATCH_SIZE);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void StateUpdate(AscendC::LocalTensor<uint16_t> &states, const AscendC::LocalTensor<uint16_t> &p) {
        int32_t state_size = STATE_BATCH_SIZE * 8, shift_const = 2;
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16p = p.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<uint16_t> cache = updateBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16cache = cache.ReinterpretCast<int16_t>();
        // Round 1
        AscendC::And(cache, states[1 * state_size], states[2 * state_size], state_size); // s1 & s2
        AscendC::Add(i16cache, i16cache, i16states[0 * state_size], state_size); // s0 + (s1 & s2)
        AscendC::Add(i16cache, i16cache, i16states[3 * state_size], state_size); // s0 + (s1 & s2) + s3

        AscendC::ShiftLeft(states[0 * state_size], cache, (uint16_t) 5, state_size); // (s0 + (s1 & s2) + s3) << 5
        AscendC::ShiftRight(cache, cache, (uint16_t) 11, state_size); // (s0 + (s1 & s2) + s3) >> 11
        AscendC::Or(states[0 * state_size], states[0 * state_size], cache, state_size); // (s0 + (s1 & s2) + s3) <<< 5

        AscendC::DataCopy(cache, states[3 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[3 * state_size], states[3 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[3 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], cache, shift_const * STATE_BATCH_SIZE); // rotate 32
        // Round 2
        shift_const = 4;
        AscendC::And(cache, states[2 * state_size], states[3 * state_size], state_size); // s2 & s3
        AscendC::Add(i16cache, i16cache, i16states[1 * state_size], state_size); // s1 + (s2 & s3)
        AscendC::Xor(cache[state_size], cache, states[4 * state_size], state_size); // s1 + (s2 & s3) ^ s4
        AscendC::Add(i16cache, i16cache[state_size], i16p, state_size); // s1 + (s2 & s3) ^ s4 + p

        AscendC::ShiftLeft(states[1 * state_size], cache, (uint16_t) 15, state_size); // (s1 + (s2 & s3) ^ s4 + p) << 15
        AscendC::ShiftRight(cache, cache, (uint16_t) 1, state_size); // (s1 + (s2 & s3) ^ s4 + p) >> 1
        AscendC::Or(states[1 * state_size], states[1 * state_size], cache, state_size); // (s1 + (s2 & s3) ^ s4 + p) <<< 15

        AscendC::DataCopy(cache, states[4 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[4 * state_size], states[4 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[4 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], cache, shift_const * STATE_BATCH_SIZE); // rotate 64

        // Round 3
        shift_const = 6;
        AscendC::And(cache, states[3 * state_size], states[4 * state_size], state_size); // s3 & s4
        AscendC::Add(i16cache, i16cache, i16states[2 * state_size], state_size); // s2 + (s3 & s4)
        AscendC::Xor(cache[state_size], cache, states[0 * state_size], state_size); // s2 + (s3 & s4) ^ s0
        AscendC::Add(i16cache, i16cache[state_size], i16p, state_size); // s2 + (s3 & s4) ^ s0 + p

        AscendC::ShiftLeft(states[2 * state_size], cache, (uint16_t) 7, state_size); // (s2 + (s3 & s4) ^ s0 + p) << 7
        AscendC::ShiftRight(cache, cache, (uint16_t) 9, state_size); // (s2 + (s3 & s4) ^ s0 + p) >> 9
        AscendC::Or(states[2 * state_size], states[2 * state_size], cache, state_size); // (s2 + (s3 & s4) ^ s0 + p) <<< 7

        AscendC::DataCopy(cache, states[0 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[0 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], states[0 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[0 * state_size], cache, (8 - shift_const) * STATE_BATCH_SIZE); // rotate 96

        // Round 4
        shift_const = 4;
        AscendC::And(cache, states[4 * state_size], states[0 * state_size], state_size); // s4 & s0
        AscendC::Add(i16cache, i16cache, i16states[3 * state_size], state_size); // s3 + (s4 & s0)
        AscendC::Add(i16cache[state_size], i16cache, i16states[1 * state_size], state_size); // s3 + (s4 & s0) + s1
        AscendC::Add(i16cache, i16cache[state_size], i16p, state_size); // s3 + (s4 & s0) + s1 + p

        AscendC::ShiftLeft(states[3 * state_size], cache, (uint16_t) 8, state_size); // (s3 + (s4 & s0) + s1 + p) << 8
        AscendC::ShiftRight(cache, cache, (uint16_t) 8, state_size); // (s3 + (s4 & s0) + s1 + p) >> 8
        AscendC::Or(states[3 * state_size], states[3 * state_size], cache, state_size); // (s3 + (s4 & s0) + s1 + p) <<< 8

        AscendC::DataCopy(cache, states[1 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[1 * state_size], states[1 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[1 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], cache, shift_const * STATE_BATCH_SIZE); // rotate 64

        // Round 5
        shift_const = 2;
        AscendC::And(cache, states[0 * state_size], states[1* state_size], state_size); // s0 & s1
        AscendC::Add(i16cache, i16cache, i16states[4 * state_size], state_size); // s4 + (s0 & s1)
        AscendC::Add(i16cache[state_size], i16cache, i16states[2 * state_size], state_size); // s4 + (s0 & s1) + s2
        AscendC::Add(i16cache, i16cache[state_size], i16p, state_size); // s4 + (s0 & s1) + s2 + p

        AscendC::ShiftLeft(states[4 * state_size], cache, (uint16_t) 13, state_size); // (s4 + (s0 & s1) + s2 + p) << 13
        AscendC::ShiftRight(cache, cache, (uint16_t) 3, state_size); // (s4 + (s0 & s1) + s2 + p) >> 13
        AscendC::Or(states[4 * state_size], states[4 * state_size], cache, state_size); // (s4 + (s0 & s1) + s2 + p) <<< 13

        AscendC::DataCopy(cache, states[2 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[2 * state_size], states[2 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE);
        AscendC::DataCopy(states[2 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], cache, shift_const * STATE_BATCH_SIZE); // rotate 32
    }
    __aicore__ inline void Initialization(AscendC::LocalTensor<uint16_t> &states, const AscendC::LocalTensor<uint16_t> &ivs, const AscendC::LocalTensor<uint16_t> &keys) {
        int32_t state_size = STATE_BATCH_SIZE * 8;
        AscendC::LocalTensor<uint16_t> cache = initBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16keys = keys.ReinterpretCast<int16_t>();
        AscendC::DataCopy(states[0 * state_size], ivs, state_size); // s0 = iv
        AscendC::DataCopy(states[1 * state_size], keys, state_size); // s1 = k
        AscendC::Duplicate(states[2 * state_size], (uint16_t) 0xFFFF, state_size); // s2 = 1^128

        AscendC::Duplicate(states[3 * state_size + 0 * STATE_BATCH_SIZE], (uint16_t) 0x1, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 1 * STATE_BATCH_SIZE], (uint16_t) 0x2, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 2 * STATE_BATCH_SIZE], (uint16_t) 0x3, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 3 * STATE_BATCH_SIZE], (uint16_t) 0x4, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 4 * STATE_BATCH_SIZE], (uint16_t) 0x5, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 5 * STATE_BATCH_SIZE], (uint16_t) 0x6, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 6 * STATE_BATCH_SIZE], (uint16_t) 0x7, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 7 * STATE_BATCH_SIZE], (uint16_t) 0x8, STATE_BATCH_SIZE); // s3 = const0
        AscendC::DataCopy(cache, states[3 * state_size], state_size); // const0 backup

        AscendC::Duplicate(states[4 * state_size + 0 * STATE_BATCH_SIZE], (uint16_t) 0x2, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 1 * STATE_BATCH_SIZE], (uint16_t) 0x3, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 2 * STATE_BATCH_SIZE], (uint16_t) 0x4, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 3 * STATE_BATCH_SIZE], (uint16_t) 0x5, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 4 * STATE_BATCH_SIZE], (uint16_t) 0x6, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 5 * STATE_BATCH_SIZE], (uint16_t) 0x7, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 6 * STATE_BATCH_SIZE], (uint16_t) 0x8, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[4 * state_size + 7 * STATE_BATCH_SIZE], (uint16_t) 0x9, STATE_BATCH_SIZE); // s4 = const1

        for (int i = 0; i < 16; i++) {
            StateUpdate(states, cache);
        }

        AscendC::Add(i16states[1 * state_size], i16states[1 * state_size], i16keys, state_size); // s1 += k
    }
    __aicore__ inline void Encryption(AscendC::LocalTensor<uint16_t> &states, AscendC::LocalTensor<uint16_t> m, AscendC::LocalTensor<uint16_t> c) {
        int32_t shift_const = 1, state_size = STATE_BATCH_SIZE * 8;
        AscendC::LocalTensor<uint16_t> cache = initBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16cache = cache.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16c = c.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16m = m.ReinterpretCast<int16_t>();
        AscendC::DataCopy(cache[0 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], states[0 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(cache[0 * state_size], states[0 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE); // s0 <<< 5

        shift_const = 6;
        AscendC::DataCopy(cache[1 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], states[0 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(cache[1 * state_size], states[0 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE); // s1 <<< 96

        AscendC::Add(i16cache[0 * state_size], i16cache[0 * state_size], i16cache[1 * state_size], state_size); // s0 <<< 5 + s1 <<< 96
        AscendC::And(cache[1 * state_size], states[2 * state_size], states[3 * state_size], state_size); // s2 & s3
        AscendC::Add(i16cache[0 * state_size], i16cache[0 * state_size], i16cache[1 * state_size], state_size); // s0 <<< 5 + s1 <<< 96 + (s2 & s3)
        AscendC::Add(i16c, i16m, i16cache[0 * state_size], state_size); // c = m + s0 <<< 5 + s1 <<< 96 + (s2 & s3)

        StateUpdate(states, m);
    }
    __aicore__ inline void Finalization(AscendC::LocalTensor<uint16_t> &states, const AscendC::LocalTensor<uint16_t> &lengths, const AscendC::LocalTensor<uint16_t> &tags) {
        int32_t shift_const = 6, state_size = STATE_BATCH_SIZE * 8;
        AscendC::LocalTensor<uint16_t> cache = initBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16cache = cache.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16tags = tags.ReinterpretCast<int16_t>();

        for (int i = 0; i < 10; i++) {
            StateUpdate(states, lengths);
        }

        AscendC::DataCopy(cache[0 * state_size + (8 - shift_const) * STATE_BATCH_SIZE], states[0 * state_size], shift_const * STATE_BATCH_SIZE);
        AscendC::DataCopy(cache[0 * state_size], states[0 * state_size + shift_const * STATE_BATCH_SIZE], (8 - shift_const) * STATE_BATCH_SIZE); // s1 <<< 96

        AscendC::Add(i16cache[0 * state_size], i16cache[0 * state_size], i16states[0 * state_size], state_size); // s0 + s1 <<< 96
        AscendC::And(cache[1 * state_size], states[2 * state_size], states[3 * state_size], state_size); // s2 & s3
        AscendC::Add(i16tags, i16cache[0 * state_size], i16cache[1 * state_size], state_size); // tag = s0 + s1 <<< 96 + (s2 & s3)
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<uint16_t> xGm;
    AscendC::GlobalTensor<uint16_t> yGm;
    AscendC::GlobalTensor<uint16_t> zGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> updateBuffer, statesBuffer, initBuffer;
};

//  extern "C" __global__ __aicore__ void ascen(GM_ADDR x, GM_ADDR y, GM_ADDR z, size_t m_batch_size)
extern "C" __global__ __aicore__ void ascen(
     __gm__ uint8_t* x, 
     __gm__ uint8_t* y, 
     __gm__ uint8_t* z, 
    size_t m_batch_size)
{
    KernelAdd op;
    op.Init(x, y, z, m_batch_size);
    op.Process(m_batch_size);
}

namespace vllm_ascend {
    
extern void morus_encrypt_do_impl(int64_t threadnum, void* enc_stream, uint16_t* input_aligned_ptr, uint16_t* key_nounce_ptr, uint16_t* output_aligned_ptr, int64_t m_batch_size)
{
    ascen<<<threadnum, nullptr, enc_stream>>>(
        input_aligned_ptr,
        key_nounce_ptr,
        output_aligned_ptr,
        m_batch_size);
}
} //namespace vllm_ascend

 