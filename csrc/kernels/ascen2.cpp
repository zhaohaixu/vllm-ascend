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
//#define M_BATCH_SIZE 100
#define STATE_BATCH_SIZE 384
#define EXE_TIME 1

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR z, size_t keystream_size) {
        size_t M_BATCH_SIZE = keystream_size / (32 * STATE_BATCH_SIZE);
        yGm.SetGlobalBuffer((__gm__ uint16_t *)y + 24 * STATE_BATCH_SIZE *  AscendC::GetBlockIdx(), 24 * STATE_BATCH_SIZE);
        zGm.SetGlobalBuffer((__gm__ uint16_t *)z + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE * M_BATCH_SIZE);
        pipe.InitBuffer(inQueueY, 1, 57 * STATE_BATCH_SIZE * sizeof(uint16_t) );
        pipe.InitBuffer(outQueueZ, 1, 32 * STATE_BATCH_SIZE * sizeof(uint16_t));
        pipe.InitBuffer(updateBuffer, 10 * STATE_BATCH_SIZE * sizeof(uint16_t));
        pipe.InitBuffer(keyBuffer, 16 * STATE_BATCH_SIZE * sizeof(uint16_t));
        //pipe.InitBuffer(initBuffer, 10* STATE_BATCH_SIZE * sizeof(uint16_t));
    }
    
    __aicore__ inline void Process(int keystream_size) {
        int m_batch_size = keystream_size / (32 * STATE_BATCH_SIZE);
        for (int r=0; r < m_batch_size; r++){
            CopyIn(r);
            Compute(r, m_batch_size);          
            CopyOut(r);
        }
    }
    
private:
    
    __aicore__ inline void CopyIn(int progress) {
        if(progress ==0){
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.AllocTensor<uint16_t>();
        AscendC::DataCopy(yLocal, yGm, 8 * STATE_BATCH_SIZE * 3);
         inQueueY.EnQue(yLocal);
        }
    }
    __aicore__ inline void Compute(int progress, int m_batch_size) {
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> key = keyBuffer.Get<uint16_t>();
        
       if(progress ==0){
        Initialization(key, yLocal);
       }
        // inQueueY.FreeTensor(yLocal);
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.AllocTensor<uint16_t>();
        // yLocal = inQueueY.AllocTensor<uint16_t>();
        //AscendC::LocalTensor<uint16_t> states = statesBuffer.Get<uint16_t>();
        //AscendC::DataCopy(xLocal, xGm[progress * 32 * STATE_BATCH_SIZE], 32 * STATE_BATCH_SIZE);
        //Initialization(states, yLocal, yLocal[8 * STATE_BATCH_SIZE]);
        Encryption(yLocal, zLocal);

        outQueueZ.EnQue<uint16_t>(zLocal);
        if(progress< m_batch_size-1){
            inQueueY.EnQue(yLocal);
        }
        else{
            inQueueY.FreeTensor(yLocal);
        }
    }
    __aicore__ inline void CopyOut(int progress) {
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.DeQue<uint16_t>();
       
        AscendC::DataCopy(zGm[32 * STATE_BATCH_SIZE* progress], zLocal, 32 * STATE_BATCH_SIZE);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void Batch_StateUpdate(AscendC::LocalTensor<uint16_t> states, const size_t batch) {
        int32_t state_size = STATE_BATCH_SIZE, shift_const = 2;
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<uint16_t> cache = updateBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16cache = cache.ReinterpretCast<int16_t>();
        // Round 1
        AscendC::Mul(i16states[37*STATE_BATCH_SIZE], i16states[((25)) * state_size], i16states[((15)) * state_size], batch*state_size);
        AscendC::ShiftLeft(cache, states[((25)) * state_size], (uint16_t) 8, batch*state_size);
        AscendC::ShiftRight(states[((25)) * state_size], states[((25)) * state_size], (uint16_t) 8, batch*state_size);
        AscendC::Add(i16states[((25)) * state_size], i16states[((25)) * state_size], i16cache, batch*state_size);
        AscendC::Add(i16states[37*STATE_BATCH_SIZE], i16states[37*STATE_BATCH_SIZE], i16states[((0)) * state_size], batch*state_size);
        AscendC::Add(i16states[37*STATE_BATCH_SIZE], i16states[37*STATE_BATCH_SIZE], i16states[((22)) * state_size], batch*state_size);
        AscendC::Or(cache, states[((17)) * state_size], states[((2)) * state_size], batch*state_size);
        AscendC::Add(i16states[((37)) * state_size], i16cache, i16states[((37)) * state_size], batch*state_size);

    }
    __aicore__ inline void Initialization(AscendC::LocalTensor<uint16_t> keys, AscendC::LocalTensor<uint16_t> states) {
        int32_t state_size = STATE_BATCH_SIZE * 8;
        //AscendC::LocalTensor<uint16_t> cache = initBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16keys = keys.ReinterpretCast<int16_t>();
        // AscendC::DataCopy(states[0 * state_size], ivs, state_size); // s0 = iv
        AscendC::DataCopy( keys,states[1 * state_size], 2 * state_size); // s1 = k
        //AscendC::Duplicate(states[2 * state_size], (uint16_t) 0xFFFF, state_size); // s2 = 1^128

        AscendC::Duplicate(states[3 * state_size + 0 * STATE_BATCH_SIZE], (uint16_t) 0x1, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 1 * STATE_BATCH_SIZE], (uint16_t) 0x2, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 2 * STATE_BATCH_SIZE], (uint16_t) 0x3, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 3 * STATE_BATCH_SIZE], (uint16_t) 0x4, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 4 * STATE_BATCH_SIZE], (uint16_t) 0x5, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 5 * STATE_BATCH_SIZE], (uint16_t) 0x6, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 6 * STATE_BATCH_SIZE], (uint16_t) 0x7, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 7 * STATE_BATCH_SIZE], (uint16_t) 0x8, STATE_BATCH_SIZE); // s3 = const0
        //AscendC::DataCopy(cache, states[3 * state_size], state_size); // const0 backup

        AscendC::Duplicate(states[3 * state_size + 8 * STATE_BATCH_SIZE], (uint16_t) 0x2, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 9 * STATE_BATCH_SIZE], (uint16_t) 0x3, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 10 * STATE_BATCH_SIZE], (uint16_t) 0x4, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 11 * STATE_BATCH_SIZE], (uint16_t) 0x5, STATE_BATCH_SIZE);
        AscendC::Duplicate(states[3 * state_size + 12 * STATE_BATCH_SIZE], (uint16_t) 0x6, STATE_BATCH_SIZE);

        for(int i=0 ; i<3; i++){
        Batch_StateUpdate(states,10);
        
        Batch_StateUpdate(states[10*STATE_BATCH_SIZE],10);
        AscendC::DataCopy(states, states[20* STATE_BATCH_SIZE], 37*STATE_BATCH_SIZE);
        
        Batch_StateUpdate(states,10);
        
        Batch_StateUpdate(states[10*STATE_BATCH_SIZE],7);
        AscendC::DataCopy(states, states[17* STATE_BATCH_SIZE], 37*STATE_BATCH_SIZE);
    
        }
        

        
        //Key feed-forward: S[0-8]+=k1, S[12-27] += k0||k1, S[29-36] += k0
        AscendC::Add(i16states[0 * state_size], i16states[0 * state_size], i16keys[1*state_size], state_size); // s1 += k
        AscendC::Add(i16states[1 * state_size + 4 * STATE_BATCH_SIZE], i16states[1 * state_size + 4 * STATE_BATCH_SIZE], i16keys, 2 * state_size);
        AscendC::Add(i16states[3 * state_size + 5 * STATE_BATCH_SIZE], i16states[3 * state_size + 5 * STATE_BATCH_SIZE], i16keys[0*state_size], state_size);
    } 
    __aicore__ inline void Encryption(AscendC::LocalTensor<uint16_t> states, AscendC::LocalTensor<uint16_t> c) {
        int state_size = STATE_BATCH_SIZE;
        AscendC::LocalTensor<int16_t> i16states = states.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<uint16_t> cache = keyBuffer.Get<uint16_t>();
        AscendC::LocalTensor<int16_t> i16cache = cache.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16c = c.ReinterpretCast<int16_t>();
       
        //i=1
        AscendC::Add(i16cache, i16states[((1)) * STATE_BATCH_SIZE], i16states[((18)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((21)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c, i16cache, i16states[((32)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c, i16cache, i16m, 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states, 4);

        //i=2
        AscendC::Add(i16cache, i16states[((5)) * STATE_BATCH_SIZE], i16states[((22)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((25)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[4 * STATE_BATCH_SIZE], i16cache, i16states[((36)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[4 * STATE_BATCH_SIZE], i16cache, i16m[4* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[4*state_size], 4);
        
        //i=3
        AscendC::Add(i16cache, i16states[((9)) * STATE_BATCH_SIZE], i16states[((26)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((29)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[8* STATE_BATCH_SIZE], i16cache, i16states[((40)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[8* STATE_BATCH_SIZE], i16cache, i16m[8* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[8*state_size], 4);
        //i=4
        AscendC::Add(i16cache, i16states[((13)) * STATE_BATCH_SIZE], i16states[((30)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((33)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[12 * STATE_BATCH_SIZE], i16cache, i16states[((44)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[12 * STATE_BATCH_SIZE], i16cache, i16m[12* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[12*state_size], 4);

        //i=5
        AscendC::Add(i16cache, i16states[((17)) * STATE_BATCH_SIZE], i16states[((34)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((37)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[16 * STATE_BATCH_SIZE], i16cache, i16states[((48)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[16 * STATE_BATCH_SIZE], i16cache, i16m[16* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[16*state_size], 4);
         AscendC::DataCopy(states, states[20* state_size], 37*state_size);
        //i=6
        AscendC::Add(i16cache, i16states[((1)) * STATE_BATCH_SIZE], i16states[((18)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((21)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[20 * STATE_BATCH_SIZE], i16cache, i16states[((32)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[20 * STATE_BATCH_SIZE], i16cache, i16m[20* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states, 4);
        

         //i=7
        AscendC::Add(i16cache, i16states[((5)) * STATE_BATCH_SIZE], i16states[((22)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((25)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[24 * STATE_BATCH_SIZE], i16cache, i16states[((36)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[24 * STATE_BATCH_SIZE], i16cache, i16m[24* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[4*state_size], 4);

        //i=8
         AscendC::Add(i16cache, i16states[((9)) * STATE_BATCH_SIZE], i16states[((26)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16cache, i16cache, i16states[((29)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        AscendC::Add(i16c[28 * STATE_BATCH_SIZE], i16cache, i16states[((40)) * STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        //AscendC::Add(i16c[28 * STATE_BATCH_SIZE], i16cache, i16m[28* STATE_BATCH_SIZE], 4*STATE_BATCH_SIZE);
        Batch_StateUpdate(states[8*state_size], 9);
        
         AscendC::DataCopy(states, states[17* state_size], 37*state_size);
        
    }
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<uint16_t> yGm;
    AscendC::GlobalTensor<uint16_t> zGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> updateBuffer, keyBuffer;
};

#define STATE_BATCH_SIZE2 384
class KernelEnc {
public:
    __aicore__ inline KernelEnc() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, size_t input_size) {
        size_t M_BATCH_SIZE = input_size / (32 * STATE_BATCH_SIZE2);
        xGm.SetGlobalBuffer((__gm__ uint16_t *)x + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        yGm.SetGlobalBuffer((__gm__ uint16_t *)y + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        zGm.SetGlobalBuffer((__gm__ uint16_t *)z + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        pipe.InitBuffer(inQueueX, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t));
        pipe.InitBuffer(inQueueY, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t) );
        pipe.InitBuffer(outQueueZ, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t));
    }
    __aicore__ inline void Process(int input_size) {
        size_t m_batch_size = input_size / (32 * STATE_BATCH_SIZE2);
        for (int r=0; r < m_batch_size; r++){
            CopyIn(r);
            Compute(r);          
            CopyOut(r);
        }
    }
    
private:
    
    __aicore__ inline void CopyIn(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.AllocTensor<uint16_t>(); 
        AscendC::DataCopy(xLocal, xGm[32 * STATE_BATCH_SIZE2 * progress], 32 * STATE_BATCH_SIZE2);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.AllocTensor<uint16_t>(); 
        AscendC::DataCopy(yLocal, yGm[32 * STATE_BATCH_SIZE2 * progress], 32 * STATE_BATCH_SIZE2);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.DeQue<uint16_t>();
        
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.AllocTensor<uint16_t>();
        AscendC::LocalTensor<int16_t> i16x = xLocal.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16y = yLocal.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16z = zLocal.ReinterpretCast<int16_t>();
        AscendC::Add(i16z, i16x, i16y, 32 * STATE_BATCH_SIZE2);
        outQueueZ.EnQue<uint16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int progress) {
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.DeQue<uint16_t>();
       
        AscendC::DataCopy(zGm[32 * STATE_BATCH_SIZE2 * progress], zLocal, 32 * STATE_BATCH_SIZE2);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<uint16_t> xGm;
    AscendC::GlobalTensor<uint16_t> yGm;
    AscendC::GlobalTensor<uint16_t> zGm;
};

class KernelDec {
public:
    __aicore__ inline KernelDec() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, size_t input_size) {
        size_t M_BATCH_SIZE = input_size / (32 * STATE_BATCH_SIZE2);
        xGm.SetGlobalBuffer((__gm__ uint16_t *)x + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        yGm.SetGlobalBuffer((__gm__ uint16_t *)y + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        zGm.SetGlobalBuffer((__gm__ uint16_t *)z + AscendC::GetBlockIdx() * 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE, 32 * STATE_BATCH_SIZE2 * M_BATCH_SIZE);
        pipe.InitBuffer(inQueueX, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t));
        pipe.InitBuffer(inQueueY, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t) );
        pipe.InitBuffer(outQueueZ, 1, 32 * STATE_BATCH_SIZE2 * sizeof(uint16_t));
    }
    __aicore__ inline void Process(int input_size) {
        size_t m_batch_size = input_size / (32 * STATE_BATCH_SIZE2);
        for (int r=0; r < m_batch_size; r++){
            CopyIn(r);
            Compute(r);          
            CopyOut(r);
        }
    }
    
private:
    
    __aicore__ inline void CopyIn(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.AllocTensor<uint16_t>(); 
        AscendC::DataCopy(xLocal, xGm[32 * STATE_BATCH_SIZE2 * progress], 32 * STATE_BATCH_SIZE2);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.AllocTensor<uint16_t>(); 
        AscendC::DataCopy(yLocal, yGm[32 * STATE_BATCH_SIZE2 * progress], 32 * STATE_BATCH_SIZE2);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int progress) {
        AscendC::LocalTensor<uint16_t> xLocal = inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> yLocal = inQueueY.DeQue<uint16_t>();
        
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.AllocTensor<uint16_t>();
        AscendC::LocalTensor<int16_t> i16x = xLocal.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16y = yLocal.ReinterpretCast<int16_t>();
        AscendC::LocalTensor<int16_t> i16z = zLocal.ReinterpretCast<int16_t>();
        AscendC::Sub(i16z, i16x, i16y, 32 * STATE_BATCH_SIZE2);
        outQueueZ.EnQue<uint16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int progress) {
        AscendC::LocalTensor<uint16_t> zLocal = outQueueZ.DeQue<uint16_t>();
       
        AscendC::DataCopy(zGm[32 * STATE_BATCH_SIZE2 * progress], zLocal, 32 * STATE_BATCH_SIZE2);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<uint16_t> xGm;
    AscendC::GlobalTensor<uint16_t> yGm;
    AscendC::GlobalTensor<uint16_t> zGm;
};

extern "C" __global__ __aicore__ void ascen_keystream(GM_ADDR keynonce, GM_ADDR keystream, size_t keystream_size)
{
    KernelAdd op;
    op.Init(keynonce, keystream, keystream_size);
    op.Process(keystream_size);
}
extern "C" __global__ __aicore__ void ascen_encrypt(GM_ADDR plaintext, GM_ADDR keystream, GM_ADDR ciphertext, size_t input_size)
{   
    KernelEnc op;
    op.Init(plaintext, keystream, ciphertext, input_size);
    op.Process(input_size);
}
extern "C" __global__ __aicore__ void ascen_decrypt(GM_ADDR ciphertext, GM_ADDR keystream, GM_ADDR plaintext, size_t input_size)
{   
    //x is ciphertext, y is keystream
    KernelEnc op;
    op.Init(ciphertext, keystream, plaintext, input_size);
    op.Process(input_size);
}


namespace vllm_ascend {

extern void ascen_keystream_impl(
    uint32_t blockDim,
    void *stream,
    void* keynonce,
    void* keystream,
    uint32_t keystream_size)
{
    ascen_keystream<<<blockDim, nullptr, stream>>>(
        keynonce,
        keystream,
        keystream_size
    );
}

extern void ascen_encrypt_impl(
    int64_t threadnum,
    void *stream,
    void* plaintext,   //输入
    void* keystream,   //密钥流
    void* ciphertext,  //输出
    uint32_t input_size)
{
    // threadnum 参与决定 block/threads
    int blocks = threadnum;
    int threads = 1;
    ascen_encrypt<<<32, nullptr, stream>>>(
        plaintext,
        keystream,
        ciphertext,
        input_size
    );
}

extern void ascen_decrypt_impl(
    int64_t threadnum,
    void *stream,
    void* plaintext,
    void* keystream,
    void* ciphertext,
    uint32_t input_size)
{
    int blocks = static_cast<int>(threadnum);
    int threads = 1;
    ascen_decrypt<<<32, nullptr, stream>>>(
        ciphertext,
        keystream,
        plaintext,
        input_size
    );
}

} // namespace vllm_ascend
