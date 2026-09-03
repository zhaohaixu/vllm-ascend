#include "kernel_operator.h"
using namespace AscendC;

constexpr int SM4_BLOCK_SIZE = 16;
constexpr int SM4_NR = 32;
constexpr int SM4_RK_WORDS = 32;
constexpr int SM4_RK_BYTES = SM4_RK_WORDS * sizeof(uint32_t); // 128
constexpr int SM4_RK_PAD_BYTES = 128;                         // 128 本身已 32B 对齐
constexpr int VEC_BLOCKS       = 64 * 4;
constexpr uint32_t MAX_BLOCKS_PER_CALL = VEC_BLOCKS;

// SM4 常量表
static const uint8_t SM4_SBOX[256] = {
    0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
    0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
    0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
    0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
    0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
    0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
    0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
    0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
    0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
    0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
    0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
    0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
    0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
    0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
    0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
    0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48
};


class KernelSM4CTR {
public:

    LocalTensor<uint32_t> Sbox;
    __aicore__ inline KernelSM4CTR() {}

    __aicore__ inline void Init(__gm__ uint8_t* rk128,
                                __gm__ uint8_t* in,
                                __gm__ uint8_t* out,
                                uint32_t dataSize)
    {
        rkGlobal.SetGlobalBuffer((__gm__ uint8_t*)rk128);
        inGlobal.SetGlobalBuffer((__gm__ uint8_t*)in);
        outGlobal.SetGlobalBuffer((__gm__ uint8_t*)out);

        this->dataSize = dataSize;
        //  这里按完整 block 处理，要求 dataSize 是 16 的整数倍
        this->totalBlocks = dataSize / SM4_BLOCK_SIZE;

        pipe.InitBuffer(rkBytesQ, 1, SM4_RK_PAD_BYTES);                         // 128B
        pipe.InitBuffer(rkWordsQ, 1, SM4_RK_WORDS * sizeof(uint32_t));          // 32 words
        pipe.InitBuffer(inQ,      1, MAX_BLOCKS_PER_CALL * SM4_BLOCK_SIZE + 64);
        pipe.InitBuffer(outQ,     1, MAX_BLOCKS_PER_CALL * SM4_BLOCK_SIZE + 64);
        pipe.InitBuffer(scratchQ, 1, VEC_BLOCKS * sizeof(uint32_t) * 4);   // bytes        state
        pipe.InitBuffer(rkxQ,     1, VEC_BLOCKS * sizeof(uint32_t) * 4);  //rk * 8  rkAll
        pipe.InitBuffer(btmpQ,    1, VEC_BLOCKS * sizeof(uint32_t) * 4);                //存储中间数据 b0~b3 
        pipe.InitBuffer(SboxBuf, 256 * sizeof(uint32_t));
        pipe.InitBuffer(RotateQ,  1, VEC_BLOCKS * sizeof(uint32_t) * 4);
        pipe.InitBuffer(tLQ,      1, VEC_BLOCKS * sizeof(uint32_t));


        LocalTensor<uint32_t>SAll = SboxBuf.Get<uint32_t>();
        Sbox = SAll;
        for (int i=0; i < 256 ;++i){
            Sbox(i) = (uint32_t)SM4_SBOX[i];
        }
    }

    __aicore__ inline void Process()
    {   
        uint32_t blockId  = GetBlockIdx();   //当前核的编号
        uint32_t blockNum = GetBlockNum();   //总共启动核数

        // 先把 totalBlocks 按核均分
        uint32_t blocksPerCore = (totalBlocks + blockNum - 1) / blockNum;

        uint32_t coreStart = blockId * blocksPerCore;
        uint32_t coreEnd   = min(coreStart + blocksPerCore, totalBlocks);

        if (coreStart >= coreEnd) {
            return;
        }

        uint32_t blocksToProcess = min(MAX_BLOCKS_PER_CALL, coreEnd - coreStart);
        // 每个核内部再按 tile 循环
        for (uint32_t tileStart = coreStart;tileStart < coreEnd; tileStart += blocksToProcess) {
            blocksToProcess = min(MAX_BLOCKS_PER_CALL, coreEnd - tileStart);
            CopyIn (tileStart, blocksToProcess);
            Compute(blocksToProcess);
            CopyOut(tileStart, blocksToProcess);
        }
    }


private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  rkBytesQ;
    TQue<TPosition::VECIN, 1>  rkWordsQ;
    TQue<TPosition::VECIN, 1>  inQ;
    TQue<TPosition::VECOUT, 1> outQ;
    TQue<TPosition::VECCALC, 1> scratchQ;
    TQue<TPosition::VECCALC, 1> rkxQ;
    TQue<TPosition::VECCALC, 1> btmpQ;
    TBuf<TPosition::VECCALC> SboxBuf;
    TQue<TPosition::VECCALC, 1> RotateQ;
    TQue<TPosition::VECCALC, 1> tLQ;


    GlobalTensor<uint8_t> rkGlobal;
    GlobalTensor<uint8_t> inGlobal;
    GlobalTensor<uint8_t> outGlobal;

    uint32_t dataSize{0};
    uint32_t totalBlocks{0};

private:
    __aicore__ inline uint32_t Rotl32(uint32_t x, uint32_t n)
    {
        return (x << n) | (x >> (32 - n));
    }

    __aicore__ inline uint32_t Tau(uint32_t x)
    {
        uint32_t b0 = SM4_SBOX[(x >> 24) & 0xFF];
        uint32_t b1 = SM4_SBOX[(x >> 16) & 0xFF];
        uint32_t b2 = SM4_SBOX[(x >> 8)  & 0xFF];
        uint32_t b3 = SM4_SBOX[(x >> 0)  & 0xFF];
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }

    // SM4 轮函数里的线性变换 L
    __aicore__ inline uint32_t L(uint32_t x)
    {
        return x ^ Rotl32(x, 2) ^ Rotl32(x, 10) ^ Rotl32(x, 18) ^ Rotl32(x, 24);
    }

    __aicore__ inline uint32_t T(uint32_t x)
    {
        return L(Tau(x));
    }

    __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)
    {
        // 1) Copy round keys bytes (128B)
        LocalTensor<uint8_t> rkBytesLT = rkBytesQ.AllocTensor<uint8_t>();
        DataCopy(rkBytesLT, rkGlobal, SM4_RK_PAD_BYTES);

        AscendC::PipeBarrier<PIPE_MTE2>();   // 先等 rkBytesLT 搬完

        // Pack to 32 big-endian uint32 words
        LocalTensor<uint32_t> rkWordsLT = rkWordsQ.AllocTensor<uint32_t>();
        #pragma unroll
        for (int w = 0; w < SM4_RK_WORDS; ++w) {
            int b = w * 4;
            uint32_t b0 = rkBytesLT(b + 0);
            uint32_t b1 = rkBytesLT(b + 1);
            uint32_t b2 = rkBytesLT(b + 2);
            uint32_t b3 = rkBytesLT(b + 3);
            rkWordsLT(w) = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        }

        // 3) Copy input blocks
        LocalTensor<uint8_t> inLocal = inQ.AllocTensor<uint8_t>();
        uint32_t inputOffset = startBlock * SM4_BLOCK_SIZE;
        uint32_t inputSize = blocksToProcess * SM4_BLOCK_SIZE;
        DataCopy(inLocal, inGlobal[inputOffset], inputSize);

        rkBytesQ.EnQue(rkBytesLT);
        rkWordsQ.EnQue(rkWordsLT);
        inQ.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t blocksToProcess)
    {
        LocalTensor<uint32_t> rkWordsLT = rkWordsQ.DeQue<uint32_t>();
        LocalTensor<uint8_t>  inLocal   = inQ.DeQue<uint8_t>();
        LocalTensor<uint8_t>  outLocal  = outQ.AllocTensor<uint8_t>();
        LocalTensor<uint8_t>  rkBytesLT = rkBytesQ.DeQue<uint8_t>(); // 仅为后面释放

        AscendC::PipeBarrier<PIPE_MTE2>();  
            
        LocalTensor<uint32_t> statelocal = scratchQ.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> state0 = statelocal;
        LocalTensor<uint32_t> state1 = statelocal[VEC_BLOCKS ];
        LocalTensor<uint32_t> state2 = statelocal[VEC_BLOCKS * 2];
        LocalTensor<uint32_t> state3 = statelocal[VEC_BLOCKS * 3];
        
        for (uint32_t b = 0; b < blocksToProcess; b += VEC_BLOCKS) {
        // uint32_t curBlocks = min((uint32_t)VEC_BLOCKS, blocksToProcess - b);
            #pragma unroll
            for (uint32_t lane = 0; lane < VEC_BLOCKS; ++lane) {
                uint32_t base = (b + lane) << 4;

                uint32_t t0 = (uint32_t)inLocal(base + 0) << 24 |
                            (uint32_t)inLocal(base + 1) << 16 |
                            (uint32_t)inLocal(base + 2) << 8  |
                            (uint32_t)inLocal(base + 3);

                uint32_t t1 = (uint32_t)inLocal(base + 4) << 24 |
                            (uint32_t)inLocal(base + 5) << 16 |
                            (uint32_t)inLocal(base + 6) << 8  |
                            (uint32_t)inLocal(base + 7);

                uint32_t t2 = (uint32_t)inLocal(base + 8) << 24 |
                            (uint32_t)inLocal(base + 9) << 16 |
                            (uint32_t)inLocal(base + 10) << 8 |
                            (uint32_t)inLocal(base + 11);

                uint32_t t3 = (uint32_t)inLocal(base + 12) << 24 |
                            (uint32_t)inLocal(base + 13) << 16 |
                            (uint32_t)inLocal(base + 14) << 8 |
                            (uint32_t)inLocal(base + 15);

                    state0(lane) = t0;
                    state1(lane) = t1;
                    state2(lane) = t2;
                    state3(lane) = t3;
                }
                SM4_Encrypt_Vector(outLocal, state0, state1, state2, state3, rkWordsLT, b);
            }



        outQ.EnQue<uint8_t>(outLocal);
        rkWordsQ.FreeTensor(rkWordsLT);
        inQ.FreeTensor(inLocal);
        rkBytesQ.FreeTensor(rkBytesLT);
        scratchQ.FreeTensor(statelocal);
    }

    __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
    {
        LocalTensor<uint8_t> outLocal = outQ.DeQue<uint8_t>();

        uint32_t outputOffset = startBlock * SM4_BLOCK_SIZE;
        uint32_t outputSize = blocksToProcess * SM4_BLOCK_SIZE;
        AscendC::PipeBarrier<PIPE_MTE3>();
        DataCopy(outGlobal[outputOffset], outLocal, outputSize);

        outQ.FreeTensor(outLocal);
    }


    __aicore__ inline void TableTransform(LocalTensor<uint32_t> src,
                                          LocalTensor<uint32_t> b0,
                                          LocalTensor<uint32_t> b1,
                                          LocalTensor<uint32_t> b2,
                                          LocalTensor<uint32_t> b3)
    {
        // b0 = ((src >> 24) & 0xFF) << 2
        AscendC::ShiftRight(b0, src, (uint32_t)24, VEC_BLOCKS);

        // b1 = ((src << 8) >> 24) << 2
        AscendC::ShiftLeft (b1, src, (uint32_t)8 , VEC_BLOCKS);
        AscendC::ShiftRight(b1, b1 , (uint32_t)24, VEC_BLOCKS);

        // b2 = ((src << 16) >> 24) << 2
        AscendC::ShiftLeft (b2, src, (uint32_t)16, VEC_BLOCKS);
        AscendC::ShiftRight(b2, b2 , (uint32_t)24, VEC_BLOCKS);

        // b3 = ((src << 24) >> 24) << 2
        AscendC::ShiftLeft (b3, src, (uint32_t)24, VEC_BLOCKS);
        AscendC::ShiftRight(b3, b3 , (uint32_t)24, VEC_BLOCKS);

        // Gather 的 offset 单位是 bytes，uint32_t 表项宽度 4B，因此乘 4
        AscendC::ShiftLeft(b0, b0, (uint32_t)2, VEC_BLOCKS);
        AscendC::ShiftLeft(b1, b1, (uint32_t)2, VEC_BLOCKS);
        AscendC::ShiftLeft(b2, b2, (uint32_t)2, VEC_BLOCKS);
        AscendC::ShiftLeft(b3, b3, (uint32_t)2, VEC_BLOCKS);

        AscendC::Gather(b0, Sbox, b0, (uint32_t)0, VEC_BLOCKS);
        AscendC::Gather(b1, Sbox, b1, (uint32_t)0, VEC_BLOCKS);
        AscendC::Gather(b2, Sbox, b2, (uint32_t)0, VEC_BLOCKS);
        AscendC::Gather(b3, Sbox, b3, (uint32_t)0, VEC_BLOCKS);

        AscendC::ShiftLeft(b0, b0, (uint32_t)24, VEC_BLOCKS);
        AscendC::ShiftLeft(b1, b1, (uint32_t)16, VEC_BLOCKS);
        AscendC::ShiftLeft(b2, b2, (uint32_t)8 , VEC_BLOCKS);

        LocalTensor<uint16_t> b0_u16 = b0.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b1_u16 = b1.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b2_u16 = b2.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b3_u16 = b3.ReinterpretCast<uint16_t>();

        LocalTensor<uint32_t> tLocal =  tLQ.AllocTensor<uint32_t>();
        LocalTensor<uint16_t> tLocal_u16 = tLocal.ReinterpretCast<uint16_t>();

        //uint16_t
        AscendC::Or(b0_u16, b0_u16, b1_u16, VEC_BLOCKS * 2);
        AscendC::Or(b0_u16, b0_u16, b2_u16, VEC_BLOCKS * 2); 
        AscendC::Or(tLocal_u16, b0_u16, b3_u16, VEC_BLOCKS * 2);

        //循环左移
        LocalTensor<uint32_t> rotateAll =  RotateQ.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> rlocal0 = rotateAll;
        LocalTensor<uint32_t> rlocal1 = rotateAll[VEC_BLOCKS];
        LocalTensor<uint32_t> rlocal2 = rotateAll[VEC_BLOCKS * 2];
        LocalTensor<uint32_t> rlocal3 = rotateAll[VEC_BLOCKS * 3];

        LocalTensor<uint16_t> rlocal0_u16 = rlocal0.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rlocal1_u16 = rlocal1.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rlocal2_u16 = rlocal2.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rlocal3_u16 = rlocal3.ReinterpretCast<uint16_t>();

        AscendC::ShiftLeft(b0, tLocal, (uint32_t)2, VEC_BLOCKS);
        AscendC::ShiftRight(rlocal0, tLocal, (uint32_t)30, VEC_BLOCKS);

        AscendC::ShiftLeft(b1, tLocal, (uint32_t)10, VEC_BLOCKS);
        AscendC::ShiftRight(rlocal1, tLocal, (uint32_t)22, VEC_BLOCKS);

        AscendC::ShiftLeft(b2, tLocal, (uint32_t)18, VEC_BLOCKS);
        AscendC::ShiftRight(rlocal2, tLocal, (uint32_t)14, VEC_BLOCKS);

        AscendC::ShiftLeft(b3, tLocal, (uint32_t)24, VEC_BLOCKS);
        AscendC::ShiftRight(rlocal3, tLocal, (uint32_t)8, VEC_BLOCKS);

        AscendC::Or(rlocal0_u16, b0_u16, rlocal0_u16, VEC_BLOCKS * 2);
        AscendC::Or(rlocal1_u16, b1_u16, rlocal1_u16, VEC_BLOCKS * 2);
        AscendC::Or(rlocal2_u16, b2_u16, rlocal2_u16, VEC_BLOCKS * 2);
        AscendC::Or(rlocal3_u16, b3_u16, rlocal3_u16, VEC_BLOCKS * 2);

        AscendC::Xor(b0_u16,  tLocal_u16, rlocal0_u16, VEC_BLOCKS * 2);
        AscendC::Xor(b1_u16,  b0_u16    , rlocal1_u16, VEC_BLOCKS * 2);
        AscendC::Xor(b2_u16,  b1_u16    , rlocal2_u16, VEC_BLOCKS * 2);
        AscendC::Xor(tLocal_u16, b2_u16 , rlocal3_u16, VEC_BLOCKS * 2);

        AscendC::ShiftLeft(src, tLocal, (uint32_t)0, VEC_BLOCKS);

        RotateQ.FreeTensor(rotateAll);
        tLQ.FreeTensor(tLocal);

    }


 __aicore__ inline void  SM4_Encrypt_Vector(LocalTensor<uint8_t> outLocal,
                           LocalTensor<uint32_t> state0,
                           LocalTensor<uint32_t> state1,
                           LocalTensor<uint32_t> state2,
                           LocalTensor<uint32_t> state3,
                           LocalTensor<uint32_t> rkWordsLT,
                           uint32_t batchStart
                        )
{      
        LocalTensor<uint32_t> rkAll =  rkxQ.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> rkq0 = rkAll;
        LocalTensor<uint32_t> rkq1 = rkAll[VEC_BLOCKS];
        LocalTensor<uint32_t> rkq2 = rkAll[VEC_BLOCKS * 2];
        LocalTensor<uint32_t> rkq3 = rkAll[VEC_BLOCKS * 3];

        LocalTensor<uint16_t> state0_u16 = state0.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> state1_u16 = state1.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> state2_u16 = state2.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> state3_u16 = state3.ReinterpretCast<uint16_t>();

        LocalTensor<uint16_t> rkq0_u16 = rkq0.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rkq1_u16 = rkq1.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rkq2_u16 = rkq2.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> rkq3_u16 = rkq3.ReinterpretCast<uint16_t>();

        LocalTensor<uint32_t> blocal = btmpQ.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> b0 = blocal;
        LocalTensor<uint32_t> b1 = blocal[VEC_BLOCKS];
        LocalTensor<uint32_t> b2 = blocal[VEC_BLOCKS * 2];
        LocalTensor<uint32_t> b3 = blocal[VEC_BLOCKS * 3];

        LocalTensor<uint16_t> b0_u16 = b0.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b1_u16 = b1.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b2_u16 = b2.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> b3_u16 = b3.ReinterpretCast<uint16_t>();

        for (int r = 0; r <SM4_NR; r+=4) {        
            AscendC::Duplicate(rkq0 , (uint32_t)rkWordsLT(r + 0), VEC_BLOCKS);
            AscendC::Xor(rkq1_u16 , state2_u16 , state3_u16 , VEC_BLOCKS * 2);
            AscendC::Xor(b1_u16   , rkq1_u16   , rkq0_u16   , VEC_BLOCKS * 2);
            AscendC::Xor(rkq0_u16 , b1_u16     , state1_u16 , VEC_BLOCKS * 2);
            TableTransform(rkq0, b0, b1, b2, b3);
            AscendC::Xor(b0_u16, state0_u16, rkq0_u16, VEC_BLOCKS * 2);
            // AscendC::Copy(state0, b0, mask, repeatTime, params);
            AscendC::ShiftLeft(state0, b0, (uint32_t)0, VEC_BLOCKS);
            
            AscendC::Xor(b1_u16, rkq1_u16, state0_u16, VEC_BLOCKS * 2);
            AscendC::Duplicate(rkq1 , (uint32_t)rkWordsLT(r + 1), VEC_BLOCKS);
            AscendC::Xor(rkq0_u16, rkq1_u16, b1_u16, VEC_BLOCKS * 2);
            TableTransform(rkq0, b0, b1, b2, b3);
            AscendC::Xor(b0_u16, state1_u16, rkq0_u16, VEC_BLOCKS * 2);
            // AscendC::Copy(state1, b0, mask, repeatTime, params);
            AscendC::ShiftLeft(state1, b0, (uint32_t)0, VEC_BLOCKS);

            AscendC::Duplicate(rkq2 , (uint32_t)rkWordsLT(r + 2), VEC_BLOCKS);
            AscendC::Xor(rkq3_u16 , state0_u16 , state1_u16 , VEC_BLOCKS * 2);
            AscendC::Xor(b0_u16  , rkq2_u16, state3_u16, VEC_BLOCKS * 2);
            AscendC::Xor(rkq0_u16, b0_u16  , rkq3_u16  , VEC_BLOCKS * 2);
            TableTransform(rkq0, b0, b1, b2, b3);
            AscendC::Xor(b0_u16, state2_u16, rkq0_u16, VEC_BLOCKS * 2);
            // AscendC::Copy(state2, b0, mask, repeatTime, params);
            AscendC::ShiftLeft(state2, b0, (uint32_t)0, VEC_BLOCKS);

            AscendC::Xor(b1_u16, rkq3_u16, state2_u16, VEC_BLOCKS * 2);
            AscendC::Duplicate(rkq3 , (uint32_t)rkWordsLT(r + 3), VEC_BLOCKS);
            AscendC::Xor(rkq0_u16, rkq3_u16, b1_u16, VEC_BLOCKS * 2);
            TableTransform(rkq0, b0, b1, b2, b3);
            AscendC::Xor(b0_u16, state3_u16, rkq0_u16, VEC_BLOCKS * 2);
            // AscendC::Copy(state3, b0, mask, repeatTime, params);
            AscendC::ShiftLeft(state3, b0, (uint32_t)0, VEC_BLOCKS);
        }

        //输出 反序
        for (uint32_t lane = 0; lane < VEC_BLOCKS; ++lane){
            // 拆成字节并写回 outLocal
            uint32_t base = (batchStart + lane) << 4;   // *16
            uint32_t t0 = state3(lane);
            uint32_t t1 = state2(lane);
            uint32_t t2 = state1(lane);
            uint32_t t3 = state0(lane);

            outLocal(base + 0)  = uint8_t(t0 >> 24);
            outLocal(base + 1)  = uint8_t(t0 >> 16);
            outLocal(base + 2)  = uint8_t(t0 >> 8);
            outLocal(base + 3)  = uint8_t(t0);

            outLocal(base + 4)  = uint8_t(t1 >> 24);
            outLocal(base + 5)  = uint8_t(t1 >> 16);
            outLocal(base + 6)  = uint8_t(t1 >> 8);
            outLocal(base + 7)  = uint8_t(t1);

            outLocal(base + 8)  = uint8_t(t2 >> 24);
            outLocal(base + 9)  = uint8_t(t2 >> 16);
            outLocal(base + 10) = uint8_t(t2 >> 8);
            outLocal(base + 11) = uint8_t(t2);

            outLocal(base + 12) = uint8_t(t3 >> 24);
            outLocal(base + 13) = uint8_t(t3 >> 16);
            outLocal(base + 14) = uint8_t(t3 >> 8);
            outLocal(base + 15) = uint8_t(t3);
        }

        btmpQ.FreeTensor(blocal); 
        rkxQ.FreeTensor(rkAll);
}



    __aicore__ inline void EncryptBlock_SM4(
        LocalTensor<uint8_t> outTensor,      // 16B sub-tensor
        LocalTensor<uint8_t> inTensor,       // 16B sub-tensor
        LocalTensor<uint32_t> rkWordsLT      // 32 x uint32 round keys (big-endian packed)
    ) {
        // 读入 128-bit 明文，按大端打成 4 个 32-bit 字
        uint32_t x0 = (uint32_t(inTensor(0))  << 24) | (uint32_t(inTensor(1))  << 16) |
                      (uint32_t(inTensor(2))  << 8)  |  uint32_t(inTensor(3));
        uint32_t x1 = (uint32_t(inTensor(4))  << 24) | (uint32_t(inTensor(5))  << 16) |
                      (uint32_t(inTensor(6))  << 8)  |  uint32_t(inTensor(7));
        uint32_t x2 = (uint32_t(inTensor(8))  << 24) | (uint32_t(inTensor(9))  << 16) |
                      (uint32_t(inTensor(10)) << 8)  |  uint32_t(inTensor(11));
        uint32_t x3 = (uint32_t(inTensor(12)) << 24) | (uint32_t(inTensor(13)) << 16) |
                      (uint32_t(inTensor(14)) << 8)  |  uint32_t(inTensor(15));

        #pragma unroll
        for (int r = 0; r < SM4_NR; r += 4) {
            x0 = x0 ^ T(x1 ^ x2 ^ x3 ^ rkWordsLT(r + 0));
            x1 = x1 ^ T(x2 ^ x3 ^ x0 ^ rkWordsLT(r + 1));
            x2 = x2 ^ T(x3 ^ x0 ^ x1 ^ rkWordsLT(r + 2));
            x3 = x3 ^ T(x0 ^ x1 ^ x2 ^ rkWordsLT(r + 3));
        }

        // SM4 输出顺序为 X35 X34 X33 X32，即最终 4 字反序输出
        outTensor(0)  = uint8_t(x3 >> 24); outTensor(1)  = uint8_t(x3 >> 16);
        outTensor(2)  = uint8_t(x3 >> 8);  outTensor(3)  = uint8_t(x3);
        outTensor(4)  = uint8_t(x2 >> 24); outTensor(5)  = uint8_t(x2 >> 16);
        outTensor(6)  = uint8_t(x2 >> 8);  outTensor(7)  = uint8_t(x2);
        outTensor(8)  = uint8_t(x1 >> 24); outTensor(9)  = uint8_t(x1 >> 16);
        outTensor(10) = uint8_t(x1 >> 8);  outTensor(11) = uint8_t(x1);
        outTensor(12) = uint8_t(x0 >> 24); outTensor(13) = uint8_t(x0 >> 16);
        outTensor(14) = uint8_t(x0 >> 8);  outTensor(15) = uint8_t(x0);
    }
};

// =========================
// Kernel Entry
// =========================
extern "C" __global__ __aicore__ void sm4_ctr_encrypt(
    __gm__ uint8_t* roundKeys128,   // GM: 128B = 32 * uint32 round keys
    __gm__ uint8_t* input,          // GM: dataSize bytes, must be multiple of 16
    __gm__ uint8_t* output,         // GM: dataSize bytes
    uint32_t dataSize)
{
    KernelSM4CTR op;
    op.Init(roundKeys128, input, output, dataSize);
    op.Process();
}

namespace vllm_ascend {
// Host wrapper: blockDim = number of cores to use
void sm4_ctr_encrypt_do_impl(uint32_t blockDim, void* stream,
                        void* roundKeys128, void* input, void* output,
                        uint32_t dataSize)
{
    sm4_ctr_encrypt<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t*)roundKeys128,
        (__gm__ uint8_t*)input,
        (__gm__ uint8_t*)output,
        dataSize);
}
}