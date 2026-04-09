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

#include "kernel_operator.h"  
using namespace AscendC;  

constexpr int BLOCK_SIZE = 64; 
constexpr int MAX_BLOCKS_PER_CALL = 2048;

class KernelChaCha20Optimized {  
public:  
    __aicore__ inline KernelChaCha20Optimized() {}  

    __aicore__ inline void Init(__gm__ void* state, __gm__ void* input, __gm__ void* output, uint32_t dataSize, uint32_t workspaceSize)  
    {  
        stateGlobal.SetGlobalBuffer((__gm__ uint16_t*)state + AscendC::GetBlockIdx() * (dataSize / sizeof(uint16_t) / AscendC::GetBlockNum()));  
        inputGlobal.SetGlobalBuffer((__gm__ uint16_t*)input + AscendC::GetBlockIdx() * (dataSize / sizeof(uint16_t) / AscendC::GetBlockNum()));  
        outputGlobal.SetGlobalBuffer((__gm__ uint16_t*)output + AscendC::GetBlockIdx() * (dataSize / sizeof(uint16_t) / AscendC::GetBlockNum())); 
        this->dataSize =  dataSize;

        if (dataSize < 8 * 1024)
            DOUBLE_BUFFER_SIZE = 128;
        else if (dataSize < 128 * 1024)
            DOUBLE_BUFFER_SIZE = 256;
        else if (dataSize < 1024 * 1024)
            DOUBLE_BUFFER_SIZE = 1024;
        else
            DOUBLE_BUFFER_SIZE = 2048;
        
        // 增加缓冲区大小
        pipe.InitBuffer(stateQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(inputQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(outputQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(tmpQueue, 2, workspaceSize);
    }  
    
    __aicore__ inline void Process()  
    {  
        uint32_t threadId = AscendC::GetBlockIdx();  
        uint32_t threadNum = AscendC::GetBlockNum();  
        uint32_t dataSizePerThread = dataSize / threadNum;

        uint32_t blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread);
        for (uint32_t i = 0; i < dataSizePerThread; i += blocksToProcess){
            blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread - i);
            CopyIn(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
            ComputeVectorized(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
            CopyOut(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
        }
    }

private:  
    __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint16_t> stateLocal = stateQueue.AllocTensor<uint16_t>();  
        AscendC::LocalTensor<uint16_t> inputLocal = inputQueue.AllocTensor<uint16_t>();  
        
        AscendC::DataCopy(inputLocal, inputGlobal[startBlock], blocksToProcess);  
        AscendC::DataCopy(stateLocal, stateGlobal[startBlock], blocksToProcess);  
        
        stateQueue.EnQue(stateLocal);  
        inputQueue.EnQue(inputLocal);  
    }  
    
    // 优化的计算函数 - 使用本地数组减少内存访问
    __aicore__ inline void ComputeVectorized(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint16_t> stateLocal = stateQueue.DeQue<uint16_t>();  
        AscendC::LocalTensor<uint16_t> inputLocal = inputQueue.DeQue<uint16_t>();  
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue.AllocTensor<uint16_t>();

        AscendC::LocalTensor<uint8_t> tmpLocal = tmpQueue.AllocTensor<uint8_t>();
        AscendC::Xor(outputLocal, stateLocal, inputLocal, tmpLocal, blocksToProcess);
        tmpQueue.FreeTensor(tmpLocal);
        
        outputQueue.EnQue<uint16_t>(outputLocal);
        stateQueue.FreeTensor(stateLocal);
        inputQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
    {
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue.DeQue<uint16_t>();
        
        AscendC::DataCopy(outputGlobal[startBlock], outputLocal, blocksToProcess);

        outputQueue.FreeTensor(outputLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> stateQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> tmpQueue;
    AscendC::GlobalTensor<uint16_t> stateGlobal;
    AscendC::GlobalTensor<uint16_t> inputGlobal;
    AscendC::GlobalTensor<uint16_t> outputGlobal;
    uint32_t dataSize;
    int DOUBLE_BUFFER_SIZE = 1024;
};

class KernelChaCha20OptimizedBatched {  
public:  
    __aicore__ inline KernelChaCha20OptimizedBatched() {}  

    __aicore__ inline void Init(__gm__ void* state, __gm__ void* input, __gm__ void* output, uint32_t dataSize, uint32_t batchSize, uint32_t workspaceSize)  
    {  
        this->dataSize = dataSize;
        this->batchSize = batchSize;
        this->dataSizePerBatch = dataSize / batchSize;
        stateGlobal.SetGlobalBuffer((__gm__ uint16_t*)state + AscendC::GetBlockIdx() * (this->dataSizePerBatch / sizeof(uint16_t) / AscendC::GetBlockNum()));  
        inputGlobal.SetGlobalBuffer((__gm__ uint16_t*)input + AscendC::GetBlockIdx() * (this->dataSizePerBatch / sizeof(uint16_t) / AscendC::GetBlockNum()));  
        outputGlobal.SetGlobalBuffer((__gm__ uint16_t*)output + AscendC::GetBlockIdx() * (this->dataSizePerBatch / sizeof(uint16_t) / AscendC::GetBlockNum())); 

        if (this->dataSizePerBatch < 8 * 1024)
            DOUBLE_BUFFER_SIZE = 128;
        else if (this->dataSizePerBatch < 128 * 1024)
            DOUBLE_BUFFER_SIZE = 256;
        else if (this->dataSizePerBatch < 1024 * 1024)
            DOUBLE_BUFFER_SIZE = 1024;
        else
            DOUBLE_BUFFER_SIZE = 2048;
        
        // 增加缓冲区大小
        pipe.InitBuffer(stateQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(inputQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(outputQueue, 2, DOUBLE_BUFFER_SIZE);  
        pipe.InitBuffer(tmpQueue, 2, workspaceSize);
    }  
    
    __aicore__ inline void Process()  
    {  
        uint32_t threadId = AscendC::GetBlockIdx();  
        uint32_t threadNum = AscendC::GetBlockNum();  
        uint32_t dataSizePerThread = this->dataSizePerBatch / threadNum;

        for (uint32_t b = 0; b < this->batchSize; b++){
            uint32_t blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread);
            for (uint32_t i = 0; i < dataSizePerThread; i += blocksToProcess){
                blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread - i);
                CopyIn(i / sizeof(uint16_t) + b * this->dataSizePerBatch / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
                ComputeVectorized(i / sizeof(uint16_t) + b * this->dataSizePerBatch / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
                CopyOut(i / sizeof(uint16_t) + b * this->dataSizePerBatch / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
            }
        }
    }

private:  
    __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint16_t> stateLocal = stateQueue.AllocTensor<uint16_t>();  
        AscendC::LocalTensor<uint16_t> inputLocal = inputQueue.AllocTensor<uint16_t>();  

        uint32_t state_startBlock = startBlock % (this->dataSizePerBatch / sizeof(uint16_t));
        
        AscendC::DataCopy(inputLocal, inputGlobal[startBlock], blocksToProcess);  
        AscendC::DataCopy(stateLocal, stateGlobal[state_startBlock], blocksToProcess);  
        
        stateQueue.EnQue(stateLocal);  
        inputQueue.EnQue(inputLocal);  
    }  
    
    // 优化的计算函数 - 使用本地数组减少内存访问
    __aicore__ inline void ComputeVectorized(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint16_t> stateLocal = stateQueue.DeQue<uint16_t>();  
        AscendC::LocalTensor<uint16_t> inputLocal = inputQueue.DeQue<uint16_t>();  
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue.AllocTensor<uint16_t>();

        AscendC::LocalTensor<uint8_t> tmpLocal = tmpQueue.AllocTensor<uint8_t>();
        AscendC::Xor(outputLocal, stateLocal, inputLocal, tmpLocal, blocksToProcess);
        tmpQueue.FreeTensor(tmpLocal);
        
        outputQueue.EnQue<uint16_t>(outputLocal);
        stateQueue.FreeTensor(stateLocal);
        inputQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
    {
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue.DeQue<uint16_t>();
        
        AscendC::DataCopy(outputGlobal[startBlock], outputLocal, blocksToProcess);

        outputQueue.FreeTensor(outputLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> stateQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> tmpQueue;
    AscendC::GlobalTensor<uint16_t> stateGlobal;
    AscendC::GlobalTensor<uint16_t> inputGlobal;
    AscendC::GlobalTensor<uint16_t> outputGlobal;
    uint32_t dataSize;
    uint32_t batchSize;
    uint32_t dataSizePerBatch;
    int DOUBLE_BUFFER_SIZE = 1024;
};

class KernelChaCha20Optimized_for_h2d {  
public:  
    __aicore__ inline KernelChaCha20Optimized_for_h2d() {}  

    __aicore__ inline void Init(__gm__ uint16_t* state, __gm__ uint16_t* input, __gm__ uint16_t* output, uint32_t dataSize, uint32_t workspace)  
    {  
        // AscendC::printf("input[%p] state[%p]", input, state);
        uint32_t localSize = (dataSize + GetBlockNum() - 1) / GetBlockNum();
        this->localSizePadding = (localSize + 32 - 1) / 32 * 32;
        this->dataSize = dataSize;

        if (GetBlockIdx() * this->localSizePadding < this->dataSize) {
            stateGlobal.SetGlobalBuffer((__gm__ uint16_t*)state + (GetBlockIdx() * this->localSizePadding / sizeof(uint16_t)));
            inputGlobal.SetGlobalBuffer((__gm__ uint16_t*)input + (GetBlockIdx() * this->localSizePadding / sizeof(uint16_t)));
            outputGlobal.SetGlobalBuffer((__gm__ uint16_t*)output + (GetBlockIdx() * this->localSizePadding / sizeof(uint16_t)));
            stateGlobal_uint8.SetGlobalBuffer((__gm__ uint8_t*)state + (GetBlockIdx() * this->localSizePadding / sizeof(uint8_t)));
            inputGlobal_uint8.SetGlobalBuffer((__gm__ uint8_t*)input + (GetBlockIdx() * this->localSizePadding / sizeof(uint8_t)));
            outputGlobal_uint8.SetGlobalBuffer((__gm__ uint8_t*)output + (GetBlockIdx() * this->localSizePadding / sizeof(uint8_t)));
            // if (GetBlockIdx() <= 23) {
            //     AscendC::printf("line 389: [%d, %d]", dataSize, (GetBlockIdx() * this->localSizePadding / sizeof(uint16_t)));
            // }

            if (dataSize < 8 * 1024)
                DOUBLE_BUFFER_SIZE = 128;
            else if (dataSize < 128 * 1024)
                DOUBLE_BUFFER_SIZE = 256;
            else if (dataSize < 1024 * 1024)
                DOUBLE_BUFFER_SIZE = 1024;
            else
                DOUBLE_BUFFER_SIZE = 2048;
            
            // 增加缓冲区大小
            pipe.InitBuffer(stateQueue, 2, DOUBLE_BUFFER_SIZE);  
            pipe.InitBuffer(inputQueue, 2, DOUBLE_BUFFER_SIZE);  
            pipe.InitBuffer(outputQueue, 2, DOUBLE_BUFFER_SIZE);  
            pipe.InitBuffer(tmpQueue, 2, workspace);
        }
    }  
    
    __aicore__ inline void Process()  
    {  
        if (GetBlockIdx() * this->localSizePadding < this->dataSize) {
            uint32_t dataSizePerThread= min(this->localSizePadding, static_cast<uint32_t>(this->dataSize - GetBlockIdx() * this->localSizePadding));
            uint32_t blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), this->localSizePadding);
            uint32_t i = 0;
            // for (; i < this->localSizePadding && i + GetBlockIdx() * this->localSizePadding <= this->dataSize; i += blocksToProcess){
            // if (GetBlockIdx() <= 23) {
            //     AscendC::printf("line 417: [%d, %d]", dataSize, dataSizePerThread);
            // }
            for (; i < dataSizePerThread; i += blocksToProcess){
                blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread - i);
                if (blocksToProcess % 32 != 0) {
                    blocksToProcess = blocksToProcess / 32 * 32;
                }
                if (blocksToProcess < 32)
                    break;
                // if (GetBlockIdx() <= 23) {
                //     AscendC::printf("line 427: [%d, %d]", i, blocksToProcess / sizeof(uint16_t));
                // }
                CopyIn(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
                ComputeVectorized(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
                CopyOut(i / sizeof(uint16_t), blocksToProcess / sizeof(uint16_t));
            }
            for(; i < dataSizePerThread; i++) {
                // outputGlobal_uint8(i) = stateGlobal_uint8(i) ^ inputGlobal_uint8(i);
                uint8_t a = stateGlobal_uint8.GetValue(i);
                uint8_t b = inputGlobal_uint8.GetValue(i);
                uint8_t c = a ^ b;
                outputGlobal_uint8.SetValue(i, c);
            }
        }
    }

private:  
    __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        LocalTensor<uint16_t> stateLocal = stateQueue.AllocTensor<uint16_t>();  
        LocalTensor<uint16_t> inputLocal = inputQueue.AllocTensor<uint16_t>();  
        
        DataCopy(inputLocal, inputGlobal[startBlock], blocksToProcess);  
        DataCopy(stateLocal, stateGlobal[startBlock], blocksToProcess);  
        
        stateQueue.EnQue(stateLocal);  
        inputQueue.EnQue(inputLocal);  
    }  
    
    __aicore__ inline void ComputeVectorized(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        LocalTensor<uint16_t> stateLocal = stateQueue.DeQue<uint16_t>();  
        LocalTensor<uint16_t> inputLocal = inputQueue.DeQue<uint16_t>();  
        LocalTensor<uint16_t> outputLocal = outputQueue.AllocTensor<uint16_t>();

        LocalTensor<uint8_t> tmpLocal = tmpQueue.AllocTensor<uint8_t>();
        AscendC::Xor(outputLocal, stateLocal, inputLocal, tmpLocal, blocksToProcess);
        tmpQueue.FreeTensor(tmpLocal);
        
        outputQueue.EnQue<uint16_t>(outputLocal);
        stateQueue.FreeTensor(stateLocal);
        inputQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
    {
        LocalTensor<uint16_t> outputLocal = outputQueue.DeQue<uint16_t>();
        
        DataCopy(outputGlobal[startBlock], outputLocal, blocksToProcess);

        outputQueue.FreeTensor(outputLocal);
    }
    
private:
    AscendC::TPipe pipe;
    TQue<TPosition::VECIN, 2> stateQueue;
    TQue<TPosition::VECIN, 2> inputQueue;
    TQue<TPosition::VECOUT, 2> outputQueue;
    TQue<TPosition::VECIN, 2> tmpQueue;
    GlobalTensor<uint16_t> stateGlobal;
    GlobalTensor<uint16_t> inputGlobal;
    GlobalTensor<uint16_t> outputGlobal;
    GlobalTensor<uint8_t> stateGlobal_uint8;
    GlobalTensor<uint8_t> inputGlobal_uint8;
    GlobalTensor<uint8_t> outputGlobal_uint8;
    uint32_t dataSize;
    uint32_t localSizePadding;
    int DOUBLE_BUFFER_SIZE = 1024;
};

// class KernelChaCha20Generation {  
// public:  
//     __aicore__ inline KernelChaCha20Generation() {}  

//     // 优化的并行四轮运算 - 减少内存访问
//     __aicore__ inline void parallelQuarterRounds(uint32_t* state,
//                                                int a1, int b1, int c1, int d1,
//                                                int a2, int b2, int c2, int d2,
//                                                int a3, int b3, int c3, int d3,
//                                                int a4, int b4, int c4, int d4)
//     {
//         // 批量读取，减少内存访问
//         uint32_t va1 = state[a1], vb1 = state[b1], vc1 = state[c1], vd1 = state[d1];
//         uint32_t va2 = state[a2], vb2 = state[b2], vc2 = state[c2], vd2 = state[d2];
//         uint32_t va3 = state[a3], vb3 = state[b3], vc3 = state[c3], vd3 = state[d3];
//         uint32_t va4 = state[a4], vb4 = state[b4], vc4 = state[c4], vd4 = state[d4];

//         // 第一步
//         va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 16) | (vd1 >> 16);
//         va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 16) | (vd2 >> 16);
//         va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 16) | (vd3 >> 16);
//         va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 16) | (vd4 >> 16);
        
//         // 第二步
//         vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 12) | (vb1 >> 20);
//         vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 12) | (vb2 >> 20);
//         vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 12) | (vb3 >> 20);
//         vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 12) | (vb4 >> 20);
        
//         // 第三步
//         va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 8) | (vd1 >> 24);
//         va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 8) | (vd2 >> 24);
//         va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 8) | (vd3 >> 24);
//         va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 8) | (vd4 >> 24);
        
//         // 第四步
//         vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 7) | (vb1 >> 25);
//         vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 7) | (vb2 >> 25);
//         vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 7) | (vb3 >> 25);
//         vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 7) | (vb4 >> 25);
        
//         // 批量写入
//         state[a1] = va1; state[b1] = vb1; state[c1] = vc1; state[d1] = vd1;
//         state[a2] = va2; state[b2] = vb2; state[c2] = vc2; state[d2] = vd2;
//         state[a3] = va3; state[b3] = vb3; state[c3] = vc3; state[d3] = vd3;
//         state[a4] = va4; state[b4] = vb4; state[c4] = vc4; state[d4] = vd4;
//     }

//     __aicore__ inline void Init(__gm__ void* state, __gm__ void* output, uint32_t dataSize)  
//     {  
//         stateGlobal.SetGlobalBuffer((__gm__ uint32_t*)state);  
//         outputGlobal.SetGlobalBuffer((__gm__ uint8_t*)output + AscendC::GetBlockIdx() * (dataSize / AscendC::GetBlockNum()));
//         this->dataSize = dataSize;  
//         this->totalBlocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;  
//         this->localBlocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE / AscendC::GetBlockNum();  
        
//         // 增加缓冲区大小
//         pipe.InitBuffer(stateQueue, 1, 16 * sizeof(uint32_t));   
//         pipe.InitBuffer(outputQueue, 2, DOUBLE_BUFFER_SIZE);  
//         pipe.InitBuffer(workingQueue, 2, 16 * sizeof(uint32_t));  
//     }  
    
//     __aicore__ inline void Process()
//     {  
//         uint32_t threadId = AscendC::GetBlockIdx();  
//         uint32_t threadNum = AscendC::GetBlockNum();  
//         uint32_t dataSizePerThread = dataSize / threadNum;
//         uint32_t blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread);

//         AscendC::LocalTensor<uint32_t> stateLocal = stateQueue.AllocTensor<uint32_t>();  
//         DataCopy(stateLocal, stateGlobal, 16);  
//         for (int i = 0; i < 16; i++) {
//             localState[i] = stateLocal.GetValue(i);
//         }
        
//         for (uint32_t i = 0; i < dataSizePerThread; i += blocksToProcess){
//             blocksToProcess = min(static_cast<uint32_t>(DOUBLE_BUFFER_SIZE), dataSizePerThread - i);
//             if (blocksToProcess > 0) {  
//                 ComputeVectorized(i / BLOCK_SIZE, blocksToProcess / BLOCK_SIZE);
//                 CopyOut(i / BLOCK_SIZE, blocksToProcess / BLOCK_SIZE);  
//             }  
//         }

//         stateQueue.FreeTensor(stateLocal);
//     }

// private:  
//     uint32_t dataSize;  
//     uint32_t totalBlocks;  
//     uint32_t localBlocks;
//     uint32_t localState[16];  // 本地状态缓存
    
//     __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)  
//     {  
//         AscendC::LocalTensor<uint32_t> stateLocal = stateQueue.DeQue<uint32_t>();  
//         stateQueue.EnQue(stateLocal);  
//     }  
    
//     // 优化的计算函数 - 使用本地数组减少内存访问
//     __aicore__ inline void ComputeVectorized(uint32_t startBlock, uint32_t blocksToProcess)  
//     {   
//         AscendC::LocalTensor<uint8_t> outputLocal = outputQueue.AllocTensor<uint8_t>();
        
//         for (uint32_t block = 0; block < blocksToProcess; block++) {
//             // 快速初始化工作状态
//             uint32_t workingState[16];
//             for (int i = 0; i < 16; i++) {
//                 workingState[i] = localState[i];
//             }
            
//             workingState[12] += startBlock + block;
            
//             // 优化的ChaCha20块变换
//             ChaCha20BlockOptimized(workingState);
            
//             // 优化的XOR操作
//             fastXOR(workingState, outputLocal, block * BLOCK_SIZE);
//         }
        
//         outputQueue.EnQue<uint8_t>(outputLocal);
//     }
    
//     // 优化的ChaCha20块变换
//     __aicore__ inline void ChaCha20BlockOptimized(uint32_t* state)
//     {
//         uint32_t originalState[16];
//         // 批量拷贝
//         for (int i = 0; i < 16; i++) {
//             originalState[i] = state[i];
//         }
        
//         // 展开前两轮以减少循环开销
//         parallelQuarterRounds(state, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); 
//         parallelQuarterRounds(state, 0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14);
//         parallelQuarterRounds(state, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); 
//         parallelQuarterRounds(state, 0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14);
        
//         // 剩余轮次
//         for (int round = 2; round < 10; round++) {
//             parallelQuarterRounds(state, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); 
//             parallelQuarterRounds(state, 0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14);  
//         }
        
//         // 批量加法
//         for (int i = 0; i < 16; i++) {
//             state[i] += originalState[i];
//         }
//     }
    
//     // 优化的XOR操作 - 直接使用指针访问
//     __aicore__ inline void fastXOR(uint32_t* keystream, 
//                                   AscendC::LocalTensor<uint8_t>& output, uint32_t blockOffset)
//     {
//         // 按64字节块处理，4字节对齐
//         for (int i = 0; i < 16; i++) {
//             uint32_t wordOffset = blockOffset + i * 4;
            
//             if (wordOffset + 3 < this->localBlocks * BLOCK_SIZE) {
//                 uint32_t outputWord = keystream[i];
                
//                 output.SetValue(wordOffset + 0, uint8_t(outputWord));
//                 output.SetValue(wordOffset + 1, uint8_t(outputWord >> 8));
//                 output.SetValue(wordOffset + 2, uint8_t(outputWord >> 16));
//                 output.SetValue(wordOffset + 3, uint8_t(outputWord >> 24));
//             }
//         }
//     }
    
//     __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
//     {
//         AscendC::LocalTensor<uint8_t> outputLocal = outputQueue.DeQue<uint8_t>();
        
//         uint32_t outputOffset = startBlock * BLOCK_SIZE;
//         uint32_t outputSize = blocksToProcess * BLOCK_SIZE;
        
//         if (outputOffset + outputSize > dataSize) {
//             outputSize = dataSize - outputOffset;
//         }
        
//         AscendC::DataCopy(outputGlobal[outputOffset], outputLocal, outputSize);
//         outputQueue.FreeTensor(outputLocal);
//     }
    
// private:
//     AscendC::TPipe pipe;
//     AscendC::TQue<AscendC::TPosition::VECIN, 1> stateQueue;
//     AscendC::TQue<AscendC::TPosition::VECOUT, 2> outputQueue;
//     AscendC::TQue<AscendC::TPosition::VECIN, 2> workingQueue;
//     AscendC::GlobalTensor<uint32_t> stateGlobal;
//     AscendC::GlobalTensor<uint8_t> outputGlobal;
//     int DOUBLE_BUFFER_SIZE = 1024;
// };

class KernelChaCha20Generation {  
public:  
    __aicore__ inline KernelChaCha20Generation() {}  

    // 优化的并行四轮运算 - 减少内存访问
    __aicore__ inline void parallelQuarterRounds(uint32_t* state,
                                               int a1, int b1, int c1, int d1,
                                               int a2, int b2, int c2, int d2,
                                               int a3, int b3, int c3, int d3,
                                               int a4, int b4, int c4, int d4)
    {
        // 批量读取，减少内存访问
        uint32_t va1 = state[a1], vb1 = state[b1], vc1 = state[c1], vd1 = state[d1];
        uint32_t va2 = state[a2], vb2 = state[b2], vc2 = state[c2], vd2 = state[d2];
        uint32_t va3 = state[a3], vb3 = state[b3], vc3 = state[c3], vd3 = state[d3];
        uint32_t va4 = state[a4], vb4 = state[b4], vc4 = state[c4], vd4 = state[d4];

        // 第一步
        va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 16) | (vd1 >> 16);
        va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 16) | (vd2 >> 16);
        va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 16) | (vd3 >> 16);
        va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 16) | (vd4 >> 16);
        
        // 第二步
        vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 12) | (vb1 >> 20);
        vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 12) | (vb2 >> 20);
        vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 12) | (vb3 >> 20);
        vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 12) | (vb4 >> 20);
        
        // 第三步
        va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 8) | (vd1 >> 24);
        va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 8) | (vd2 >> 24);
        va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 8) | (vd3 >> 24);
        va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 8) | (vd4 >> 24);
        
        // 第四步
        vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 7) | (vb1 >> 25);
        vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 7) | (vb2 >> 25);
        vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 7) | (vb3 >> 25);
        vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 7) | (vb4 >> 25);
        
        // 批量写入
        state[a1] = va1; state[b1] = vb1; state[c1] = vc1; state[d1] = vd1;
        state[a2] = va2; state[b2] = vb2; state[c2] = vc2; state[d2] = vd2;
        state[a3] = va3; state[b3] = vb3; state[c3] = vc3; state[d3] = vd3;
        state[a4] = va4; state[b4] = vb4; state[c4] = vc4; state[d4] = vd4;
    }

    __aicore__ inline void Init(__gm__ void* state, __gm__ void* output, uint32_t dataSize)  
    {  
        stateGlobal.SetGlobalBuffer((__gm__ uint32_t*)state);  
        outputGlobal.SetGlobalBuffer((__gm__ uint8_t*)output);  
        this->dataSize = dataSize;  
        this->totalBlocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;  
        
        // 增加缓冲区大小
        pipe.InitBuffer(stateQueue, 1, 16 * sizeof(uint32_t));   
        pipe.InitBuffer(outputQueue, 1, MAX_BLOCKS_PER_CALL * BLOCK_SIZE * sizeof(uint8_t));  
        pipe.InitBuffer(workingQueue, 1, 16 * sizeof(uint32_t));  
    }  
    
    __aicore__ inline void Process()
    {  
        uint32_t blockId = AscendC::GetBlockIdx();  
        uint32_t startBlock = blockId * MAX_BLOCKS_PER_CALL;  
        uint32_t endBlock = min(startBlock + MAX_BLOCKS_PER_CALL, totalBlocks);  
        uint32_t blocksToProcess = endBlock - startBlock;  
        
        if (blocksToProcess > 0) {  
            CopyIn(startBlock, blocksToProcess);  
            ComputeVectorized(startBlock, blocksToProcess);
            CopyOut(startBlock, blocksToProcess);  
        }  
    }

private:  
    uint32_t dataSize;  
    uint32_t totalBlocks;  
    uint32_t localState[16];  // 本地状态缓存
    
    __aicore__ inline void CopyIn(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint32_t> stateLocal = stateQueue.AllocTensor<uint32_t>();  
        
        AscendC::DataCopy(stateLocal, stateGlobal, 16);  
        
        stateQueue.EnQue(stateLocal);  
    }  
    
    // 优化的计算函数 - 使用本地数组减少内存访问
    __aicore__ inline void ComputeVectorized(uint32_t startBlock, uint32_t blocksToProcess)  
    {  
        AscendC::LocalTensor<uint32_t> stateLocal = stateQueue.DeQue<uint32_t>();   
        AscendC::LocalTensor<uint8_t> outputLocal = outputQueue.AllocTensor<uint8_t>();
        
        // 批量读取初始状态到本地数组
        for (int i = 0; i < 16; i++) {
            localState[i] = stateLocal.GetValue(i);
        }
        
        for (uint32_t block = 0; block < blocksToProcess; block++) {
            // 快速初始化工作状态
            uint32_t workingState[16];
            for (int i = 0; i < 16; i++) {
                workingState[i] = localState[i];
            }
            
            workingState[12] += startBlock + block;
            
            // 优化的ChaCha20块变换
            ChaCha20BlockOptimized(workingState);
            
            // 优化的XOR操作
            fastXOR(workingState, outputLocal, block * BLOCK_SIZE);
        }
        
        outputQueue.EnQue<uint8_t>(outputLocal);
        stateQueue.FreeTensor(stateLocal);
    }
    
    // 优化的ChaCha20块变换
    __aicore__ inline void ChaCha20BlockOptimized(uint32_t* state)
    {
        uint32_t originalState[16];
        // 批量拷贝
        for (int i = 0; i < 16; i++) {
            originalState[i] = state[i];
        }
        
        // 剩余轮次
        for (int round = 0; round < 10; round++) {
            parallelQuarterRounds(state, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); 
            parallelQuarterRounds(state, 0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14);  
        }
        
        // 批量加法
        for (int i = 0; i < 16; i++) {
            state[i] += originalState[i];
        }
    }
    
    // 优化的XOR操作 - 直接使用指针访问
    __aicore__ inline void fastXOR(uint32_t* keystream, 
                                  AscendC::LocalTensor<uint8_t>& output, uint32_t blockOffset)
    {
        // 按64字节块处理，4字节对齐
        for (int i = 0; i < 16; i++) {
            uint32_t wordOffset = blockOffset + i * 4;
            
            if (wordOffset + 3 < MAX_BLOCKS_PER_CALL * BLOCK_SIZE) {
                uint32_t outputWord = keystream[i];
                
                output.SetValue(wordOffset + 0, uint8_t(outputWord));
                output.SetValue(wordOffset + 1, uint8_t(outputWord >> 8));
                output.SetValue(wordOffset + 2, uint8_t(outputWord >> 16));
                output.SetValue(wordOffset + 3, uint8_t(outputWord >> 24));
            }
        }
    }
    
    __aicore__ inline void CopyOut(uint32_t startBlock, uint32_t blocksToProcess)
    {
        AscendC::LocalTensor<uint8_t> outputLocal = outputQueue.DeQue<uint8_t>();
        
        uint32_t outputOffset = startBlock * BLOCK_SIZE;
        uint32_t outputSize = blocksToProcess * BLOCK_SIZE;
        
        if (outputOffset + outputSize > dataSize) {
            outputSize = dataSize - outputOffset;
        }
        
        AscendC::DataCopy(outputGlobal[outputOffset], outputLocal, outputSize);
        outputQueue.FreeTensor(outputLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> stateQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workingQueue;
    AscendC::GlobalTensor<uint32_t> stateGlobal;
    AscendC::GlobalTensor<uint8_t> outputGlobal;
};

extern "C" __global__ __aicore__ void chacha20_encrypt_optimized(
    __gm__ uint16_t* state, 
    __gm__ uint16_t* input, 
    __gm__ uint16_t* output, 
    uint32_t dataSize,
    uint32_t workspaceSize)
{
    KernelChaCha20Optimized op;
    op.Init(state, input, output, dataSize, workspaceSize);
    op.Process();
}

extern "C" __global__ __aicore__ void chacha20_encrypt_optimized_batch(
    __gm__ uint16_t* state, 
    __gm__ uint16_t* input, 
    __gm__ uint16_t* output, 
    uint32_t dataSize,
    uint32_t batchSize,
    uint32_t workspaceSize)
{
    KernelChaCha20OptimizedBatched op;
    op.Init(state, input, output, dataSize, batchSize, workspaceSize);
    op.Process();
}

extern "C" __global__ __aicore__ void chacha20_encrypt_optimized_unalign(
    __gm__ uint16_t* state, 
    __gm__ uint16_t* input, 
    __gm__ uint16_t* output, 
    uint32_t dataSize,
    uint32_t workspaceSize)
{
    KernelChaCha20Optimized_for_h2d op;
    op.Init(state, input, output, dataSize, workspaceSize);
    op.Process();
}

extern "C" __global__ __aicore__ void chacha20_encrypt_generation(
    __gm__ void* state, 
    __gm__ void* output, 
    uint32_t dataSize)
{
    KernelChaCha20Generation op;
    op.Init(state, output, dataSize);
    op.Process();
}

namespace vllm_ascend {

extern void chacha20_encrypt_do_impl(void *stream, void* state, void* input, void* output, uint32_t dataSize, uint32_t workspaceSize)
{
    chacha20_encrypt_optimized<<<32, nullptr, stream>>>(
        state, 
        input, 
        output, 
        dataSize,
        workspaceSize);
}

extern void chacha20_encrypt_do_batch_impl(void *stream, void* state, void* input, void* output, uint32_t dataSize, uint32_t batchSize, uint32_t workspaceSize)
{
    chacha20_encrypt_optimized_batch<<<32, nullptr, stream>>>(
        state, 
        input, 
        output, 
        dataSize,
        batchSize,
        workspaceSize);
}

extern void chacha20_encrypt_do_unalign_impl(void *stream, void* state, void* input, void* output, uint32_t dataSize, uint32_t workspaceSize)
{
    chacha20_encrypt_optimized_unalign<<<32, nullptr, stream>>>(
        state, 
        input, 
        output, 
        dataSize,
        workspaceSize);
}

extern void chacha20_encrypt_generate_mask_impl(uint32_t blockDim, void *stream, void* state, void* output, uint32_t dataSize)
{
    chacha20_encrypt_generation<<<blockDim, nullptr, stream>>>(
        state, 
        output, 
        dataSize);
}

} // namespace vllm_ascend