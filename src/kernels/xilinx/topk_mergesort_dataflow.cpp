 #include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskTopK;
using hlslib::Stream;

void TopK_MergeSortDF_V1_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<CONFIG_DTYPE, 8> &streamDataOutL,
    Stream<CONFIG_DTYPE, 8> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(dim0==1);
    assert(dim1==MaxSliceLen);

    LoopVecsPerPE:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        #pragma HLS PIPELINE II=1

        const unsigned d0 = 0;
        const unsigned indxS = iVec;

        MemoryPackF_t vec = inputTn[indxS];

        LoopPush:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            if(i%2==0){
                streamDataOutL.Push(vec[i]);
            }else{
                streamDataOutR.Push(vec[i]);
            }
        }
    }
}

void TopK_MergeSortDF_V1_UnitWrite(
    MemoryPackF_t *outputTn,
    Stream<CONFIG_DTYPE, 1024> &streamDataInL,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(dim0==1);
    assert(dim1==MaxSliceLen);

    LoopVecsPerPE:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        #pragma HLS PIPELINE II=1

        const unsigned d0 = 0;
        const unsigned indxD = iVec;

        MemoryPackF_t vec;

        LoopPush:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            vec[i] = streamDataInL.Pop();
        }

        outputTn[indxD] = vec;
    }
}

/*
template<
    unsigned windowWidth, // 2,4,8,...,512
    unsigned depthInputs,
    unsigned depthOutputs
>
void TopK_MergeSortDF_V1_UnitMergeX(
    Stream<CONFIG_DTYPE, depthInputs> &streamDataInL,
    Stream<CONFIG_DTYPE, depthInputs> &streamDataInR,
    Stream<CONFIG_DTYPE, depthOutputs> &streamDataOutL,
    Stream<CONFIG_DTYPE, depthOutputs> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    assert(dim1 % windowWidth == 0);
    assert(dim1 <= MaxSliceLen); 
    assert(windowWidth != (MaxSliceLen/2));

    constexpr unsigned win2 = (2 * windowWidth);
    constexpr unsigned MaxSliceLenBy2 = MaxSliceLen/2;

    int f1 = 0;
    int f2 = windowWidth;
    int i2 = windowWidth;
    int i3 = win2;
    if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
    if(i3 >= MaxSliceLen) i3 = MaxSliceLen;

    LoopMergeArrays:
    for (int i = 0; i < MaxSliceLen; i++) {
        #pragma HLS pipeline II=1

        CONFIG_DTYPE t1 = streamDataInL.Pop();
        CONFIG_DTYPE t2 = (f2 == i3) ? 0 : streamDataInR.Pop();
        const unsigned offset = i % win2;

        if(f2 == i3 || (f1 < i2 && t1 <= t2)) {
            if(offset<windowWidth){
                streamDataOutL.Push(t1);
            }else{
                streamDataOutR.Push(t1);
            }
            f1++;
        } else {
            if(offset<windowWidth){
                streamDataOutL.Push(t2);
            }else{
                streamDataOutR.Push(t2);
            }
            assert(f2 < i3);
            f2++;
        }
        if(f1 == i2 && f2 == i3) {
            f1 = i3;
            i2 += 2*windowWidth;
            i3 += 2*windowWidth;
            if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
            if(i3 >= MaxSliceLen) i3 = MaxSliceLen;
            f2 = i2;
        }
    }
}
*/

template<
     unsigned windowWidth, // 2,4,8,...,512
     unsigned depthInputs,
     unsigned depthOutputs
>
void TopK_MergeSortDF_V1_UnitMergeX(
     Stream<CONFIG_DTYPE, depthInputs> &streamDataInL,
     Stream<CONFIG_DTYPE, depthInputs> &streamDataInR,
     Stream<CONFIG_DTYPE, depthOutputs> &streamDataOutL,
     Stream<CONFIG_DTYPE, depthOutputs> &streamDataOutR,
     const unsigned dim0,
     const unsigned dim1){

    assert(dim1 % windowWidth == 0);
    assert(dim1 <= MaxSliceLen);
    //assert(windowWidth != (MaxSliceLen/2));

    constexpr unsigned win2 = (2 * windowWidth);
    constexpr unsigned MaxSliceLenBy2 = MaxSliceLen/2;

    const unsigned pairsToBeMerged = MaxSliceLen / win2;

    LoopPairs:
    for(unsigned pair=0; pair<pairsToBeMerged; pair++){
        CONFIG_DTYPE buffPair[win2];
#pragma HLS ARRAY_PARTITION variable=buffPair complete dim=1

        // ---------------------------------------------
        // 1. Fetch two pairs before merging
        LoopFetchPairs:
        for(unsigned w=0; w<win2; w++){
            if(w<windowWidth){
                buffPair[w] = streamDataInL.Pop();
            }else{
                buffPair[w] = streamDataInR.Pop();
            }
        }

        // ---------------------------------------------
        // 2. Merge the pairs
        unsigned f1 = 0;
        unsigned f2 = windowWidth;
        unsigned i2 = windowWidth;
        unsigned i3 = win2;
        if(i2 >= win2) i2 = win2;
        if(i3 >= win2) i3 = win2;

        LoopMerge1:
        for(unsigned i=0; i<win2; i++){
            #pragma HLS PIPELINE II=1

            CONFIG_DTYPE t1 = buffPair[f1];
            CONFIG_DTYPE t2 = (f2 == i3) ? 0 : buffPair[f2];

            if(f2 == i3 || (f1 < i2 && t1 <= t2)) {
                if(pair%2==0){
                    streamDataOutL.Push(t1);
                }else{
                    streamDataOutR.Push(t1);
                }
                f1++;
            } else {
                assert(f2 < i3);
                if(pair%2==0){
                    streamDataOutL.Push(t2);
                }else{
                    streamDataOutR.Push(t2);
                }
                f2++;
            }
        }
    }
}

template<
    unsigned windowWidth, // 512 only
    unsigned depthInputs,
    unsigned depthOutput
>
void TopK_MergeSortDF_V1_UnitMergeLast(
    Stream<CONFIG_DTYPE, depthInputs> &streamDataInL,
    Stream<CONFIG_DTYPE, depthInputs> &streamDataInR,
    Stream<CONFIG_DTYPE, depthOutput> &streamDataOutL,
    const unsigned dim0,
    const unsigned dim1){

    assert(dim1 % windowWidth == 0);
    assert(dim1 <= MaxSliceLen);
    //assert(windowWidth != (MaxSliceLen/2));

    constexpr unsigned win2 = (2 * windowWidth);
    constexpr unsigned MaxSliceLenBy2 = MaxSliceLen/2;

    const unsigned pairsToBeMerged = MaxSliceLen / win2;

    LoopPairs:
    for(unsigned pair=0; pair<pairsToBeMerged; pair++){
        CONFIG_DTYPE buffPair[win2];

        // ---------------------------------------------
        // 1. Fetch two pairs before merging
        LoopFetchPairs:
        for(unsigned w=0; w<win2; w++){
            if(w<windowWidth){
                buffPair[w] = streamDataInL.Pop();
            }else{
                buffPair[w] = streamDataInR.Pop();
            }
        }

        // ---------------------------------------------
        // 2. Merge the pairs
        unsigned f1 = 0;
        unsigned f2 = windowWidth;
        unsigned i2 = windowWidth;
        unsigned i3 = win2;
        if(i2 >= win2) i2 = win2;
        if(i3 >= win2) i3 = win2;

        LoopMerge1:
        for(unsigned i=0; i<win2; i++){
#pragma HLS PIPELINE II=1

            CONFIG_DTYPE t1 = buffPair[f1];
            CONFIG_DTYPE t2 = (f2 == i3) ? 0 : buffPair[f2];

            if(f2 == i3 || (f1 < i2 && t1 <= t2)) {
                streamDataOutL.Push(t1);
                f1++;
            } else {
                assert(f2 < i3);
                streamDataOutL.Push(t2);
                f2++;
            }
        }
    }
}

void TopK_MergeSortDF_V1_UnitMerge1(
    Stream<CONFIG_DTYPE, 8> &streamDataInL,
    Stream<CONFIG_DTYPE, 8> &streamDataInR,
    Stream<CONFIG_DTYPE, 2> &streamDataOutL,
    Stream<CONFIG_DTYPE, 2> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 1;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, 8, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge2(
    Stream<CONFIG_DTYPE, 2> &streamDataInL,
    Stream<CONFIG_DTYPE, 2> &streamDataInR,
    Stream<CONFIG_DTYPE, 4> &streamDataOutL,
    Stream<CONFIG_DTYPE, 4> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 2;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge4(
    Stream<CONFIG_DTYPE, 4> &streamDataInL,
    Stream<CONFIG_DTYPE, 4> &streamDataInR,
    Stream<CONFIG_DTYPE, 8> &streamDataOutL,
    Stream<CONFIG_DTYPE, 8> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 4;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge8(
    Stream<CONFIG_DTYPE, 8> &streamDataInL,
    Stream<CONFIG_DTYPE, 8> &streamDataInR,
    Stream<CONFIG_DTYPE, 16> &streamDataOutL,
    Stream<CONFIG_DTYPE, 16> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 8;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge16(
    Stream<CONFIG_DTYPE, 16> &streamDataInL,
    Stream<CONFIG_DTYPE, 16> &streamDataInR,
    Stream<CONFIG_DTYPE, 32> &streamDataOutL,
    Stream<CONFIG_DTYPE, 32> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 16;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge32(
    Stream<CONFIG_DTYPE, 32> &streamDataInL,
    Stream<CONFIG_DTYPE, 32> &streamDataInR,
    Stream<CONFIG_DTYPE, 64> &streamDataOutL,
    Stream<CONFIG_DTYPE, 64> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 32;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge64(
    Stream<CONFIG_DTYPE, 64> &streamDataInL,
    Stream<CONFIG_DTYPE, 64> &streamDataInR,
    Stream<CONFIG_DTYPE, 128> &streamDataOutL,
    Stream<CONFIG_DTYPE, 128> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 64;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge128(
    Stream<CONFIG_DTYPE, 128> &streamDataInL,
    Stream<CONFIG_DTYPE, 128> &streamDataInR,
    Stream<CONFIG_DTYPE, 256> &streamDataOutL,
    Stream<CONFIG_DTYPE, 256> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 128;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge256(
    Stream<CONFIG_DTYPE, 256> &streamDataInL,
    Stream<CONFIG_DTYPE, 256> &streamDataInR,
    Stream<CONFIG_DTYPE, 512> &streamDataOutL,
    Stream<CONFIG_DTYPE, 512> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 256;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge512(
    Stream<CONFIG_DTYPE, 512> &streamDataInL,
    Stream<CONFIG_DTYPE, 512> &streamDataInR,
    Stream<CONFIG_DTYPE, 1024> &streamDataOutL,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 512;

    TopK_MergeSortDF_V1_UnitMergeLast<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        //MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl; 
#endif

    #pragma HLS DATAFLOW

    Stream<CONFIG_DTYPE, 8> streamRead_W1[2];
#pragma HLS STREAM variable=streamRead_W1 depth=8

    Stream<CONFIG_DTYPE, 2> streamW1_W2[2];
#pragma HLS STREAM variable=streamReadToW1 depth=2

    Stream<CONFIG_DTYPE, 4> streamW2_W4[2];
#pragma HLS STREAM variable=streamReadToW1 depth=4

    Stream<CONFIG_DTYPE, 8> streamW4_W8[2];
#pragma HLS STREAM variable=streamReadToW1 depth=8

    Stream<CONFIG_DTYPE, 16> streamW8_W16[2];
#pragma HLS STREAM variable=streamReadToW1 depth=16

    Stream<CONFIG_DTYPE, 32> streamW16_W32[2];
#pragma HLS STREAM variable=streamReadToW1 depth=32

    Stream<CONFIG_DTYPE, 64> streamW32_W64[2];
#pragma HLS STREAM variable=streamReadToW1 depth=64

    Stream<CONFIG_DTYPE, 128> streamW64_W128[2];
#pragma HLS STREAM variable=streamReadToW1 depth=128

    Stream<CONFIG_DTYPE, 256> streamW128_W256[2];
#pragma HLS STREAM variable=streamReadToW1 depth=256

    Stream<CONFIG_DTYPE, 512> streamW256_W512[2];
#pragma HLS STREAM variable=streamReadToW1 depth=512

    Stream<CONFIG_DTYPE, 1024> streamW512_Write;
#pragma HLS STREAM variable=streamReadToW1 depth=1024


#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < 2; i++) {
        streamRead_W1[i].set_name(("streamRead_W1[" + std::to_string(i) + "]").c_str());
        streamW1_W2[i].set_name(("streamW1_W2[" + std::to_string(i) + "]").c_str());
        streamW2_W4[i].set_name(("streamW2_W4[" + std::to_string(i) + "]").c_str());
        streamW4_W8[i].set_name(("streamW4_W8[" + std::to_string(i) + "]").c_str());
        streamW8_W16[i].set_name(("streamW8_W16[" + std::to_string(i) + "]").c_str());
        streamW16_W32[i].set_name(("streamW16_W32[" + std::to_string(i) + "]").c_str());
        streamW32_W64[i].set_name(("streamW32_W64[" + std::to_string(i) + "]").c_str());
        streamW64_W128[i].set_name(("streamW64_W128[" + std::to_string(i) + "]").c_str());
        streamW128_W256[i].set_name(("streamW128_W256[" + std::to_string(i) + "]").c_str());
        streamW256_W512[i].set_name(("streamW256_W512[" + std::to_string(i) + "]").c_str());
    }
    streamW512_Write.set_name("streamW512_Write");
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitRead, 
        inputTn, 
        streamRead_W1[0], 
        streamRead_W1[1], 
        dim0, 
        dim1, 
        vecsPerSlice);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge1,
        streamRead_W1[0],
        streamRead_W1[1],
        streamW1_W2[0],
        streamW1_W2[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge2,
        streamW1_W2[0],
        streamW1_W2[1],
        streamW2_W4[0],
        streamW2_W4[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge4,
        streamW2_W4[0],
        streamW2_W4[1],
        streamW4_W8[0],
        streamW4_W8[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge8,
        streamW4_W8[0],
        streamW4_W8[1],
        streamW8_W16[0],
        streamW8_W16[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge16,
        streamW8_W16[0],
        streamW8_W16[1],
        streamW16_W32[0],
        streamW16_W32[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge32,
        streamW16_W32[0],
        streamW16_W32[1],
        streamW32_W64[0],
        streamW32_W64[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge64,
        streamW32_W64[0],
        streamW32_W64[1],
        streamW64_W128[0],
        streamW64_W128[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge128,
        streamW64_W128[0],
        streamW64_W128[1],
        streamW128_W256[0],
        streamW128_W256[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge256,
        streamW128_W256[0],
        streamW128_W256[1],
        streamW256_W512[0],
        streamW256_W512[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge512,
        streamW256_W512[0],
        streamW256_W512[1],
        streamW512_Write,
        dim0,
        dim1);

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitWrite, 
        outputTn, 
        streamW512_Write, 
        dim0, 
        dim1, 
        vecsPerSlice); //vecsPerOutputSlice 

    HLSLIB_DATAFLOW_FINALIZE();

}

extern "C"{
void task_topk(
        const MemoryPackF_t *inputTn,
        //MemoryPackI_t *indicesSplitedTn,
        MemoryPackF_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=indicesSplitedTn offset=slave bundle=gmem2 max_read_burst_length=2 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=kValue bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerSlice bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerOutputSlice bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TopK_MergeSortDF_V1(inputTn, indicesSplitedTn, dim0, dim1, kValue, vecsPerSlice, vecsPerOutputSlice);
}
}
