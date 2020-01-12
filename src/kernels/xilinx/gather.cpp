#include "VectorizationHelper.h"
#include <stdio.h>
#include <cassert>

//The latency is reported for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20
template<typename DType, int VecDepthInput, int VecDepthIndicesOutput>
void Gather(
        VectorizedArray<DType, VecDepthInput> *inputTn,
        VectorizedArray<int, VecDepthIndicesOutput> *indicesTn,
        VectorizedArray<DType, VecDepthIndicesOutput> *outputTn,
        int indices_axis,
        int inputDim0,
        int inputDim1,
        int inputDim2,
        int indicesDim0,
        int indicesDim1,
        int indicesDim2){
    assert(inputDim0 == indicesDim0);
    assert(inputDim1 == indicesDim1);
    assert(indices_axis == 1);
    unsigned long indxS1, indxS2, indxD;
    unsigned long BxNxKxD = indicesDim0 * indicesDim1 * indicesDim2 * inputDim2;
    int d0idx, d1idx, d2idx,d2input;

    unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
    VectorizedArray<DType, VecDepthInput> inputCache;
#pragma HLS array_partition variable=inputCache complete dim=0

    unsigned long indicesCacheVecIdx, indicesCacheVecSubIdx, lastIndicesCacheVecIdx;
    VectorizedArray<int, VecDepthIndicesOutput> indicesCache;
#pragma HLS array_partition variable=indicesCache complete dim=0

    unsigned long outputCacheVecIdx, outputCacheVecSubIdx;
    VectorizedArray<DType, VecDepthIndicesOutput> outputCache;
#pragma HLS array_partition variable=outputCache complete dim=0

    d0idx = 0;
    d1idx = 0;
    d2idx = 0;
    d2input = 0;
    lastInputCacheVecIdx=-1; lastIndicesCacheVecIdx=-1;
    //Nested loop for B,N,K,D
    LoopIter: for(unsigned long iter=0; iter<BxNxKxD; iter++){
#pragma HLS LOOP_TRIPCOUNT min=6553600 max=6553600
        // Only calculate this on start of the loop D
        if(d2input==0){
            indxS1 = d0idx*indicesDim1*indicesDim2 + d1idx*indicesDim2 + d2idx;
        }

        //----------------------------------------
        indicesCacheVecIdx = FlatIdx_to_VecIdx(VecDepthIndicesOutput, indxS1);
        indicesCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthIndicesOutput, indxS1);
        if(indicesCacheVecIdx!=lastIndicesCacheVecIdx){
            indicesCache = indicesTn[indicesCacheVecIdx];
        }
        lastIndicesCacheVecIdx = indicesCacheVecIdx;

        //----------------------------------------
        indxS2 = d0idx*indicesDim1*inputDim2 + indicesCache.vec[indicesCacheVecSubIdx]*inputDim2 + d2input;

        //----------------------------------------
        inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepthInput, indxS2);
        inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthInput, indxS2);
        if(inputCacheVecIdx!=lastInputCacheVecIdx){
            inputCache = inputTn[inputCacheVecIdx];
        }
        lastInputCacheVecIdx = inputCacheVecIdx;

        //----------------------------------------
        indxD = d0idx*indicesDim1*indicesDim2*inputDim2 + d1idx*indicesDim2*inputDim2 + d2idx*inputDim2 + d2input;
        outputCacheVecIdx = FlatIdx_to_VecIdx(VecDepthIndicesOutput, indxD);
        outputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthIndicesOutput, indxD);

        //----------------------------------------
        outputCache.vec[outputCacheVecSubIdx] = inputCache.vec[inputCacheVecSubIdx];
        if(outputCacheVecSubIdx==(VecDepthIndicesOutput-1) || iter==(BxNxKxD-1)){
            outputTn[outputCacheVecIdx] = outputCache;
        }

        //========================================
        if(d2input==inputDim2-1){
            d2input = 0;
            if(d2idx==indicesDim2-1){
                d2idx = 0;
                if(d1idx==indicesDim1-1){
                    d1idx=0;
                    d0idx++;
                }else{
                    d1idx++;
                }
            }else{
                d2idx++;
            }
        }else{
            d2input++;
        }
    }
}

extern "C"{
void task_gather(
    VectorizedArray<float, CONFIG_GATHER_INPUTTN_M_AXI_WIDTH> *inputTn,
    VectorizedArray<int, CONFIG_M_AXI_WIDTH> *indicesTn,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
    int indices_axis,
    int inputDim0,
    int inputDim1,
    int inputDim2,
    int indicesDim0,
    int indicesDim1,
    int indicesDim2){
#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesTn      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=indicesTn      bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control

#pragma HLS INTERFACE s_axilite port=indices_axis   bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim0      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim1      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim2      bundle=control

#pragma HLS INTERFACE s_axilite port=indicesDim0    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim1    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim2    bundle=control

#pragma HLS INTERFACE s_axilite port=return         bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=indicesTn
#pragma HLS data_pack variable=outputTn

    Gather<float, CONFIG_GATHER_INPUTTN_M_AXI_WIDTH, CONFIG_M_AXI_WIDTH>(
        inputTn,
        indicesTn,
        outputTn,
        indices_axis,
        inputDim0,
        inputDim1,
        inputDim2,
        indicesDim0,
        indicesDim1,
        indicesDim2);
}
}
