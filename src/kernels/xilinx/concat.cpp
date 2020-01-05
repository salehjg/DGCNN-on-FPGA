/*
Shape1=5x1024x1x128x,   , Shape2=5x1024x1x64x, 
Shape1=5x1024x1x192x,   , Shape2=5x1024x1x128x, 
Shape1=5x1024x1x64x,    , Shape2=5x1024x1x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,  
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,
*/

#include "VectorizationHelper.h"
#include <stdio.h>
template <typename DType, int VecDepth>
void concat2(
    VectorizedArray<DType, VecDepth> *inputTn1,
    VectorizedArray<DType, VecDepth> *inputTn2,
    VectorizedArray<DType, VecDepth> *outputTn,
    unsigned int dimA0,
    unsigned int dimA1,
    unsigned int dimA2,
    unsigned int dimA3,
    unsigned int dimB0,
    unsigned int dimB1,
    unsigned int dimB2,
    unsigned int dimB3){

    unsigned long cacheVecIdx1, cacheVecIdx2;
    unsigned long lastCacheVecIdx1=-1, lastCacheVecIdx2=-1;
    unsigned long outputVecIdx, outputSubVecIdx;
    unsigned long indxS1, indxS2, indxD;
    unsigned int  dimR0, dimR1, dimR2, dimR3;

    VectorizedArray<DType, VecDepth> cacheTn1;
    VectorizedArray<DType, VecDepth> cacheTn2;
    VectorizedArray<DType, VecDepth> buff;
#pragma HLS array_partition variable=cacheTn1 complete dim=0
#pragma HLS array_partition variable=cacheTn2 complete dim=0
#pragma HLS array_partition variable=buff complete dim=0


    dimR0 = dimA0;
    dimR1 = dimA1;
    dimR2 = dimA2;
    dimR3 = dimA3 + dimB3;

    indxS1 = 0;
    indxS2 = 0;

    Loop1: for(int d0=0;d0<dimR0;d0++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        Loop2: for(int d1=0;d1<dimR1;d1++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            Loop3: for(int d2=0;d2<dimR2;d2++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=20

                Loop4R: for(int d3=0;d3<dimR3;d3++){
#pragma HLS LOOP_TRIPCOUNT min=6 max=320
#pragma HLS PIPELINE II=1

                    //printf("d0:%d, d1:%d, d2:%d, d3:%d\n",d0,d1,d2,d3);
                    indxS1 =d0*dimA1*dimA2*dimA3 +
                            d1*dimA2*dimA3+
                            d2*dimA3;
                    indxS2 =d0*dimB1*dimB2*dimB3 +
                            d1*dimB2*dimB3+
                            d2*dimB3;
                    if(d3<dimA3){
                        indxS1 += d3;
                        indxS2 += 0;
                    }else{
                        indxS1 += dimA3-1;
                        indxS2 += d3 - dimA3;
                    }

                    indxD = (d0)*dimR1*dimR2*dimR3 +
                            (d1)*dimR2*dimR3+
                            (d2)*dimR3+
                            (d3);

                    //printf("indxS1:%d, indxS2:%d, indxD:%d\n",(int)indxS1, (int)indxS2,(int)indxD);      

                    //1. Cache needed input elements without wasting bandwidth
                    cacheVecIdx1 = FlatIdx_to_VecIdx(VecDepth, indxS1);
                    cacheVecIdx2 = FlatIdx_to_VecIdx(VecDepth, indxS2);
                    //printf("cacheVecIdx1:%d, cacheVecIdx2:%d\n",(int)cacheVecIdx1,(int)cacheVecIdx2);
                    if(cacheVecIdx1 != lastCacheVecIdx1){
                        cacheTn1 = inputTn1[cacheVecIdx1];
                        //printf("****input1 read vId=%d\n", (int)cacheVecIdx1);
                    }
                    if(cacheVecIdx2 != lastCacheVecIdx2){
                        cacheTn2 = inputTn2[cacheVecIdx2];
                        //printf("****input2 read vId=%d\n", (int)cacheVecIdx2);
                    }
                    lastCacheVecIdx1 = cacheVecIdx1;
                    lastCacheVecIdx2 = cacheVecIdx2;

                    //2. Use cached data to fill an output tensor vector consisting of 'VecDepth' number of 'DType' words
                    outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
                    outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
                    //printf("outputVecIdx:%d, outputSubVecIdx:%d\n",(int)outputVecIdx,(int)outputSubVecIdx);
                    if(d3<dimA3){
                        buff.vec[outputSubVecIdx] = cacheTn1.vec[FlatIdx_to_VecSubIdx(VecDepth, indxS1)];
                    }else{
                        buff.vec[outputSubVecIdx] = cacheTn2.vec[FlatIdx_to_VecSubIdx(VecDepth, indxS2)];
                    }
                    if(outputSubVecIdx==(VecDepth-1)){
                        //3. Write output buff when it's ready.
                        outputTn[FlatIdx_to_VecIdx(VecDepth, indxD)] = buff;
                        //printf("output write vId=%d\n", (int)FlatIdx_to_VecIdx(VecDepth, indxD));
                    }
                    //printf("\n\n");
                }


            }
        }
    }


}

extern "C" {
void task_concat(
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn1,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn2,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,

        unsigned int dimA0,
        unsigned int dimA1,
        unsigned int dimA2,
        unsigned int dimA3,

        unsigned int dimB0,
        unsigned int dimB1,
        unsigned int dimB2,
        unsigned int dimB3){

#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dimA0      bundle=control
#pragma HLS INTERFACE s_axilite port=dimA1      bundle=control
#pragma HLS INTERFACE s_axilite port=dimA2      bundle=control
#pragma HLS INTERFACE s_axilite port=dimA3      bundle=control
#pragma HLS INTERFACE s_axilite port=dimB0      bundle=control
#pragma HLS INTERFACE s_axilite port=dimB1      bundle=control
#pragma HLS INTERFACE s_axilite port=dimB2      bundle=control
#pragma HLS INTERFACE s_axilite port=dimB3      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn1
#pragma HLS data_pack variable=inputTn2
#pragma HLS data_pack variable=outputTn

    concat2<float, CONFIG_M_AXI_WIDTH>(
        inputTn1,
        inputTn2,
        outputTn,
        dimA0,
        dimA1,
        dimA2,
        dimA3,
        dimB0,
        dimB1,
        dimB2,
        dimB3);

}
}
