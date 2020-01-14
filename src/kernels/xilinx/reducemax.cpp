/*
** ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
** ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 
**
** ReduceMax: reductionDim=1, DIM2 SHOULD BE ONE, ARGS(0,1,2)=DIM0x(DIM1)xDIM3,
** ReduceMax: reductionDim=2,                             , ARGS(0,1,2)=[DIM0*DIM1]x(DIM2)xDIM3
*/

#include "VectorizationHelper.h"
#include <stdio.h>

#define CONFIG_SLICE_SIZE       1024 

template <typename DType, int VecDepth>
void reducemax_rank3_ftf(        
    VectorizedArray<DType, VecDepth> *inputTn,
    VectorizedArray<DType, VecDepth> *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim2){

    float buff_tmp[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0
    float buff_rslt[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    unsigned long indxS, indxD;
    unsigned long d0d1d2 = dim0 * dim1 * dim2;
    int d0,d1,d2;

    unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
    unsigned long outputVecIdx, outputSubVecIdx;
    VectorizedArray<DType, VecDepth> inputCache;
    VectorizedArray<DType, VecDepth> outputCache;
#pragma HLS array_partition variable=inputCache complete dim=0
#pragma HLS array_partition variable=outputCache complete dim=0

    lastInputCacheVecIdx = -1;
    d0=0;d1=0;d2=0;

    LoopMain: for(unsigned long iter=0; iter<d0d1d2; iter++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400 
#pragma HLS PIPELINE II=1

        indxS = d0*dim1*dim2 + (d1)*dim2 + d2;

        //-----------------------------------------------------
        inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        if(inputCacheVecIdx != lastInputCacheVecIdx){
            inputCache = inputTn[inputCacheVecIdx];
        }
        lastInputCacheVecIdx = inputCacheVecIdx;

        //-----------------------------------------------------
        inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS);

        //-----------------------------------------------------
        if(d1==0){
            buff_rslt[d2] = inputCache.vec[inputCacheVecSubIdx];
        }else{
            buff_tmp[d2] = inputCache.vec[inputCacheVecSubIdx];
        }

        //-----------------------------------------------------
        if(d1!=0){
            if(buff_tmp[d2]>buff_rslt[d2]){
                buff_rslt[d2] = buff_tmp[d2];
            }
        }

        //-----------------------------------------------------
        if( d1 == ((unsigned long)dim1-1) ){
            // After processing all dim2 slices within current dim1 slice, write back 
            // reduced slice into output tensor
            indxD = d0*dim2 + d2;
            outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
            outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
            outputCache.vec[outputSubVecIdx] = buff_rslt[d2]; 
            if(outputSubVecIdx==(VecDepth-1) || iter==(d0d1d2-1)){
                outputTn[outputVecIdx] = outputCache;
            }
        }


        //=====================================================
        if( d2 == (dim2-1) ){
            d2=0;
            if( d1 == (dim1-1) ){
                d0++;
                d1=0;
            }else{
                d1++;
            }
        }else{
            d2++;
        }

    }
}

extern "C" {
void task_reducemax(
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim2,
    const int overaxis0,
    const int overaxis1,
    const int overaxis2){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control

#pragma HLS INTERFACE s_axilite port=overaxis0  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2  bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control
#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=outputTn

    if(!overaxis0 && overaxis1 && !overaxis2){
        reducemax_rank3_ftf<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, dim0, dim1, dim2);
    }

}

}