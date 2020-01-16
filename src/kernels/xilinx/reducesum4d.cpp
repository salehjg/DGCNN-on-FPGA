/*
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
 */

#include "AxiHelper.h"
#include <stdio.h>

#define CONFIG_SLICE_SIZE       1024
#define CONFIG_MAX_POW_Y        3

#define MAX_POW_Y_MINUS_ONE     (CONFIG_MAX_POW_Y-1)

template<typename DType, int VecDepth>
void ReduceSumRank4Axes012(
        PackedArray<DType, VecDepth> *inputTn,
        PackedArray<DType, VecDepth> *outputTn,
        const int pow_y,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3){

    float buff_tmp[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0
    float buff_rslt[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    unsigned long indxS, indxD;
    unsigned long d0d1d2d3 = dim0 * dim1 * dim2 * dim3;
    int d0,d1,d2,d3;
    int pow_y_minus_one = pow_y -1;

    unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
    unsigned long outputVecIdx, outputSubVecIdx;
    PackedArray<DType, VecDepth> inputCache;
    PackedArray<DType, VecDepth> outputCache;
#pragma HLS array_partition variable=inputCache complete dim=0
#pragma HLS array_partition variable=outputCache complete dim=0

    lastInputCacheVecIdx = -1;
    d0=0;d1=0;d2=0;d3=0;

    LoopMain: for(unsigned long iter=0; iter<d0d1d2d3; iter++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400 
#pragma HLS PIPELINE II=1

        indxS = (d0)*dim1*dim2*dim3 + (d1)*dim2*dim3 + (d2)*dim3 + d3;

        //-----------------------------------------------------
        inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        if(inputCacheVecIdx != lastInputCacheVecIdx){
            inputCache = inputTn[inputCacheVecIdx];
        }
        lastInputCacheVecIdx = inputCacheVecIdx;

        //-----------------------------------------------------
        inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS);

        //-----------------------------------------------------
        // Init the rslt buffer
        if(d0==0 && d1==0 && d2==0) buff_rslt[d3] = 0;

        // Read into the temp buffer
        buff_tmp[d3] = inputCache.vec[inputCacheVecSubIdx];

        //-----------------------------------------------------
        float pow_rslt = buff_tmp[d3];
        LoopPow:for(int ipwr=0;ipwr<(MAX_POW_Y_MINUS_ONE);ipwr++){
            if(ipwr<pow_y_minus_one){
                pow_rslt = pow_rslt * pow_rslt;
            }
        }
        buff_rslt[d3] = buff_rslt[d3] + pow_rslt;

        //=====================================================
        if( d3 == (dim3-1) ){
            d3=0;
            if( d2 == (dim2-1) ){
                d2=0;
                if( d1 == (dim1-1)){
                    d1=0;
                    d0++;
                }else{
                    d1++;
                }
                
            }else{
                d2++;
            }
        }else{
            d3++;
        }
    }

    //-----------------------------------------------------
    //After processing all dim2 slices within the inputTn,
    //write back reduced slice into output tensor
    LoopWrite: for(unsigned int i=0; i<dim3; i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
        outputVecIdx = FlatIdx_to_VecIdx(VecDepth, i);
        outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, i);
        outputCache.vec[outputSubVecIdx] = buff_rslt[i]; 
        if(outputSubVecIdx==(VecDepth-1) || i==(dim3-1)){
            outputTn[outputVecIdx] = outputCache;
        }
    }
}

extern "C" {
void task_reducesum4d(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        PackedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
        const int pow_y,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3){

#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=pow_y      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control
#pragma HLS INTERFACE s_axilite port=dim3       bundle=control

#pragma HLS INTERFACE s_axilite port=overaxis0  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis3  bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=outputTn

    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3){
        ReduceSumRank4Axes012<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3);
    }
}
}