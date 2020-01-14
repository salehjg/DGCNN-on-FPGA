//Latency is for 5x1024x20x64 and 5x1024x20x64 ??
#include "VectorizationHelper.h"
#include <stdio.h>

template<typename DType, int VecDepth>
void _task_matops(
        VectorizedArray<DType, VecDepth> *inputTn1,
        VectorizedArray<DType, VecDepth> *inputTn2,
        VectorizedArray<DType, VecDepth> *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int dim0B_IsNotZero,
        const int dim1B_IsNotZero,
        const int dim2B_IsNotZero,
        const int dim3B_IsNotZero,
        const int mode){

    unsigned long indxS1,indxS2;
    const unsigned long d0d1d2d3 = dim0*dim1*dim2*dim3;
    int d0,d1,d2,d3;
    d0=0;d1=0;d2=0;d3=0;

    unsigned long inputCacheVecIdx1, inputCacheVecSubIdx1, lastInputCacheVecIdx1;
    unsigned long inputCacheVecIdx2, inputCacheVecSubIdx2, lastInputCacheVecIdx2;
    VectorizedArray<DType, VecDepth> inputCache1;
#pragma HLS array_partition variable=inputCache1 complete dim=0
    VectorizedArray<DType, VecDepth> inputCache2;
#pragma HLS array_partition variable=inputCache2 complete dim=0
    VectorizedArray<DType, VecDepth> outputCache;
#pragma HLS array_partition variable=outputCache complete dim=0

    lastInputCacheVecIdx1=-1;lastInputCacheVecIdx2=-1;

    //Fused loops for dim0 to dim3
    for(unsigned long iter=0; iter<d0d1d2d3; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=6553600 max=6553600
        //Content of loop dim3 should be here

        indxS1 =    d0 * dim1 * dim2 * dim3 +
                    d1 * dim2 * dim3 +
                    d2 * dim3 + 
                    d3;
        indxS2 =    d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
                    d1 * dim2B * dim3B * dim1B_IsNotZero +
                    d2 * dim3B * dim2B_IsNotZero +
                    d3 * dim3B_IsNotZero;

        inputCacheVecIdx1 = FlatIdx_to_VecIdx(VecDepth, indxS1);
        inputCacheVecIdx2 = FlatIdx_to_VecIdx(VecDepth, indxS2);
        inputCacheVecSubIdx1 = FlatIdx_to_VecSubIdx(VecDepth, indxS1);
        inputCacheVecSubIdx2 = FlatIdx_to_VecSubIdx(VecDepth, indxS2);
        if(inputCacheVecIdx1!=lastInputCacheVecIdx1){
            inputCache1 = inputTn1[inputCacheVecIdx1];
        }
        if(inputCacheVecIdx2!=lastInputCacheVecIdx2){
            inputCache2 = inputTn2[inputCacheVecIdx2];
        }
        lastInputCacheVecIdx1 = inputCacheVecIdx1;
        lastInputCacheVecIdx2 = inputCacheVecIdx2;

        if(mode==0)//Add
        {
            outputCache.vec[inputCacheVecSubIdx1] =
                inputCache1.vec[inputCacheVecSubIdx1] + inputCache2.vec[inputCacheVecSubIdx2];
        }
        else if(mode==1)//Sub
        {
            outputCache.vec[inputCacheVecSubIdx1] =
                inputCache1.vec[inputCacheVecSubIdx1] - inputCache2.vec[inputCacheVecSubIdx2];
        }
        else if(mode==2)//Mul (element wise)
        {
            outputCache.vec[inputCacheVecSubIdx1] =
                inputCache1.vec[inputCacheVecSubIdx1] * inputCache2.vec[inputCacheVecSubIdx2];
        }
        else if(mode==3)//Div (element wise)
        {
            outputCache.vec[inputCacheVecSubIdx1] =
                inputCache1.vec[inputCacheVecSubIdx1] / inputCache2.vec[inputCacheVecSubIdx2];
        }

        //-----------------------------------------------------
        if( inputCacheVecSubIdx1==(VecDepth-1) || iter==(d0d1d2d3-1) ){
            outputTn[inputCacheVecIdx1] = outputCache;
        }

        //=====================================================
        //House keeping if-statements for fused loops:
        if(d3==(dim3-1)){
            d3=0;
            if(d2==(dim2-1)){
                d2=0;
                if(d1==(dim1-1)){
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


}

extern "C" {
void task_matops(
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn1,
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn2,
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int dim0B_IsNotZero,
        const int dim1B_IsNotZero,
        const int dim2B_IsNotZero,
        const int dim3B_IsNotZero,
        const int mode){

#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control
#pragma HLS INTERFACE s_axilite port=dim3       bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B_IsNotZero  bundle=control

#pragma HLS INTERFACE s_axilite port=mode       bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn1
#pragma HLS data_pack variable=inputTn2
#pragma HLS data_pack variable=outputTn

    _task_matops<float, CONFIG_M_AXI_WIDTH>(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dim3,
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            dim0B_IsNotZero,
            dim1B_IsNotZero,
            dim2B_IsNotZero,
            dim3B_IsNotZero,
            mode);
}
}
