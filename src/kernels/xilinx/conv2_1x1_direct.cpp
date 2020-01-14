#include "VectorizationHelper.h"
#include <cassert>

#define CONFIG_MAX_WEIGHT_BUF_D2    (328)
#define CONFIG_MAX_WEIGHT_BUF_D3    (1024)
#define CONFIG_MAX_WEIGHT_BUF_SIZE  (CONFIG_MAX_WEIGHT_BUF_D2*CONFIG_MAX_WEIGHT_BUF_D3)
#define CONFIG_UNROLL_FACTOR        2
#define CONFIG_REDUCTION_LEN        64


template<typename DType, int ReductionLen>
float ParallelReduction1D(
    const float *inputBuffer,
    const int len){

    DType lastResult=0;
    DType buff[ReductionLen];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    unsigned long iterations = ((len-1) / ReductionLen) + 1;
    unsigned long indxS, indxD;

    int tripLoopIter=1024/ReductionLen;

    LoopIter: for(unsigned long gIter=0; gIter<iterations; gIter++){
#pragma HLS LOOP_TRIPCOUNT min=tripLoopIter max=tripLoopIter
        // 1. read data into buff[:]
        LoopRead: for(int i=0;i<ReductionLen;i++){
#pragma HLS PIPELINE II=1
            indxS = gIter * ReductionLen + i;
            if(indxS<len){
                buff[i] = inputBuffer[indxS];
            }else{
                buff[i] = 0;
            }
            
        }

        //---------------------------------------
        LoopStride: for(int stride=ReductionLen/2; stride>0; stride/=2){
#pragma HLS PIPELINE II=1
            LoopReduce: for(int i=0; i<ReductionLen/2; i++){
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min=ReductionLen/2 max=ReductionLen/2
                if(i<stride){
                    buff[i] = buff[i] + buff[i+stride];
                }
            }
        }

        lastResult += buff[0];
    }

    return lastResult;
}

// Latency report is for InputTn=5x1024x1x320 and WeightTn=1x1x320x1024
template<typename DType, int VecDepth, int UnrollFactor, int ReductionLen>
void conv2_1x1_direct(
    VectorizedArray<DType, VecDepth> *inputTn,
    VectorizedArray<DType, VecDepth> *weightTn,
    VectorizedArray<DType, VecDepth> *biasTn,
    VectorizedArray<DType, VecDepth> *outputTn,
    unsigned int dim0D,
    unsigned int dim1D,
    unsigned int dim2D,
    unsigned int dim3D,
    unsigned int dim0W,
    unsigned int dim1W,
    unsigned int dim2W,
    unsigned int dim3W,
    unsigned int dim0B){

    unsigned long indxS1,indxS2,indxD;
    int b,n,k,ch;
    const unsigned long d0d1d2w3 = dim0D * dim1D * dim2D;

    DType buff_weight[CONFIG_MAX_WEIGHT_BUF_D2][CONFIG_MAX_WEIGHT_BUF_D3];
    DO_PRAGMA(HLS array_partition variable=buff_weight block factor=UnrollFactor dim=1)

    DType buff_bias[CONFIG_MAX_WEIGHT_BUF_D3];
    DO_PRAGMA(HLS array_partition variable=buff_bias block factor=UnrollFactor dim=1)

    DType buff_reduction[CONFIG_MAX_WEIGHT_BUF_D2];


    //Only 1x1 kernels
    assert(dim0W==1);
    assert(dim1W==1);
    assert(dim2W*dim3W<CONFIG_MAX_WEIGHT_BUF_SIZE);



    // 0. Loading entire biasTn on local memory considering it is small enough and of rank one.
    For_B1:for(int i3=0;i3<CONFIG_MAX_WEIGHT_BUF_D3;i3++){
        if(i3<dim3W){
            buff_bias[i3] = biasTn[FlatIdx_to_VecIdx(VecDepth, i3)].vec[FlatIdx_to_VecSubIdx(VecDepth, i3)];
        }else{
            buff_bias[i3] = 0;
        }
    }



    // 1. Loading entire weightTn on local memory considering it is small enough and of shape 1x1x*x*.
    For_W1:for(int i2=0;i2<CONFIG_MAX_WEIGHT_BUF_D2;i2++){
        For_W2:for(int i3=0;i3<CONFIG_MAX_WEIGHT_BUF_D3;i3++){
            if(i2<dim2W && i3<dim3W){
                indxS2 = i2*dim3W + i3;
                buff_weight[i2][i3] = weightTn[FlatIdx_to_VecIdx(VecDepth, indxS2)].vec[FlatIdx_to_VecSubIdx(VecDepth, indxS2)];
            }else{
                buff_weight[i2][i3] = 0;
            }
        }
    }



    // 2. Doing actual 1x1 convolution to mimic shared mlp layer. loops for b,n and k are fused together.
    b=0;n=0;k=0;
    For_Fused:for(unsigned long iter=0; iter<d0d1d2w3; iter++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120

        unsigned long outputCacheVecIdx, outputCacheVecSubIdx;
        VectorizedArray<DType, VecDepth> outputCache;
#pragma HLS array_partition variable=outputCache complete dim=0

        For_ChOut:for(int ch=0;ch<CONFIG_MAX_WEIGHT_BUF_D3;ch++){
DO_PRAGMA(HLS UNROLL factor=UnrollFactor)

            if(ch<dim3W){
                //DType sum=0;

                unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
                VectorizedArray<DType, VecDepth> inputCache;
#pragma HLS array_partition variable=inputCache complete dim=0
                lastInputCacheVecIdx=-1;

                // To avoid wasting m_axi bus bandwidth, dim3D should be devidable to VecDepth.
                For_D:for(int d=0;d<dim3D;d++){
#pragma HLS LOOP_TRIPCOUNT min=320 max=320
#pragma HLS PIPELINE II=1
                    //--------------------------------------------------------------
                    indxS1 = b*dim1D*dim2D*dim3D + n*dim2D*dim3D + k*dim3D + d;

                    //--------------------------------------------------------------
                    inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS1);
                    inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS1);
                    if(inputCacheVecIdx!=lastInputCacheVecIdx){
                        inputCache = inputTn[inputCacheVecIdx];
                    }
                    lastInputCacheVecIdx = inputCacheVecIdx;

                    //--------------------------------------------------------------
                    //sum += inputCache.vec[inputCacheVecSubIdx] * buff_weight[d][ch];
                    buff_reduction[d] = inputCache.vec[inputCacheVecSubIdx] * buff_weight[d][ch];
                }

                //--------------------------------------------------------------
                indxD = b*dim1D*dim2D*dim3W+ n*dim2D*dim3W+ k*dim3W+ ch;
                outputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
                outputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);

                //--------------------------------------------------------------
                DType sum = ParallelReduction1D<DType, ReductionLen>(buff_reduction, dim3D);
                outputCache.vec[outputCacheVecSubIdx] = sum + buff_bias[ch];

                //--------------------------------------------------------------
                if(outputCacheVecSubIdx == (VecDepth-1) || ch==(dim3W-1)){
                    outputTn[outputCacheVecIdx] = outputCache;
                }

            }
        }

        //==============================================================
        if(k == ((unsigned long)dim2D-1)){
            k=0;
            if(n == ((unsigned long)dim1D-1)){
                n=0;
                b++;
            }else{
                n++;
            }
        }else{
            k++;
        }
    }





}

extern "C"{
void task_conv2_1x1_direct(
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *weightTn,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *biasTn,
    VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
    unsigned int dim0D,
    unsigned int dim1D,
    unsigned int dim2D,
    unsigned int dim3D,
    unsigned int dim0W,
    unsigned int dim1W,
    unsigned int dim2W,
    unsigned int dim3W,
    unsigned int dim0B){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=weightTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=biasTn     offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=weightTn   bundle=control
#pragma HLS INTERFACE s_axilite port=biasTn     bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3D      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3W      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=weightTn
#pragma HLS data_pack variable=biasTn
#pragma HLS data_pack variable=outputTn

    conv2_1x1_direct<float, CONFIG_M_AXI_WIDTH, CONFIG_UNROLL_FACTOR, CONFIG_REDUCTION_LEN>(
        inputTn, 
        weightTn, 
        biasTn, 
        outputTn, 
        dim0D, 
        dim1D, 
        dim2D, 
        dim3D, 
        dim0W, 
        dim1W, 
        dim2W, 
        dim3W, 
        dim0B);

}
}
