/*
* Shape=5x1024x3    FFT
* Shape=5x1024x64   FFT
*/
#include "AxiHelper.h"
#include "xilinx/config.h"
#include <hls_stream.h>

template <typename DType, int VecDepth>
void SubfuncSliceReadBurst(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        hls::stream<DType> &inStream,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2){

    unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx, indxS;
    PackedArray<DType, VecDepth> inputCache;
#pragma HLS array_partition variable=inputCache complete dim=0

    unsigned long d0d1d2 = dim0*dim1*dim2;
    int d0,d1,d2;

    d0=0;d1=0;d2=0;
    lastInputCacheVecIdx=-1;

    LoopReduce: for(unsigned long iter=0; iter<d0d1d2; iter++){
#pragma HLS LOOP_TRIPCOUNT min=15360 max=327680
#pragma HLS PIPELINE

        indxS = (d0)*dim1*dim2 + (d1)*dim2 + (d2);

        //----------------------------------------------------
        inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        if(inputCacheVecIdx!=lastInputCacheVecIdx){
            inputCache = inputTn[inputCacheVecIdx];
        }
        lastInputCacheVecIdx = inputCacheVecIdx;

        //----------------------------------------------------
        inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS);
        inStream << inputCache.vec[inputCacheVecSubIdx];

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

template <typename DType, int VecDepth>
void SubfuncSliceReduceSum(
        hls::stream<DType> &inStream,
        hls::stream<DType> &outStream,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2){

    //Simple for-loop reduction
    DType sum = 0;
    unsigned long d0d1d2 = dim0*dim1*dim2;
    int d0,d1,d2;

    d0=0;d1=0;d2=0;
    LoopReduce: for(unsigned long iter=0; iter<d0d1d2; iter++){
#pragma HLS LOOP_TRIPCOUNT min=15360 max=327680
#pragma HLS PIPELINE

        if(d2==0) sum=0;
        sum += inStream.read();
        if(d2==(dim2-1)) outStream<<sum;

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

template <typename DType, int VecDepth>
void SubfuncSliceWrite(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
        hls::stream<DType> &inStream,
        unsigned int dim0,
        unsigned int dim1){
    
    unsigned long outputVecIdx, outputSubVecIdx, indxD; 
    PackedArray<DType, VecDepth> outputCache; 
#pragma HLS array_partition variable=outputCache complete dim=0

    int d0,d1;
    unsigned long d0d1 = dim0*dim1;

    d0=0;d1=0;

    LoopWrite: for(unsigned long iter=0; iter<d0d1; iter++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
#pragma HLS PIPELINE

        indxD = d0*dim1 + d1;
        outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
        outputSubVecIdx  = FlatIdx_to_VecSubIdx(VecDepth, indxD);
        outputCache.vec[outputSubVecIdx] = inStream.read();
        if(outputSubVecIdx == VecDepth-1 || iter == (d0d1-1) ){
            outputTn[outputVecIdx] = outputCache;
        }

        //=====================================================
        if( d1 == (dim1-1) ){
            d0++;
            d1=0;
        }else{
            d1++;
        }
    }

}

// Dataflow Version
template <typename DType, int VecDepth>
void ReduceSumRank3Axis2(
        PackedArray<DType, VecDepth> *inputTn,
        PackedArray<DType, VecDepth> *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2){

    hls::stream<DType> datastream1;
    hls::stream<DType> datastream2;
#pragma HLS STREAM variable=datastream1  depth=32
#pragma HLS STREAM variable=datastream2  depth=32
#pragma HLS DATAFLOW

    // Read all slices of dim2 from input tensor(vectorized) into word stream(not vectorized)
    SubfuncSliceReadBurst<DType, VecDepth>(inputTn, datastream1, dim0, dim1, dim2);

    // Simple for-loop reduction
    SubfuncSliceReduceSum<DType, VecDepth>(datastream1, datastream2, dim0, dim1, dim2);

    // outputTn is of shape Dim0xDim1
    SubfuncSliceWrite<DType, VecDepth>(outputTn, datastream2, dim0, dim1);
}

extern "C" {
void task_reducesum(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        PackedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
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

    if(!overaxis0 && !overaxis1 && overaxis2){
        ReduceSumRank3Axis2<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, dim0, dim1, dim2);
    }
}
}
