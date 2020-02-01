#include "AxiHelper.h"
#include "ReductionHelper.h"
#include "xilinx/config.h"
#include <cassert>
#include <hls_stream.h>

#define CONFIG_MAX_COMMON_DIM       1024

template<
    typename DType, 
    unsigned int VecDepthInputTn1, 
    unsigned int VecDepthInputTn2, 
    unsigned int MaxCommonDim, 
    unsigned int TileSizeOutputTn
    >
void SubfuncReadInputs(
        PackedArray<DType, VecDepthInputTn1> *inputTn1, //Same as MatA
        PackedArray<DType, VecDepthInputTn2> *inputTn2, //Same as MatB
        hls::stream<DType> (&outStream1)[TileSizeOutputTn*TileSizeOutputTn],
        hls::stream<DType> (&outStream2)[TileSizeOutputTn*TileSizeOutputTn],
        int dim0,
        int dim1A,
        int dim2B,
        int commonDim
){
    unsigned long inputTn1CacheVecIdx, inputTn1CacheVecSubIdx, lastInputTn1CacheVecIdx;
    PackedArray<DType, VecDepthInputTn1> inputTn1Cache;
#pragma HLS array_partition variable=inputTn1Cache complete dim=0

    unsigned long inputTn2CacheVecIdx, inputTn2CacheVecSubIdx, lastInputTn2CacheVecIdx;
    PackedArray<DType, VecDepthInputTn1> inputTn2Cache;
#pragma HLS array_partition variable=inputTn2Cache complete dim=0

    unsigned long indxS1, indxS2, indxD;
    int d2b,d1a;

    const int tileCount = ((dim1A-1)/TileSizeOutputTn+1) * ((dim2B-1)/TileSizeOutputTn+1);
    const int loopMainInnerBound = dim1A-TileSizeOutputTn;

    const int tripCount_tileCount = ((1024-1)/TileSizeOutputTn+1) * ((1024-1)/TileSizeOutputTn+1);

    LoopBatch: for(int batch=0; batch<dim0; batch++){
        d2b=0; d1a=0;
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopMain: for(unsigned int iter=0; iter<tileCount; iter++){
#pragma HLS LOOP_TRIPCOUNT min=tripCount_tileCount max=tripCount_tileCount
            if(d1a==0){
                // 1. Read a tile of inputTn2 which is slower than
                //    reading a tile of inputTn1 at the begining
                //    of the virtual dim1A's loop.

                int valid_w;
                if(d2b+TileSizeOutputTn > dim2B){
                    valid_w = dim2B - d2b;
                }else{
                    valid_w = TileSizeOutputTn;
                }

                lastInputTn2CacheVecIdx=-1;
                LoopReadTn2j: for(int bj=0; bj<commonDim; bj++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                    LoopReadTn2i: for(int bi=0; bi<TileSizeOutputTn; bi++){
#pragma HLS LOOP_TRIPCOUNT min=TileSizeOutputTn max=TileSizeOutputTn
#pragma HLS PIPELINE II=1
                        if(bi<valid_w){
                            indxS2 = (batch)*commonDim*dim2B +
                                     (bj)*dim2B +
                                     (d2b+bi);
                            inputTn2CacheVecIdx = FlatIdx_to_VecIdx(VecDepthInputTn2, indxS2);
                            inputTn2CacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthInputTn2, indxS2);
                            if(inputTn2CacheVecIdx != lastInputTn2CacheVecIdx){
                                inputTn2Cache = inputTn2[inputTn2CacheVecIdx];
                            }
                            lastInputTn2CacheVecIdx = inputTn2CacheVecIdx;
                            LoopFeedStream2: for(int uj=0; uj<TileSizeOutputTn; uj++){
#pragma HLS UNROLL
                            	outStream2[uj*TileSizeOutputTn + bi] << inputTn2Cache.vec[inputTn2CacheVecSubIdx];
                            }
                        }else{
                        	LoopFeedStream2Pad: for(int uj=0; uj<TileSizeOutputTn; uj++){
#pragma HLS UNROLL
                            	outStream2[uj*TileSizeOutputTn + bi] << 0;
                            }
                        }
                    }
                }
            }

            //-----------------------------------------------------
            // 2. Read a tile of inputTn1 which is faster, more
            //    frequently, at the inner virtual dim2b's loop

            int valid_h;
            if(d1a+TileSizeOutputTn > dim1A){
                valid_h = dim1A - d1a;
            }else{
                valid_h = TileSizeOutputTn;
            }

            lastInputTn1CacheVecIdx=-1;
            LoopReadTn1j: for(int aj=0; aj<TileSizeOutputTn; aj++){
#pragma HLS LOOP_TRIPCOUNT min=TileSizeOutputTn max=TileSizeOutputTn
                LoopReadTn1i: for(int ai=0; ai<commonDim; ai++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE II=1
                    if(aj<valid_h){
                        indxS1 = (batch)*dim1A*commonDim +
                                 (d1a+aj)*commonDim +
                                 (ai);
                        inputTn1CacheVecIdx = FlatIdx_to_VecIdx(VecDepthInputTn1, indxS1);
                        inputTn1CacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthInputTn1, indxS1);
                        if(inputTn1CacheVecIdx != lastInputTn1CacheVecIdx){
                            inputTn1Cache = inputTn1[inputTn1CacheVecIdx];
                        }
                        lastInputTn1CacheVecIdx = inputTn1CacheVecIdx;

                        LoopFeedStream1: for(int ui=0; ui<TileSizeOutputTn; ui++){
#pragma HLS UNROLL
                        	outStream1[aj*TileSizeOutputTn + ui] << inputTn1Cache.vec[inputTn1CacheVecSubIdx];
                        }
                    }else{
                    	LoopFeedStream1Pad: for(int ui=0; ui<TileSizeOutputTn; ui++){
#pragma HLS UNROLL
                        	outStream1[aj*TileSizeOutputTn + ui] << 0;
                        }
                    }

                }
            }

            if(d1a >= loopMainInnerBound){
                d1a=0;
                d2b+=TileSizeOutputTn;
            }else{
                d1a+=TileSizeOutputTn;
            }
        }
    }

}

template <
    typename DType,
    unsigned int MaxCommonDim
    >
void SubfuncCalculateOneOutputElement(
        hls::stream<DType> &inStream1,
        hls::stream<DType> &inStream2,
        hls::stream<DType> &outStream,
        unsigned int commonDim){
    //Simple for-loop reduction
    DType sum = 0;

    LoopReduce: for(unsigned int i=0; i<commonDim; i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE II=1
        sum += inStream1.read() * inStream2.read();
    }

    outStream << sum;
}


template<
    typename DType, 
    unsigned int VecDepthInputTn1, 
    unsigned int VecDepthInputTn2, 
    unsigned int MaxCommonDim, 
    unsigned int TileSizeOutputTn
    >
void SubfuncWriteOutput(
        DType *outputTn, 
        hls::stream<DType> (&inStream)[TileSizeOutputTn*TileSizeOutputTn],
        int dim0,
        int dim1A,
        int dim2B,
        int commonDim
){
    unsigned long indxD;
    int d2b,d1a;
    const int tileCount = ((dim1A-1)/TileSizeOutputTn+1) * ((dim2B-1)/TileSizeOutputTn+1);
    const int loopMainInnerBound = dim1A-TileSizeOutputTn;
    const int tripCount_tileCount = ((1024-1)/TileSizeOutputTn+1) * ((1024-1)/TileSizeOutputTn+1);
    int x,y,xx,yy;

    LoopBatch: for(int batch=0; batch<dim0; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopMain: for(unsigned int iter1=0; iter1<tileCount; iter1++){
#pragma HLS LOOP_TRIPCOUNT min=tripCount_tileCount max=tripCount_tileCount
            //Fused loops for (x,y), pay attention that virtual x-loop is the inner loop. So gmem writes will be in order.
            LoopProcess1:for(unsigned int iter2=0; iter2<TileSizeOutputTn*TileSizeOutputTn; iter2++){
#pragma HLS LOOP_TRIPCOUNT min=TileSizeOutputTn*TileSizeOutputTn max=TileSizeOutputTn*TileSizeOutputTn
#pragma HLS UNROLL
                if(iter2==0){
                    x=0;
                    y=0;
                }
                if(iter1==0){
                    d2b=0; d1a=0;
                }

                //========================================================
                //Writing output as row-major array.
                xx = x + d2b;
                yy = y + d1a;
                indxD = (batch)*dim1A*dim2B + (yy)*dim2B + (xx);
                DType rslt = inStream[iter2].read();
                if(yy<dim1A && xx<dim2B){
                    outputTn[indxD] = rslt;
                }

                //========================================================
                if(x == TileSizeOutputTn-1){
                    x=0;
                    y++;
                }else{
                    x++;
                }
                if(iter2==(TileSizeOutputTn*TileSizeOutputTn-1)){
                    if(d1a >= loopMainInnerBound){
                        d1a=0;
                        d2b+=TileSizeOutputTn;
                    }else{
                        d1a+=TileSizeOutputTn;
                    }
                }
            }

        }
    }

}



/*
* The reported latency is for Shape1=5x1024x64  Shape2=5x64x1024
*
* DType:            
* VecDepthInputTn1:
* VecDepthInputTn2:
* MaxCommonDim:
* TileSizeOutputTn: 
*/

template<
    typename DType, 
    unsigned int VecDepthInputTn1, 
    unsigned int VecDepthInputTn2, 
    unsigned int MaxCommonDim, 
    unsigned int TileSizeOutputTn
    >
void BatchMatMul(
    PackedArray<DType, VecDepthInputTn1> *inputTn1, //Same as MatA
    PackedArray<DType, VecDepthInputTn2> *inputTn2, //Same as MatB
    DType *outputTn, //Same as MatC = MatA x MatB
    int dim0,
    int dim1A,
    int dim2B,
    int commonDim){

    hls::stream<DType> inStreams1[TileSizeOutputTn*TileSizeOutputTn];
    hls::stream<DType> inStreams2[TileSizeOutputTn*TileSizeOutputTn];
    hls::stream<DType> outStreams[TileSizeOutputTn*TileSizeOutputTn];

#pragma HLS STREAM variable=inStreams1  depth=8
#pragma HLS STREAM variable=inStreams2  depth=8
#pragma HLS STREAM variable=outStreams  depth=8
#pragma HLS DATAFLOW

    // ======================================================================
    SubfuncReadInputs<
        DType, 
        VecDepthInputTn1,
        VecDepthInputTn2,
        MaxCommonDim, 
        TileSizeOutputTn>(
            inputTn1,
            inputTn2,
            inStreams1,
            inStreams2,
            dim0,
            dim1A,
            dim2B,
            commonDim);

    // ======================================================================
    For_Compute: for(int uc=0; uc<TileSizeOutputTn*TileSizeOutputTn; uc++){
#pragma HLS UNROLL
        
        SubfuncCalculateOneOutputElement<
            DType,
            MaxCommonDim>(
                inStreams1[uc],
                inStreams2[uc],
                outStreams[uc],
                commonDim);
    }  

    // ======================================================================
    SubfuncWriteOutput<
        DType,
        VecDepthInputTn1,
        VecDepthInputTn2,
        MaxCommonDim,
        TileSizeOutputTn>(
            outputTn, 
            outStreams,
            dim0,
            dim1A,
            dim2B,
            commonDim);
}

/*
template <typename DType>
void BatchMatMulAXI32(
    DType *inputTn1, //Same as MatA
    DType *inputTn2, //Same as MatB
    DType *outputTn,

    unsigned int dim0A, //Batch
    unsigned int dim1A, //rows
    unsigned int dim2A, //cols

    unsigned int dim0B, //Batch
    unsigned int dim1B, //rows
    unsigned int dim2B  //cols
){
    assert(dim0A==dim0B);
    assert(dim2A==dim1B);

    int commonDim = dim2A;
    unsigned long indxS1,indxS2,indxD;

    DType buff1[CONFIG_BUFF_DEPTH][CONFIG_MAX_COMMON_DIM]; //Row major buffer
    DType buff2[CONFIG_MAX_COMMON_DIM][CONFIG_BUFF_DEPTH]; //Row major buffer


    //Because burst reading a window of MatB is slower than MatA, outermost for-loop must belong to MatB to maximize data reuse for this matrix.
    LoopBatch:for(int batch=0; batch<dim0A; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopTiles1:for(int d2B=0; d2B<dim2B; d2B+=CONFIG_BUFF_DEPTH){
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
            LoopTiles2:for(int d1A=0; d1A<dim1A; d1A+=CONFIG_BUFF_DEPTH){
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
                //===============================================================================================================================
                //Sec. 1 : Burst read current window of matrix B only once in the begining of the loop. (TRYING TO MAXIMIZE DATA REUSE)
                //Avoiding logic between nested loops to make some optimization techniques possible.
                if(d1A==0){

                    int valid_w;
                    if(d2B+CONFIG_BUFF_DEPTH > dim2B){
                        valid_w = dim2B - d2B;
                    }else{
                        valid_w = CONFIG_BUFF_DEPTH;
                    }

                    LoopInitZero0:for(int j=commonDim; j<CONFIG_MAX_COMMON_DIM; j++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                        LoopInitZero1:for(int i=valid_w; i<CONFIG_BUFF_DEPTH; i++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                            buff2[j][i] = 0;
                        }
                    }

                    //burst reading the separated rows in burst mode
                    LoopBurstReadA_J:for(int j=0; j<commonDim;j++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        LoopBurstReadA_I:for(int i=0; i<valid_w;i++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
                            indxS2 = (batch)*commonDim*dim2B + (j)*dim2B + (d2B+i);
                            buff2[j][i] = inputTn2[indxS2];
                        }
                    }
                }

                //===============================================================================================================================
                //Sec. 2 : Burst read different windows of matrix A. (LESS DATA REUSE)
                int valid_h,valid_len;
                if(d1A+CONFIG_BUFF_DEPTH > dim1A){
                    valid_h = dim1A - d1A;
                }else{
                    valid_h = CONFIG_BUFF_DEPTH;
                }
                valid_len = valid_h * commonDim;

                 LoopInitZero2:for(int j=valid_h; j<CONFIG_BUFF_DEPTH; j++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                    LoopInitZero3:for(int i=commonDim; i<CONFIG_MAX_COMMON_DIM; i++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                        buff1[j][i] = 0;
                    }
                }

                //burst reading the whole buffer size worth of elements of matA
                LoopBurstReadB_J:for(int j=0; j<valid_h;j++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
                    LoopBurstReadB_I:for(int i=0; i<commonDim;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        indxS1 = (batch)*dim1A*commonDim + (d1A+j)*commonDim + (i);
                        buff1[j][i] = inputTn1[indxS1];
                    }
                }

                //=====================================================================================
                //Sec. 3: Process content of two local buffers and produce the output elements for (x, y) .
                int x,y;
                x=0;
                y=0;
                int xx,yy;
                //Fused loops for (x,y), pay attention that virtual x-loop is the inner loop. So gmem writes will be in order.
                LoopProcess1:for(int iter=0; iter<CONFIG_BUFF_DEPTH*CONFIG_BUFF_DEPTH; iter++){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                    DType sum=0;
                    LoopProcess2:for(int q=0; q<commonDim; q++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        sum += buff1[y][q] * buff2[q][x];
                    }
                    //Writing output as row-major array.
                    xx = x+d2B;
                    yy = y+d1A;
                    indxD = (batch)*dim1A*dim2B + (yy)*dim2B + (xx);

                    outputTnCacheVecIdx = FlatIdx_to_VecIdx(VecDepthOutputTn, indxD);
                    outputTnCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepthOutputTn, indxD);
                    outputTnCache.vec[outputTnCacheVecSubIdx] = sum;

                    if(yy<dim1A && xx<dim2B){
                        outputTn[indxD] = sum;;
                    }

                    //=====================================
                    if(x == CONFIG_BUFF_DEPTH-1){
                        x=0;
                        y++;
                    }else{
                        x++;
                    }
                }

            }
        }
    }

}
*/

extern "C"{
void task_matmul(
        PackedArray<float, CONFIG_MATMUL_INPUTTN1_M_AXI_WIDTH> *inputTn1,
        PackedArray<float, CONFIG_MATMUL_INPUTTN2_M_AXI_WIDTH> *inputTn2,
        float *outputTn,
        unsigned int dim0,
        unsigned int dim1A,
        unsigned int dim2B,
        unsigned int commonDim){

#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1A      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B      bundle=control
#pragma HLS INTERFACE s_axilite port=commonDim  bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn1
#pragma HLS data_pack variable=inputTn2

    BatchMatMul<
        float,
        CONFIG_MATMUL_INPUTTN1_M_AXI_WIDTH,
        CONFIG_MATMUL_INPUTTN2_M_AXI_WIDTH,
        CONFIG_MAX_COMMON_DIM,
        CONFIG_MATMUL_TILESIZE_SIZE>(
            inputTn1, 
            inputTn2, 
            outputTn,
            dim0,
            dim1A,
            dim2B,
            commonDim);

}
}
