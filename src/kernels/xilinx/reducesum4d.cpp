#include <cassert>
#include <iostream>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskReduceSum4D;
using hlslib::Stream;

// for dataflow version only:
//constexpr unsigned CONFIG_MAX_SLICE_SIZE = 1024;
//constexpr unsigned PipeDepth = 1;

constexpr unsigned MAX_POW_Y_MINUS_ONE = (MaxPowY-1);


/**
 * @brief      Reduces the input tensor in the given dimensions.
 *             Currently, only TTTF reduction combination is supported.
 *             For 'LoopCompute', the best achievable II is 4.
 *             The latency is reported for inputTn of shape 5x1024x20x128 
 *             This kernel complies with the padded last dim policy.
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  pow_y     The pow y
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 * @param[in]  dim3      The dim 3
 */
void ReduceSumRank4Axes012_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    assert(pow_y>=1 && pow_y<=MaxPowY);

    CONFIG_DTYPE buffResult1[CONFIG_M_AXI_WIDTH];
#pragma HLS ARRAY_PARTITION variable=buffResult1 complete dim=1

    unsigned indxS;

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned pow_y_minus_one = pow_y -1;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    LoopSlice0:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8

        LoopClear:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            buffResult1[i]=0;
        }

        LoopBatch:
        for(unsigned batch=0; batch<batchSize; batch++){
            #pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
            #pragma HLS PIPELINE II=1
            indxS = batch*vecsPerSlice + iVec;
            MemoryPackF_t vec = inputTn[indxS];
            LoopCompute:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                CONFIG_DTYPE rslt = vec[i];
                LoopPow:
                for(unsigned ipwr=0; ipwr<MAX_POW_Y_MINUS_ONE; ipwr++){
                    #pragma HLS UNROLL
                    if(ipwr<pow_y_minus_one){
                        rslt = rslt * rslt;
                    }
                }
                buffResult1[i] = buffResult1[i] + rslt;
            }
        }

        LoopOutput:
        MemoryPackF_t outVec;
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            outVec[i]=buffResult1[i];
        }
        outputTn[iVec] = outVec;
    }
}

/*
void UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, PipeDepth> &streamsData,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3){

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;
    
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        LoopBatch:
        for(unsigned batch=0; batch<batchSize; batch++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
#pragma HLS PIPELINE II=1
            const unsigned indxS = batch*vecsPerSlice + iVec;
            MemoryPackF_t vec = inputTn[indxS];
            streamsData.Push(vec);
        }
    }
}

void UnitPowY(
    Stream<MemoryPackF_t, PipeDepth> &streamsData,
    Stream<MemoryPackF_t, PipeDepth> &streamsPowOut,
    const unsigned pow_y,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3){

    const unsigned pow_y_minus_one = pow_y -1;
    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        LoopBatch:
        for(unsigned batch=0; batch<batchSize; batch++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
#pragma HLS PIPELINE II=1
            
            MemoryPackF_t vec = streamsData.Pop();
            MemoryPackF_t vecOut;
            LoopCompute:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
#pragma HLS UNROLL
                CONFIG_DTYPE rslt = vec[i];
                LoopPow:
                for(unsigned ipwr=0; ipwr<MAX_POW_Y_MINUS_ONE; ipwr++){
                    #pragma HLS UNROLL
                    if(ipwr<pow_y_minus_one){
                        rslt = rslt * rslt;
                    }
                }
                vecOut[i] = rslt;
            }
            streamsPowOut.Push(vecOut);
        }
    }
}

void UnitReduction(
    Stream<MemoryPackF_t, PipeDepth> &streamsPowIn,
    Stream<MemoryPackF_t, PipeDepth> &streamsResults,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3){

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    CONFIG_DTYPE buffResult[CONFIG_MAX_SLICE_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffResult cyclic factor=16 dim=1

    LoopInitBuffs:
    for(unsigned i=0; i<CONFIG_MAX_SLICE_SIZE; i++){
#pragma HLS PIPELINE II=1
        buffResult[i] = 0;
    }

    LoopSlice:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8

        MemoryPackF_t accVec(0.0f);

        LoopBatch:
        for(unsigned batch=0; batch<batchSize; batch++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
#pragma HLS PIPELINE II=4

            MemoryPackF_t tmpVec = streamsPowIn.Pop();
            accVec = accVec + tmpVec;
        }

        LoopReduceUnrolled:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++) {
#pragma HLS UNROLL
            buffResult[iVec*CONFIG_MAX_SLICE_SIZE+i] = accVec[i];
        }
    }

    LoopResults:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
#pragma HLS PIPELINE II=1

        MemoryPackF_t tmpVec(0.0f);
        LoopResultsUnrolled:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++) {
#pragma HLS UNROLL
            tmpVec[i] = buffResult[iVec*CONFIG_MAX_SLICE_SIZE+i];
        }
        streamsResults.Push(tmpVec);
    }
}

void UnitWrite(
    Stream<MemoryPackF_t, PipeDepth> &streamsResults,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3){

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
#pragma HLS PIPELINE II=1
        MemoryPackF_t vec = streamsResults.Pop();
        outputTn[iVec] = vec;
    }
}
*/

extern "C" {
void task_reducesum4d(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=pow_y bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=dim2 bundle=control
#pragma HLS INTERFACE s_axilite port=dim3 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis0 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis3 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

/*
    #ifdef KERNEL_LOGS
        cout<<"Simulation mode is enabled."<<endl;
    #endif
        
#pragma HLS DATAFLOW

    Stream<MemoryPackF_t, PipeDepth> streamData;
    Stream<MemoryPackF_t, PipeDepth> streamPow;
    Stream<MemoryPackF_t, PipeDepth> streamsResults; 
#pragma HLS STREAM variable=streamData depth=PipeDepth
#pragma HLS STREAM variable=streamPow depth=PipeDepth
#pragma HLS STREAM variable=streamsResults depth=PipeDepth

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(UnitRead, inputTn, streamData, dim0, dim1, dim2, dim3);
    HLSLIB_DATAFLOW_FUNCTION(UnitPowY, streamData, streamPow, pow_y, dim0, dim1, dim2, dim3);
    HLSLIB_DATAFLOW_FUNCTION(UnitReduction, streamPow, streamsResults, dim0, dim1, dim2, dim3);
    HLSLIB_DATAFLOW_FUNCTION(UnitWrite, streamsResults, outputTn, dim0, dim1, dim2, dim3);

    HLSLIB_DATAFLOW_FINALIZE();
*/
    ReduceSumRank4Axes012_V1(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3);

}
}
