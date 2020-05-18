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

constexpr unsigned MAX_POW_Y_MINUS_ONE = (MaxPowY-1);

void ReduceSumRank4Axes012_V3_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, PipeDepth> &streamsData,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3){

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    LoopBatch:
    for(unsigned batch=0; batch<batchSize; batch++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400

        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
#pragma HLS PIPELINE II=1
            const unsigned indxS = batch*vecsPerSlice + iVec;
            streamsData.Push(inputTn[indxS]);
        }

    }
}

void ReduceSumRank4Axes012_V3_UnitProcess(
        Stream<MemoryPackF_t, PipeDepth> &streamsData,
        MemoryPackF_t *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){

    assert(pow_y>=1 && pow_y<=MaxPowY);

    CONFIG_DTYPE buffResult1[MaxSliceLen];
#pragma HLS ARRAY_PARTITION variable=buffResult1 cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    unsigned indxS;

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned pow_y_minus_one = pow_y -1;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    LoopInit0:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        LoopInit1:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            buffResult1[iVec*CONFIG_M_AXI_WIDTH+i]=0;
        }
    }

    LoopBatch:
    for(unsigned batch=0; batch<batchSize; batch++){
        #pragma HLS LOOP_TRIPCOUNT min=102400 max=102400

        LoopSlice0:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
            #pragma HLS PIPELINE II=1

            indxS = batch*vecsPerSlice + iVec;
            MemoryPackF_t vec = streamsData.Pop();

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
                buffResult1[iVec*CONFIG_M_AXI_WIDTH + i] += rslt;
            }

        }
    }

    LoopSlice1:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        
        MemoryPackF_t outVec;

        LoopOutput:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            outVec[i]=buffResult1[iVec*CONFIG_M_AXI_WIDTH+i];
        }

        outputTn[iVec] = outVec;
    }
}

/**
 * @brief      Reduces the input tensor in the given dimensions.
 *             Currently, only TTTF reduction combination is supported.
 *             For 'LoopCompute', the best achievable II is 5.
 *             The latency is reported for inputTn of shape 5x1024x20x128
 *             This kernel complies with the padded last dim policy.
 *             This version(v3) alleviates external memory access stalls using dataflow scheme and FIFO depth.
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  pow_y     The pow y
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 * @param[in]  dim3      The dim 3
 */
void ReduceSumRank4Axes012_V3(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){
#pragma HLS DATAFLOW

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    Stream<MemoryPackF_t, PipeDepth> streamData;
#pragma HLS STREAM variable=streamData depth=PipeDepth

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(ReduceSumRank4Axes012_V3_UnitRead, 
        inputTn, streamData, dim0, dim1, dim2, dim3);
    HLSLIB_DATAFLOW_FUNCTION(ReduceSumRank4Axes012_V3_UnitProcess, 
        streamData, outputTn, pow_y, dim0, dim1, dim2, dim3);

    HLSLIB_DATAFLOW_FINALIZE();
}

/*
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

void ReduceSumRank4Axes012_V2(
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


    constexpr unsigned MAXSLICELEN_LOCAL = 1024;


    CONFIG_DTYPE buffResult1[MAXSLICELEN_LOCAL];
#pragma HLS ARRAY_PARTITION variable=buffResult1 cyclic factor=16 dim=1

    unsigned indxS;

    const unsigned batchSize = dim0*dim1*dim2;
    const unsigned pow_y_minus_one = pow_y -1;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    LoopInit0:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
        LoopInit1:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            buffResult1[iVec*CONFIG_M_AXI_WIDTH+i]=0;
        }
    }

    LoopBatch:
    for(unsigned batch=0; batch<batchSize; batch++){
        #pragma HLS LOOP_TRIPCOUNT min=102400 max=102400

        LoopSlice0:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
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
                buffResult1[iVec*CONFIG_M_AXI_WIDTH + i] += rslt;
            }

        }
    }


    LoopSlice1:
    for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        
        MemoryPackF_t outVec;

        LoopOutput:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            #pragma HLS UNROLL
            outVec[i]=buffResult1[iVec*CONFIG_M_AXI_WIDTH+i];
        }

        outputTn[iVec] = outVec;
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
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem1
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

    #ifdef KERNEL_LOGS
        cout<<"Simulation mode is enabled."<<endl;
    #endif

    //ReduceSumRank4Axes012_V1(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3);
    //ReduceSumRank4Axes012_V2(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3); 
    ReduceSumRank4Axes012_V3(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3); 
}
}
