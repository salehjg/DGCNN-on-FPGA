#include <cassert>
#include <iostream>
#include <limits>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/TreeReduce.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using hlslib::Stream;

constexpr unsigned MAX_POW_Y_MINUS_ONE = (ConfigTaskReduce::Sum4D::MaxPowY-1);

/**
 * @brief      Reduces the input tensor of rank 3 over the axis 2.(FFT) 
 *             This kernel complies with the padded last dim policy:
 *                  1) For the input tensor, last dimension should be padded to be divisible by m_axi_width
 *                  2) For the output tensor which should be rank 2 of shape dim0xdim1Padded, the allocated 
 *                     memory should cover the padded elements(dim1 to dim1Padded).
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 */
void ReduceSum3Axis2_V1(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){
    
    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSliceIn = dim2Padded/CONFIG_M_AXI_WIDTH;
    const unsigned dim1Padded = MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSliceOut = dim1Padded/CONFIG_M_AXI_WIDTH;
    constexpr unsigned buffVecCount = ConfigTaskReduce::Sum3D::MaxSliceLen/CONFIG_M_AXI_WIDTH;

    CONFIG_DTYPE buffResult1[ConfigTaskReduce::Sum3D::MaxSliceLen];
#pragma HLS ARRAY_PARTITION variable=buffResult1 cyclic factor=16 dim=1
    
    MemoryPackF_t vecOut; 

    LoopBatch0:
    for(unsigned batchD0=0; batchD0<dim0; batchD0++){
		#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopBatch1:
        for(unsigned batchD1=0; batchD1<dim1; batchD1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

            LoopSlice0:
            for(unsigned iVec=0; iVec<buffVecCount; iVec++){
                const bool validAddress = iVec<vecsPerSliceIn;
                const unsigned indxS =  batchD0*dim1*vecsPerSliceIn+
                                        batchD1*vecsPerSliceIn+
                                        ((validAddress)?iVec:(vecsPerSliceIn-1));
                const MemoryPackF_t vec = inputTn[indxS];
        
                LoopSlice1Unrolled:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    buffResult1[iVec*CONFIG_M_AXI_WIDTH+i] = (validAddress)?vec[i]:0;
                }
            }

            CONFIG_DTYPE reduced = hlslib::TreeReduce<CONFIG_DTYPE, hlslib::op::Add<CONFIG_DTYPE>, ConfigTaskReduce::Sum3D::MaxSliceLen>(buffResult1);
            const unsigned vecOutSubIndex = batchD1%CONFIG_M_AXI_WIDTH;
            const unsigned vecOutIndex = batchD0*vecsPerSliceOut + batchD1/CONFIG_M_AXI_WIDTH;
            vecOut[vecOutSubIndex] = reduced;
            if( vecOutSubIndex==(CONFIG_M_AXI_WIDTH-1) || batchD1==(dim1-1) ){
                outputTn[vecOutIndex] = vecOut;
            }
        }
    }
}

void ReduceSumRank4Axes012_V3_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, ConfigTaskReduce::Sum4D::PipeDepth> &streamsData,
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
        Stream<MemoryPackF_t, ConfigTaskReduce::Sum4D::PipeDepth> &streamsData,
        MemoryPackF_t *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){

    assert(pow_y>=1 && pow_y<=ConfigTaskReduce::Sum4D::MaxPowY);

    CONFIG_DTYPE buffResult1[ConfigTaskReduce::Sum4D::MaxSliceLen];
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

    Stream<MemoryPackF_t, ConfigTaskReduce::Sum4D::PipeDepth> streamData;
#pragma HLS STREAM variable=streamData depth=ConfigTaskReduce::Sum4D::PipeDepth

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(ReduceSumRank4Axes012_V3_UnitRead, 
        inputTn, streamData, dim0, dim1, dim2, dim3);
    HLSLIB_DATAFLOW_FUNCTION(ReduceSumRank4Axes012_V3_UnitProcess, 
        streamData, outputTn, pow_y, dim0, dim1, dim2, dim3);

    HLSLIB_DATAFLOW_FINALIZE();
}

CONFIG_DTYPE _Max(CONFIG_DTYPE val1, CONFIG_DTYPE val2){
#pragma HLS INLINE
    return (val2>val1) ? val2 : val1;
}

/**
 * @brief      Reduces the input tensor of rank 3 in the middle axis(FTF) with the max op.
 *             Currently, 'LoopSlice0' achieves II=3.
 *             The latency will be reported for 5x1024x20x128.
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 */
void ReduceMax3Axis1_V2(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim2Padded/CONFIG_M_AXI_WIDTH;
    constexpr unsigned buffVecCount = ConfigTaskReduce::Max3D::MaxSliceLen/CONFIG_M_AXI_WIDTH;

    CONFIG_DTYPE buffResult1[buffVecCount][CONFIG_M_AXI_WIDTH];
#pragma HLS ARRAY_PARTITION variable=buffResult1 complete dim=2

    LoopD0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5120 max=5120

        LoopClear0:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            #pragma HLS LOOP_TRIPCOUNT min=buffVecCount max=buffVecCount
            #pragma HLS PIPELINE II=1
            LoopClear1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                buffResult1[iVec][i] = numeric_limits<CONFIG_DTYPE>::min();
            }
        }

        LoopD1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=20 max=20

            LoopSlice0:
            for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
                #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                #pragma HLS PIPELINE II=1
                const unsigned indxS = d0*dim1*vecsPerSlice + d1*vecsPerSlice + iVec;
                MemoryPackF_t vec = inputTn[indxS];
                LoopCompute:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    const CONFIG_DTYPE rslt = vec[i];
                    buffResult1[iVec][i] = _Max(buffResult1[iVec][i], rslt);
                }
            }
        }

        LoopOutput0:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
            #pragma HLS PIPELINE II=1
            const unsigned indxD = d0*vecsPerSlice + iVec;
            MemoryPackF_t outVec;

            LoopOutput1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                outVec[i] = buffResult1[iVec][i];
            }

            outputTn[indxD] = outVec;
        }
    }
}

extern "C" {

/**
 * @brief      Reduces the input tensor of rank 3 or rank 4 with one of the two operators(sum, max).
 *             Currently these modes are supported:
 *             1) ReduceSum, Rank3 & FFT
 *             2) ReduceSum, Rank4 & TTTF
 *             3) ReduceMax, Rank3 & FTF
 *             For mode(1) pow_y, dim3, and overaxis3 are don't cares.
 *             For mode(2) there is not any don't cares.
 *             For mode(3) pow_y, dim3, and overaxis3 are don't cares.
 *             
 *
 * @param[in]  inputTn    The input tn
 * @param      outputTn   The output tn
 * @param[in]  mode       The operation mode(1,2,or 3)
 * @param[in]  pow_y      The pow y
 * @param[in]  dim0       The dim 0
 * @param[in]  dim1       The dim 1
 * @param[in]  dim2       The dim 2
 * @param[in]  dim3       The dim 3
 */
void task_reduce(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned mode,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=pow_y bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=dim2 bundle=control
#pragma HLS INTERFACE s_axilite port=dim3 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    assert(mode==1 || mode==2 || mode==3);

    if(mode==1){
#ifdef KERNEL_LOGS
        cout<<"ReduceSum3Axis2_V1 is selected."<<endl;
#endif
        ReduceSum3Axis2_V1(inputTn, outputTn, dim0, dim1, dim2);
    }

    if(mode==2){
#ifdef KERNEL_LOGS
        cout<<"ReduceSumRank4Axes012_V3 is selected."<<endl;
#endif
        ReduceSumRank4Axes012_V3(inputTn, outputTn, pow_y, dim0, dim1, dim2, dim3); 
    }

    if(mode==3){
#ifdef KERNEL_LOGS
        cout<<"ReduceMax3Axis1_V2 is selected."<<endl;
#endif
        ReduceMax3Axis1_V2(inputTn, outputTn, dim0, dim1, dim2);
    }
    
}
}
