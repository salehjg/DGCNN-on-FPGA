/*
ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 
ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 

ReduceMax: reductionDim=1, DIM2 SHOULD BE ONE, ARGS(0,1,2)=DIM0x(DIM1)xDIM3,
ReduceMax: reductionDim=2, ARGS(0,1,2)=[DIM0*DIM1]x(DIM2)xDIM3
*/

#include <cassert>
#include <iostream>
#include <limits>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskReduceMax;
using hlslib::Stream;

constexpr unsigned CONFIG_MAX_SLICE_SIZE = 1024;

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
    //FTF

    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim2Padded/CONFIG_M_AXI_WIDTH;
    constexpr unsigned buffVecCount = CONFIG_MAX_SLICE_SIZE/CONFIG_M_AXI_WIDTH;

    CONFIG_DTYPE buffResult1[CONFIG_MAX_SLICE_SIZE/CONFIG_M_AXI_WIDTH][CONFIG_M_AXI_WIDTH];
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

/*
#define CONFIG_SLICE_SIZE       1024 
#define CONFIG_SHIFT_SIZE       3

using MemoryPack_t = hlslib::DataPack<CONFIG_DTYPE, CONFIG_M_AXI_WIDTH>;
using hlslib::Stream;

void reducemax_rank3_ftf_shiftreg_scheme(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    MemoryPack_t buff[CONFIG_SHIFT_SIZE+1][CONFIG_SLICE_SIZE/CONFIG_M_AXI_WIDTH];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=16 dim=2

    MemoryPack_t rslt[CONFIG_SLICE_SIZE/CONFIG_M_AXI_WIDTH];
#pragma HLS ARRAY_PARTITION variable=rslt cyclic factor=16 dim=1

    const unsigned int vim2 = DivCeil<unsigned int>(dim2, CONFIG_M_AXI_WIDTH);
    printf("dim0: %d, dim1: %d, dim2: %d, vim2: %d\n", dim0, dim1, dim2, vim2);

    LoopD0: for(unsigned int d0=0; d0<dim0; d0++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5

        printf("#######d0: %d\n", d0);
        LoopInit0: for(unsigned int j=0; j<CONFIG_SHIFT_SIZE; j++){
            LoopInit1: for(unsigned int v2=0; v2<vim2; v2++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                LoopInitSubVec: for(unsigned int q=0; q<CONFIG_M_AXI_WIDTH; q++){
                    buff[j][v2][q] = -1*INFINITY;
                }
            }
        }

        LoopD1: for(unsigned int d1=0; d1<dim1; d1++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopV2: for(unsigned int v2=0; v2<vim2; v2++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                LoopSubVec: for(unsigned int q=0; q<CONFIG_M_AXI_WIDTH; q++){
                    buff[CONFIG_SHIFT_SIZE][v2][q] = (inputTn[d0*dim1*vim2+ d1*vim2+ v2][q] > buff[0][v2][q]) ?
                                            inputTn[d0*dim1*vim2+ d1*vim2+ v2][q] :
                                            buff[0][v2][q];
                }

                LoopShift: for(unsigned int j=0; j<CONFIG_SHIFT_SIZE; j++){
                    buff[j][v2] = buff[j+1][v2];
                }
            }

        }

        LoopFinishUp0: for(unsigned int v2=0; v2<vim2; v2++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            rslt[v2] = buff[0][v2];
        }

        LoopFinishUp1: for(unsigned int j=0; j<CONFIG_SHIFT_SIZE; j++){
            LoopFinishUp2: for(unsigned int v2=0; v2<vim2; v2++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                LoopFinshUpSubVec: for(unsigned int q=0; q<CONFIG_M_AXI_WIDTH; q++){
                    rslt[v2][q] = (rslt[v2][q] < buff[j][v2][q]) ? buff[j][v2][q] : rslt[v2][q];
                }
            }
        }

        LoopFinishUp3: for(unsigned int v2=0; v2<vim2; v2++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            outputTn[d0*vim2+v2] = rslt[v2];
        }
    }
}
*/


extern "C" {
void task_reducemax(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const int overaxis0,
    const int overaxis1,
    const int overaxis2){
#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=dim2 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis0 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1 bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if(!overaxis0 && overaxis1 && !overaxis2){
        //assert(dim2%CONFIG_M_AXI_WIDTH==0);
        //reducemax_rank3_ftf_shiftreg_scheme(inputTn, outputTn, dim0, dim1, dim2);

        ReduceMax3Axis1_V2(inputTn, outputTn, dim0, dim1, dim2);
    }

}

}
