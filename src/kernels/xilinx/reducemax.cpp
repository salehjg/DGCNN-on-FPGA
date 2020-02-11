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

#include <cassert>
#include <stdio.h> 
#include <math.h>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

#define CONFIG_SLICE_SIZE       1024 
#define CONFIG_SHIFT_SIZE       3

using MemoryPack_t = hlslib::DataPack<CONFIG_DTYPE, CONFIG_M_AXI_WIDTH>;
using hlslib::Stream;

void reducemax_rank3_ftf_shiftreg_scheme(
    MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim2){

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

extern "C" {
void task_reducemax(
    MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
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

    if(!overaxis0 && overaxis1 && !overaxis2){
        assert(dim2%CONFIG_M_AXI_WIDTH==0);
        reducemax_rank3_ftf_shiftreg_scheme(inputTn, outputTn, dim0, dim1, dim2);
    }

}

}

//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================


/*
void reducemax_rank3_ftf(        
    MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim2){

    CONFIG_DTYPE buff_rslt[CONFIG_SLICE_SIZE];
#pragma HLS ARRAY_PARTITION variable=buff_rslt cyclic factor=16 dim=0

    unsigned int d0d1d2 = dim0 * dim1 * dim2;
    unsigned int outputIdx;
    unsigned int packCount = DivCeil<unsigned int>(dim0*dim1*dim2, CONFIG_M_AXI_WIDTH);

    MemoryPack_t tmpBuff;
    MemoryPack_t outBuff;

    outputIdx = 0;


    LoopMain: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS LOOP_TRIPCOUNT min=1048576 max=1048576
#pragma HLS PIPELINE II=1

        const unsigned int idx = iter*CONFIG_M_AXI_WIDTH;
        const unsigned int d1 = idx / dim2;
        const unsigned int d2 = idx % dim2;
        tmpBuff = inputTn[iter];

        LoopCompare: for(unsigned int i =0; i<CONFIG_M_AXI_WIDTH; i++){
#pragma HLS DEPENDENCE variable=buff_rslt inter false
#pragma HLS UNROLL
            if(d1==0){
                buff_rslt[d2+i] = -1*INFINITY;
            }

            const CONFIG_DTYPE tVal = tmpBuff[i];
            if(buff_rslt[d2+i] < tVal){
                buff_rslt[d2+i] = tVal;
            }

            if(d1==(dim1-1)){
                outBuff[i] = buff_rslt[d2 + i];
            }
        }

        if(d1==(dim1-1)){
            outputTn[outputIdx++] = outBuff;
        }

    }

}


extern "C" {
void task_reducemax(
    MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
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

    if(!overaxis0 && overaxis1 && !overaxis2){
        assert(dim2%CONFIG_M_AXI_WIDTH==0);
        reducemax_rank3_ftf(inputTn, outputTn, dim0, dim1, dim2);
    }

}

}

*/


//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================
//=================================================================================================================================================


/*
#define N 16
#define M 16


void Stencil2D(float const memory_in[N * M], float memory_out[N * M]) {

  float aboveee[M];
  float center[M];

  // The first two rows are buffered in separate pipelines

  LoopInit0: for (int i = 0; i < M; ++i) {
    #pragma HLS PIPELINE
    aboveee[i] = memory_in[i];
  }

  LoopInit1: for (int i = 0; i < M; ++i) {
    #pragma HLS PIPELINE
    center[i] = memory_in[M + i];
  }

  // The remaining rows can be streamed

  LoopMain1: for (int i = 1; i < N - 1; ++i) {
      LoopMain2: for (int j = 0; j < M; ++j) {
      #pragma HLS PIPELINE II=1

      const auto below = memory_in[(i + 1) * M + j];

      constexpr float factor = 0.3333;
      const auto average = factor * (aboveee[j] + center[j] + below);

      aboveee[j] = center[j];
      center[j] = below;
      #pragma HLS DEPENDENCE variable=aboveee false
      #pragma HLS DEPENDENCE variable=center false
      #pragma HLS DEPENDENCE variable=aboveee intra RAW true

      memory_out[i * M + j] = average;
    }
  }
}

extern "C" {
// Top-level entry function, not relevant for this example
void task_reducemax(float const *in, float *out) {
  #pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave
  #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave
  #pragma HLS INTERFACE s_axilite port=in
  #pragma HLS INTERFACE s_axilite port=out
  #pragma HLS INTERFACE s_axilite port=return
    Stencil2D(in, out);
}
}

*/
