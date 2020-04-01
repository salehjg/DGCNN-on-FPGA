/*
* Shape=5x1024x3    FFT
* Shape=5x1024x64   FFT
*/

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

constexpr unsigned CONFIG_MAX_SLICE_SIZE = 64;

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
    constexpr unsigned buffVecCount = CONFIG_MAX_SLICE_SIZE/CONFIG_M_AXI_WIDTH;

    CONFIG_DTYPE buffResult1[CONFIG_MAX_SLICE_SIZE];
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

            CONFIG_DTYPE reduced = hlslib::TreeReduce<CONFIG_DTYPE, hlslib::op::Add<CONFIG_DTYPE>, CONFIG_MAX_SLICE_SIZE>(buffResult1);
            const unsigned vecOutSubIndex = batchD1%CONFIG_M_AXI_WIDTH;
            const unsigned vecOutIndex = batchD0*vecsPerSliceOut + batchD1/CONFIG_M_AXI_WIDTH;
            vecOut[vecOutSubIndex] = reduced;
            if( vecOutSubIndex==(CONFIG_M_AXI_WIDTH-1) || batchD1==(dim1-1) ){
                outputTn[vecOutIndex] = vecOut;
            }
        }
    }
}

extern "C" {
void task_reducesum(
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

    if(!overaxis0 && !overaxis1 && overaxis2){
        ReduceSum3Axis2_V1(inputTn, outputTn, dim0, dim1, dim2);
    }
}
}
