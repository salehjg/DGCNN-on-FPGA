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
using namespace ConfigTaskTranspose;

constexpr int PIPEDEPTH = 16;

void BatchTranspose_V2_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<CONFIG_DTYPE, PIPEDEPTH> streamWords[CONFIG_M_AXI_WIDTH],
    const unsigned _dim0,
    const unsigned _dim1,
    const unsigned _dim2){

    const unsigned dim0 = _dim0;
    const unsigned dim1 = _dim1;
    const unsigned dim2 = _dim2;
    const unsigned vecsPerSlice = DivCeil<unsigned>(dim2, CONFIG_M_AXI_WIDTH);

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        LoopDim2:
        for(unsigned id2=0; id2<vecsPerSlice; id2++){
            LoopDim1:
            for(unsigned d1=0; d1<dim1; d1++){
                #pragma HLS PIPELINE II=1

                const unsigned indxS = d0*dim1*vecsPerSlice + d1*vecsPerSlice + id2;
                MemoryPackF_t vec = inputTn[indxS];

                LoopWords:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    CONFIG_DTYPE val = vec[i];
                    //cout<<"Pushed Val: "<<val<<endl;
                    streamWords[i].Push(val);
                }

            }
        }
    }

}

void BatchTranspose_V2_UnitWrite(
    Stream<CONFIG_DTYPE, PIPEDEPTH> streamWords[CONFIG_M_AXI_WIDTH],
    MemoryPackF_t *outputTn,
    const unsigned _dim0,
    const unsigned _dim1,
    const unsigned _dim2){

    const unsigned dim0 = _dim0;
    const unsigned dim1 = _dim2;
    const unsigned dim2 = _dim1;

    const unsigned vecsPerSlice = DivCeil<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned dim1ByAxiWidth = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        LoopiDim1:
        for(unsigned id1=0; id1<dim1ByAxiWidth; id1++){
            LoopDim2:
            for(unsigned id2=0; id2<vecsPerSlice; id2++){
                LoopDim1:
                for(unsigned dd1=0; dd1<CONFIG_M_AXI_WIDTH; dd1++){ 
                    #pragma HLS UNROLL
                    //cout<<"-------------"<<endl;
                    const unsigned d1 = id1*CONFIG_M_AXI_WIDTH + dd1;
                    MemoryPackF_t vec;

                    LoopTranspose01:
                    for(unsigned i=0; i<PIPEDEPTH; i++){
                        #pragma HLS PIPELINE II=1
                        CONFIG_DTYPE val = streamWords[dd1].Pop();
                        //cout<<"Popped Val: "<<val<<endl;
                        vec[i] = val;
                    }

                    const bool cond = d1<dim1;
                    const unsigned indxD = (cond)? d0*dim1*vecsPerSlice + d1*vecsPerSlice + id2 : 0;
                    if(cond){
                        outputTn[indxD] = vec;
                    }
                }
            }

        }
    }

}

void BatchTranspose_V2(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){
#pragma HLS DATAFLOW

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    assert(PIPEDEPTH==CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    Stream<CONFIG_DTYPE, PIPEDEPTH> streamWords[CONFIG_M_AXI_WIDTH];
#pragma HLS STREAM variable=streamWords depth=PIPEDEPTH

#ifndef HLSLIB_SYNTHESIS
    for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; ++i) {
        streamWords[i].set_name(("streamWords[" + std::to_string(i) + "]").c_str());
    }
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(BatchTranspose_V2_UnitRead, 
        inputTn, streamWords, dim0, dim1, dim2);
    HLSLIB_DATAFLOW_FUNCTION(BatchTranspose_V2_UnitWrite, 
        streamWords, outputTn, dim0, dim1, dim2);

    HLSLIB_DATAFLOW_FINALIZE();
}




extern "C"{
void task_transpose(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2){
#pragma HLS INTERFACE m_axi     port=inputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn  offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2      bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    BatchTranspose_V2(inputTn, outputTn, dim0, dim1, dim2);
}
}
