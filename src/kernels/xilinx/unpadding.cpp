#include <cassert>
#include <iostream>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskUnpadding;

/**
 * @brief      Unpads the padded input tensor on the last dimension.
 *             Currently 
 *                1)The input tensor's last dimension should be greater than m_axi_width and
 *                  should be divisible by m_axi_width.
 *                2)The same conditions as (1) are applied to 'dim1Unpadded'
 *
 * @param[in]  inputTn       The input tn
 * @param      outputTn      The output tn
 * @param[in]  dim0          The dim 0
 * @param[in]  dim1          The dim 1
 * @param[in]  dim1Unpadded  The dim 1 unpadded
 */
void UnpadLastDimSuperVec(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Unpadded){

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    //assert(dim1>=CONFIG_M_AXI_WIDTH); //XOCC crashes when this line is uncommented.
    assert(dim1%CONFIG_M_AXI_WIDTH==0);
    
    assert(dim1Unpadded>=CONFIG_M_AXI_WIDTH);
    assert(dim1Unpadded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Unpadded<dim1);

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);

    const auto idim1 = dim1/CONFIG_M_AXI_WIDTH;
    const auto idim1Unpadded = dim1Unpadded/CONFIG_M_AXI_WIDTH;
    unsigned int indxS, indxD;

#ifdef KERNEL_LOGS
    cout<<"idim1: "<<idim1<<endl;
    cout<<"idim1Unpadded: "<<idim1Unpadded<<endl;
#endif

    LoopD0:
    for(unsigned int d0=0; d0<dim0; d0++){
        LoopD1:
        for(unsigned int id1=0; id1<idim1Unpadded; id1++){
            #pragma HLS PIPELINE II=1
            indxS = d0*idim1+id1;
            indxD = d0*idim1Unpadded+id1;
#ifdef KERNEL_LOGS
            cout<<"## d0: "<<d0<<" id1: "<<id1<<" indxS: "<<indxS<<" indxD: "<<indxD<<endl;
#endif
            MemoryPackF_t tmpVec1 = inputTn[indxS];
            outputTn[indxD] = tmpVec1;
        }
    }
}

extern "C" {
void task_unpad_last_dim(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Unpadded){
#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave    bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave    bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control

#pragma HLS INTERFACE s_axilite port=dim0           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1Unpadded   bundle=control 
#pragma HLS INTERFACE s_axilite port=return         bundle=control
    if(dim1>CONFIG_M_AXI_WIDTH){
        UnpadLastDimSuperVec(inputTn, outputTn, dim0, dim1, dim1Unpadded);
    }
}
}
