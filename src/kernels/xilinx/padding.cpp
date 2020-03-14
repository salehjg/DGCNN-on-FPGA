#include <cassert>
#include <stdio.h>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

#define LOCAL_BUFF_LEN 64

void PadLastDimSubWord(
    const MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
    const int reverseSwitch,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm){

    // Sub Vec Padding Kernel
    assert(dim1<CONFIG_M_AXI_WIDTH);
    assert(dim1Padded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Padded>dim1);
    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(lcm<=LOCAL_BUFF_LEN);

    //unsigned int lcm, gcd; 
    //lcm=LCM(dim1, CONFIG_M_AXI_WIDTH) and gcd = GCD(dim1 , CONFIG_M_AXI_WIDTH)

    const unsigned int bunchVecCount = lcm/CONFIG_M_AXI_WIDTH;
    const unsigned int bunchSliceCount = lcm/dim1;
    const unsigned int vecPerOutputSlice = dim1Padded/CONFIG_M_AXI_WIDTH;
    const unsigned int limitS=DivCeil<unsigned int>(dim0*dim1, CONFIG_M_AXI_WIDTH);
    const unsigned int limitD=DivCeil<unsigned int>(dim0*dim1Padded, CONFIG_M_AXI_WIDTH);
    unsigned int indxS, indxD;

    CONFIG_DTYPE buff[LOCAL_BUFF_LEN];
    MemoryPack_t tmpVec1;
    MemoryPack_t tmpVec2;


    LoopIter0:
    for(unsigned int iter=0; 
        iter<DivCeil<unsigned int>(dim0, bunchVecCount);
        iter++){

        LoopDim0Init:
        for(unsigned int id0=0; id0<bunchVecCount; id0++){
            #pragma HLS PIPELINE II=1
            // id0 is the local index of vector, NOT the index of slice in dim0.
            
            ///TODO: check limits for indxS 
            indxS = iter*bunchVecCount+id0;
            if(indxS<limitS){
                tmpVec1 = inputTn[indxS];
            
                LoopUnrol0:
                for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    buff[id0*CONFIG_M_AXI_WIDTH+i] = tmpVec1[i];
                }
            }
        }

        LoopIter1:
        for(unsigned int iter1=0;
            iter1<bunchSliceCount;
            iter1++){
            #pragma HLS PIPELINE II=1

            // Because we have "dim1<CONFIG_M_AXI_WIDTH", we can ignore the need 
            // for a "for-loop" of:
            // for(d1=0; d1<dim1Padded/CONFIG_M_AXI_WIDTH; d1++) 
            // and just put the slice in the first index(d1=0)
        	//for(unsigned int d1=0; d1<dim1Padded/CONFIG_M_AXI_WIDTH; d1++)
        	unsigned int d1=0;
            {
                if(d1==0){
                	LoopUnrol1:
					for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
						#pragma HLS UNROLL
						if(i<dim1){
							tmpVec2[i] = buff[iter1*dim1+i];
						}else{
						    tmpVec2[i] = 0;
						}
					}
                }

                indxD = iter*(bunchSliceCount*vecPerOutputSlice)+
                        iter1*vecPerOutputSlice+
                        d1;
                if(indxD<limitD){
                	outputTn[indxD] = tmpVec2;
                }
            }
        }

    }
}


extern "C" {
void task_pad_last_dim(
    const MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
    const int reverseSwitch,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm){

#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave    bundle=gmem0
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave    bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control

#pragma HLS INTERFACE s_axilite port=reverseSwitch  bundle=control

#pragma HLS INTERFACE s_axilite port=dim0           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1Padded     bundle=control
#pragma HLS INTERFACE s_axilite port=lcm            bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    PadLastDimSubWord(
        inputTn,
        outputTn,
        reverseSwitch,
        dim0,
        dim1,
        dim1Padded,
        lcm);
}
}
