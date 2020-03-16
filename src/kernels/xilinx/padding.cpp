#include <cassert>
#include <iostream>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

#define LOCAL_BUFF_LEN 64
using namespace std;

void PadLastDimSuperVec(
    MemoryPack_t const inputTn[],
    MemoryPack_t outputTn[],
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded){

#ifndef SYNTHESIS_MODE
    cout<<"Simulation mode is enabled."<<endl;
#endif

    assert(dim1>=CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);
    
    assert(dim1Padded>=CONFIG_M_AXI_WIDTH);
    assert(dim1Padded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Padded>dim1);

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);

    const auto idim1 = dim1/CONFIG_M_AXI_WIDTH;
    const auto idim1Padded = dim1Padded/CONFIG_M_AXI_WIDTH;
    unsigned int indxS, indxD;


#ifndef SYNTHESIS_MODE
    cout<<"idim1: "<<idim1<<endl;
    cout<<"idim1Padded: "<<idim1Padded<<endl;
#endif

    LoopD0:
    for(unsigned int d0=0; d0<dim0; d0++){
        LoopD1:
        for(unsigned int id1=0; id1<idim1; id1++){
            #pragma HLS PIPELINE II=1
            indxS = d0*idim1+id1;
            indxD = d0*idim1Padded+id1;
#ifndef SYNTHESIS_MODE
            cout<<"## d0: "<<d0<<" id1: "<<id1<<" indxS: "<<indxS<<" indxD: "<<indxD<<endl;
#endif
            MemoryPack_t tmpVec1 = inputTn[indxS];
            outputTn[indxD] = tmpVec1;
        }
    }


}

void PadLastDimSubVec(
    MemoryPack_t const inputTn[],
    MemoryPack_t outputTn[],
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm){

    // Sub Vec Padding Kernel

#ifndef SYNTHESIS_MODE
    cout<<"Simulation mode is enabled."<<endl;
#endif

    //assert(dim1<CONFIG_M_AXI_WIDTH); //XOCC crashes when this line is uncommented.
	//if(!(dim1<CONFIG_M_AXI_WIDTH)){assert(0);}

    assert(dim1Padded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Padded>dim1);
    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(lcm<=LOCAL_BUFF_LEN);

    const unsigned int bunchVecCount = lcm/CONFIG_M_AXI_WIDTH;
    const unsigned int bunchSliceCount = lcm/dim1;
    const unsigned int vecPerOutputSlice = dim1Padded/CONFIG_M_AXI_WIDTH;
    const auto limitS=DivCeil<unsigned int>(dim0*dim1, CONFIG_M_AXI_WIDTH);
    const auto limitD=DivCeil<unsigned int>(dim0*dim1Padded, CONFIG_M_AXI_WIDTH);
    unsigned int indxS, indxD, indxL1, indxL2;

    CONFIG_DTYPE buff[LOCAL_BUFF_LEN];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buff cyclic factor=CONFIG_M_AXI_WIDTH dim=1)

    MemoryPack_t tmpVec1, tmpVec2;
    const auto bunchCount = DivCeil<unsigned int>(dim0*dim1, lcm);
#ifndef SYNTHESIS_MODE
    cout<<"limitS: "<<limitS<<endl;
    cout<<"limitD: "<<limitD<<endl;
#endif
    LoopIter0:
    for(unsigned int iter=0; 
        iter<bunchCount;
        iter++){
        LoopDim0Init:
        for(unsigned int id0=0; id0<bunchVecCount; id0++){
            #pragma HLS PIPELINE II=1
#ifndef SYNTHESIS_MODE
            cout<<"## iter: " << iter << " id0: "<<id0<<endl;
#endif       
            // id0 is the local index of vector, NOT the index of slice in dim0.
            
            ///TODO: check limits for indxS 
            indxS = iter*bunchVecCount+id0;
            if(indxS<limitS){
#ifndef SYNTHESIS_MODE
                cout<<"*indxS: "<<indxS<<endl;
#endif
                tmpVec1 = inputTn[indxS];
            }else{
                for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    tmpVec1[i]=0;
                }
#ifndef SYNTHESIS_MODE
                cout<<"limitS is hit *indxS: "<<indxS<<endl;
#endif
            }

            LoopUnrol0:
            for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                indxL1=id0*CONFIG_M_AXI_WIDTH+i;
#ifndef SYNTHESIS_MODE
                cout<<"--indxL1: "<<indxL1<<endl;
#endif
                buff[indxL1] = tmpVec1[i];
            }

        }

        LoopIter1:
        for(unsigned int iter1=0;
            iter1<bunchSliceCount;
            iter1++){
            #pragma HLS PIPELINE II=1
#ifndef SYNTHESIS_MODE
            cout<<"## iter: " << iter << " iter1: "<<iter1<<endl;
#endif
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
						    indxL2=iter1*dim1+i;
#ifndef SYNTHESIS_MODE
						    cout<<"==indxL2: "<<indxL2<<endl;
#endif
							tmpVec2[i] = buff[indxL2];
						}else{
						    tmpVec2[i] = 0;
						}
					}
                }

                indxD = iter*(bunchSliceCount*vecPerOutputSlice)+
                        iter1*vecPerOutputSlice+
                        d1;
                if(indxD<limitD){
#ifndef SYNTHESIS_MODE
                    cout<<"**indxD: "<<indxD<<endl;
#endif
                	outputTn[indxD] = tmpVec2;
                }else{
#ifndef SYNTHESIS_MODE
                    cout<<"limitD is hit **indxD: "<<indxD<<endl;
#endif
                }
            }
        }

    }
}


extern "C" {
void task_pad_last_dim(
    MemoryPack_t const inputTn[],
    MemoryPack_t outputTn[],
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm){
#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave    bundle=gmem0
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave    bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control

#pragma HLS INTERFACE s_axilite port=dim0           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1Padded     bundle=control
#pragma HLS INTERFACE s_axilite port=lcm            bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    if(dim1<CONFIG_M_AXI_WIDTH){
        PadLastDimSubVec(
            inputTn,
            outputTn,
            dim0,
            dim1,
            dim1Padded,
            lcm);
    }else{
        PadLastDimSuperVec(
            inputTn,
            outputTn,
            dim0,
            dim1,
            dim1Padded);
    }
    
}
}
