#include <cassert>
#include <stdio.h> 
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

void _DataMoverV1(
//        MemoryPackF_t *dataBank0,
        MemoryPackF_t *dataBank1,
        MemoryPackF_t *dataBank2,
        MemoryPackF_t *dataBank3,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
#pragma HLS inline
    
    //assert(srcBank>=0 && srcBank<=3);
    //assert(destBank>=0 && destBank<=3);
    assert(srcBank>=1 && srcBank<=3);
    assert(destBank>=1 && destBank<=3);

    LoopIter:
    for(unsigned iter=0; iter<vecCount; iter++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
#pragma HLS PIPELINE II=1
        //const MemoryPackF_t buff0 = dataBank0[iter];
        const MemoryPackF_t buff1 = dataBank1[iter];
        const MemoryPackF_t buff2 = dataBank2[iter];
        const MemoryPackF_t buff3 = dataBank3[iter];
        //---------------------------------------------------------
        /*
        if(destBank==0){
            dataBank0[iter] = (srcBank==0)? buff0:
                              (srcBank==1)? buff1:
                              (srcBank==2)? buff2:
                              buff3;

        }else*/ if(destBank==1){
            dataBank1[iter] = //(srcBank==0)? buff0:
                              (srcBank==1)? buff1:
                              (srcBank==2)? buff2:
                              buff3;

        }else if(destBank==2){
            dataBank2[iter] = //(srcBank==0)? buff0:
                              (srcBank==1)? buff1:
                              (srcBank==2)? buff2:
                              buff3;

        }else if(destBank==3){
            dataBank3[iter] = //(srcBank==0)? buff0:
                              (srcBank==1)? buff1:
                              (srcBank==2)? buff2:
                              buff3;

        }else{
            assert(0);
        }
    }
}

extern "C" {
void task_datamover(
        //MemoryPackF_t *dataBank0,
        MemoryPackF_t *dataBank1,
        MemoryPackF_t *dataBank2,
        MemoryPackF_t *dataBank3,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
//#pragma HLS INTERFACE m_axi port=dataBank0 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dataBank1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dataBank2 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=dataBank3 offset=slave bundle=gmem4
//#pragma HLS INTERFACE s_axilite port=dataBank0 bundle=control
#pragma HLS INTERFACE s_axilite port=dataBank1 bundle=control
#pragma HLS INTERFACE s_axilite port=dataBank2 bundle=control
#pragma HLS INTERFACE s_axilite port=dataBank3 bundle=control
#pragma HLS INTERFACE s_axilite port=srcBank bundle=control
#pragma HLS INTERFACE s_axilite port=destBank bundle=control
#pragma HLS INTERFACE s_axilite port=vecCount bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // For this kernel to be usable for OclTensorI, size of the data types should match. 
    assert(CONFIG_DTYPE_SIZE == sizeof(unsigned));

    _DataMoverV1(
//        dataBank0,
        dataBank1,
        dataBank2,
        dataBank3,
        srcBank,
        destBank,
        vecCount);
}
}
