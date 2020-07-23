#include <cassert>
#include <stdio.h> 
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

void _DataMoverV1(
#ifdef USEMEMORYBANK0
        MemoryPackF_t *dataBank0,
#endif
#ifdef USEMEMORYBANK1        
        MemoryPackF_t *dataBank1,
#endif
#ifdef USEMEMORYBANK2
        MemoryPackF_t *dataBank2,
#endif
#ifdef USEMEMORYBANK3        
        MemoryPackF_t *dataBank3,
#endif  
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    
#pragma HLS INLINE
   
#ifndef USEMEMORYBANK0 
    assert(srcBank!=0 && destBank!=0);
#endif
#ifndef USEMEMORYBANK1 
    assert(srcBank!=1 && destBank!=1);
#endif
#ifndef USEMEMORYBANK2 
    assert(srcBank!=2 && destBank!=2);
#endif
#ifndef USEMEMORYBANK3
    assert(srcBank!=3 && destBank!=3);
#endif
    LoopIter:
    for(unsigned iter=0; iter<vecCount; iter++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
#pragma HLS PIPELINE II=1

#ifdef USEMEMORYBANK0    
        const MemoryPackF_t buff0 = dataBank0[iter];
#endif  
#ifdef USEMEMORYBANK1        
        const MemoryPackF_t buff1 = dataBank1[iter];
#endif  
#ifdef USEMEMORYBANK2  
        const MemoryPackF_t buff2 = dataBank2[iter];
#endif  
#ifdef USEMEMORYBANK3  
        const MemoryPackF_t buff3 = dataBank3[iter];
#endif  

        //---------------------------------------------------------
        
        if(destBank==0){
#ifdef USEMEMORYBANK0  
            dataBank0[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
                buff0;
#endif  

        }else if(destBank==1){
#ifdef USEMEMORYBANK1  
            dataBank1[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff1;
#endif  
        }else if(destBank==2){
#ifdef USEMEMORYBANK2  
            dataBank2[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff2;
#endif  
        }else if(destBank==3){
#ifdef USEMEMORYBANK3  
            dataBank3[iter] = 
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff3;
#endif  
        }else{
            assert(0);
        }
    }
}

extern "C" {
void task_datamover(
#ifdef USEMEMORYBANK0
        MemoryPackF_t *dataBank0,
#endif
#ifdef USEMEMORYBANK1
        MemoryPackF_t *dataBank1,
#endif
#ifdef USEMEMORYBANK2
        MemoryPackF_t *dataBank2,
#endif
#ifdef USEMEMORYBANK3
        MemoryPackF_t *dataBank3,
#endif
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
#ifdef USEMEMORYBANK0
    #pragma HLS INTERFACE m_axi port=dataBank0 offset=slave bundle=gmem1 max_read_burst_length=64 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=dataBank0 bundle=control
#endif
#ifdef USEMEMORYBANK1
    #pragma HLS INTERFACE m_axi port=dataBank1 offset=slave bundle=gmem2 max_read_burst_length=64 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=dataBank1 bundle=control
#endif
#ifdef USEMEMORYBANK2
    #pragma HLS INTERFACE m_axi port=dataBank2 offset=slave bundle=gmem3 max_read_burst_length=64 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=dataBank2 bundle=control
#endif
#ifdef USEMEMORYBANK3
    #pragma HLS INTERFACE m_axi port=dataBank3 offset=slave bundle=gmem4 max_read_burst_length=64 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=dataBank3 bundle=control
#endif
#pragma HLS INTERFACE s_axilite port=srcBank bundle=control
#pragma HLS INTERFACE s_axilite port=destBank bundle=control
#pragma HLS INTERFACE s_axilite port=vecCount bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // For this kernel to be usable for OclTensorI, size of the data types should match. 
    assert(CONFIG_DTYPE_SIZE == sizeof(unsigned));

    _DataMoverV1(
#ifdef USEMEMORYBANK0
        dataBank0,
#endif
#ifdef USEMEMORYBANK1
        dataBank1,
#endif
#ifdef USEMEMORYBANK2
        dataBank2,
#endif
#ifdef USEMEMORYBANK3
        dataBank3,
#endif
        srcBank,
        destBank,
        vecCount);
}
}
