#include "AxiHelper.h"
#include "xilinx/config.h"
#include <stdio.h>
#include <cassert>

constexpr unsigned CONFIG_INDICES_MAX_SILCE_LEN = 32;

//The latency is reported for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20
void GatherAxis1_V1(
    const MemoryPackF_t *inputTn,
    const MemoryPackI_t *indicesTn,
    MemoryPackF_t *outputTn,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){

    assert(inputDim0 == indicesDim0); //B
    assert(inputDim1 == indicesDim1); //N
    const unsigned B = inputDim0;
    const unsigned N = inputDim1;
    const unsigned K = indicesDim2;
    const unsigned D = inputDim2;

    // IndicesTn: BxNxK
    // InputTn:   BxNxD
    
    const unsigned paddedK = MakeDivisible<unsigned>(K, CONFIG_M_AXI_WIDTH);
    const unsigned paddedD = MakeDivisible<unsigned>(D, CONFIG_M_AXI_WIDTH); 

    const unsigned vecsPerSliceIndicesTn = paddedK / CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceInputTn = paddedD / CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceOutputTn = vecsPerSliceInputTn;

    unsigned buffIndices[CONFIG_INDICES_MAX_SILCE_LEN];
#pragma HLS ARRAY_PARTITION variable=buffIndices cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    LoopB:
    for(unsigned b=0; b<B; b++){
        LoopN:
        for(unsigned n=0; n<N; n++){
            
            // Read an slice of indicesTn and buffer it.
            LoopReadIndices:
            for(unsigned ik=0; ik<vecsPerSliceIndicesTn; ik++){
                const unsigned indxI = b*N*vecsPerSliceIndicesTn + n*vecsPerSliceIndicesTn + ik;
                LoopFill1Unrolled:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    buffIndices[ik*CONFIG_M_AXI_WIDTH+i] = indicesTn[indxI];
                }
            }

            LoopK:
            for(unsigned k=0; k<K; k++){
                const unsigned currentLocalIndex = buffIndices[k];

                // Copy data slices related to 'currentLocalIndex'
                LoopD:
                for(unsigned id=0; id<vecsPerSliceInputTn; id++){
                    const unsigned indxS =  b*N*vecsPerSliceInputTn + 
                                            currentLocalIndex*vecsPerSliceInputTn +
                                            id;
                    const unsigned indxD =  b*N*K*vecsPerSliceOutputTn + 
                                            n*K*vecsPerSliceInputTn +
                                            k*vecsPerSliceInputTn +
                                            id;
                    outputTn[indxD] = inputTn[indxS];
                }
            }

        }
    }

}

extern "C"{
void task_gather(
    const MemoryPackF_t *inputTn,
    const MemoryPackI_t *indicesTn,
    MemoryPackF_t *outputTn,
    unsigned indicesAxis,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){
#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesTn      offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=indicesTn      bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control
#pragma HLS INTERFACE s_axilite port=indicesAxis   bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim0      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim1      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim2      bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim0    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim1    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim2    bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    if(indicesAxis==1){
        GatherAxis1_V1(
            inputTn,
            indicesTn,
            outputTn,
            inputDim0,
            inputDim1,
            inputDim2,
            indicesDim0,
            indicesDim1,
            indicesDim2);
    }
    
}
}
