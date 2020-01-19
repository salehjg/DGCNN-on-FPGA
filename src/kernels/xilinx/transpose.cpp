// B=5, Dim1=1024, Dim2=1,3,64

#include "AxiHelper.h"
#include "xilinx/config.h"
#include <stdio.h>

#define CONFIG_BLOCK_WIDTH  16
#define CONFIG_BLOCK_HEIGHT 64

//Latency is for 5x1024x64 and CONFIG_W_H=(16,64)
template <typename DType>
void BatchTransposeAXI32(
    const DType* inputTn,
    DType* outputTn,
    int dim0,               //batch
    int dim1,               //rows
    int dim2                //columns
){
    DType buff[CONFIG_BLOCK_HEIGHT][CONFIG_BLOCK_WIDTH];
    unsigned long indxS,indxD;
    int tmp1, tmp2;
    // *******************************************
    // * inputTn must be row-major.
    // * dim0 = batch
    // * dim1 = rows
    // * dim2 = columns
    // *******************************************
    LoopBatch:for(int batch=0; batch<dim0; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopTiles1:for(int d2=0; d2<dim2; d2+=CONFIG_BLOCK_WIDTH){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            LoopTiles2:for(int d1=0; d1<dim1; d1+=CONFIG_BLOCK_HEIGHT){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                //=====================================================
                //Sec.1: Read current block into local buffer
                LoopReadJ:for(int j=0; j<CONFIG_BLOCK_HEIGHT; j++){
                    LoopReadI:for(int i=0; i<CONFIG_BLOCK_WIDTH; i++){
                        tmp1 = d1+j;
                        tmp2 = d2+i;
                        if(tmp1<dim1 && tmp2<dim2){
                            indxS = (batch)*dim1*dim2 +
                                    (tmp1)*dim2 +
                                    (tmp2);
                            buff[j][i] = inputTn[indxS];
                        }
                    }
                }

                //=====================================================
                //Sec.2: Write current block's data into output tensor
                LoopWriteI:for(int i=0; i<CONFIG_BLOCK_WIDTH; i++){
                    LoopWriteJ:for(int j=0; j<CONFIG_BLOCK_HEIGHT; j++){
                        tmp1 = d2+i;
                        tmp2 = d1+j;
                        if(tmp1<dim2 && tmp2<dim1){
                            indxD = (batch)*dim2*dim1 +
                                    (tmp1)*dim1+
                                    (tmp2);
                            outputTn[indxD] = buff[j][i];
                            //cout<<"IndexD = " << indxD << endl;
                        }
                    }
                }

                //=====================================================
            }
        }
    }
}

extern "C"{
void task_transpose(
        float *inputTn,
        float *outputTn,
        int dim0,
        int dim1,
        int dim2){
#pragma HLS INTERFACE m_axi     port=inputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn  offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  bundle=control

#pragma HLS INTERFACE s_axilite port=dim0      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

    BatchTransposeAXI32<float>(inputTn,outputTn,dim0,dim1,dim2);
}
}
