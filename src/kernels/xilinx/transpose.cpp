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
using namespace ConfigTaskTranspose;

/**
 * @brief      Batch transpose of the input tensor of rank 3.
 *             This kernel uses Axi32 for the input and the output tensors.
 *             This kernel complies with the padded last dim policy:
 *               1) inputTn and outputTn is considered to be padded in the last dim to 
 *                  be divisible by m_axi512's width.
 *               2) inputTn of shape axbx1 will be considered to be padded as axbx16(for m_axi512).
 *             The latency will be reported for an input tensor of shape 5x1024x64 and TileWidth and TileHeight of 16,64.
 *
 * @param[in]  inputTn   The input tn (row-major)
 * @param      outputTn  The output tn (row-major)
 * @param[in]  dim0      The dim 0 (batchSize)
 * @param[in]  dim1      The dim 1 (rows)
 * @param[in]  dim2      The dim 2 (cols)
 */
void BatchTransposeAXI32(
    const CONFIG_DTYPE* inputTn,
    CONFIG_DTYPE* outputTn,
    const unsigned dim0,               //batch
    const unsigned dim1,               //rows
    const unsigned dim2){                //columns

    const unsigned dim1Padded = MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);

    CONFIG_DTYPE buff[TileHeight][TileWidth];
    unsigned indxS,indxD;
    unsigned tmp1, tmp2;
    // *******************************************
    // * inputTn must be row-major.
    // * dim0 = batch
    // * dim1 = rows
    // * dim2 = columns
    // *******************************************
    LoopBatch:for(unsigned batch=0; batch<dim0; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopTiles1:for(unsigned d2=0; d2<dim2; d2+=TileWidth){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
            LoopTiles2:for(unsigned d1=0; d1<dim1; d1+=TileHeight){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
                //Sec.1: Read current block into local buffer
                LoopReadJ:for(unsigned j=0; j<TileHeight; j++){
                    LoopReadI:for(unsigned i=0; i<TileWidth; i++){
                        tmp1 = d1+j;
                        tmp2 = d2+i;
                        if(tmp1<dim1 && tmp2<dim2){
                            indxS = (batch)*dim1*dim2Padded +
                                    (tmp1)*dim2Padded +
                                    (tmp2);
                            buff[j][i] = inputTn[indxS];
                        }
                    }
                }

                //Sec.2: Write current block's data into output tensor
                LoopWriteI:for(unsigned i=0; i<TileWidth; i++){
                    LoopWriteJ:for(unsigned j=0; j<TileHeight; j++){
                        tmp1 = d2+i;
                        tmp2 = d1+j;
                        if(tmp1<dim2 && tmp2<dim1){
                            indxD = (batch)*dim2*dim1Padded +
                                    (tmp1)*dim1Padded+
                                    (tmp2);
                            outputTn[indxD] = buff[j][i];
                            //cout<<"IndexD = " << indxD << endl;
                        }
                    }
                }
            }
        }
    }
}

extern "C"{
void task_transpose(
        const float *inputTn,
        float *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2){
#pragma HLS INTERFACE m_axi     port=inputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2      bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    BatchTransposeAXI32(inputTn, outputTn, dim0, dim1, dim2);
}
}
