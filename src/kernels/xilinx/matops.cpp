#include <cassert>
#include <stdio.h> 
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using MemoryPackK_t = hlslib::DataPack<CONFIG_DTYPE, CONFIG_M_AXI_WIDTH>;
#define CONFIG_ND_1D_MAX_SLICE 1024

/**
 * @brief      MatOps for two same-rank and same-shape input tensors such that: a <op> b, 1<=rank<=4
 *             The latency will be reported for 5x1024x1024 and 5x1024x1024
 *             
 * @param[in]  inputTn1  input tensor 1
 * @param[in]  inputTn2  input tensor 2
 * @param[out] outputTn  output tensor
 * @param[in]  dim0      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim1      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim2      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim3      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim0B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim1B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim2B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim3B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  rank1     inputTn1's rank which should be equals to rank2
 * @param[in]  rank2     inputTn2's rank which should be equals to rank1
 * @param[in]  mode      operation, 0:Add, 1:Sub, 2:Mul, 3:Div
 */
void matops_nd_nd(
        MemoryPackK_t *inputTn1,
        MemoryPackK_t *inputTn2,
        MemoryPackK_t *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int rank1,
        const int rank2,
        const int mode){
    
    assert(rank1==rank2);
    assert(rank1>=1 && rank1<=4);
    //assert(dim0==dim0B && dim1==dim1B && dim2==dim2B && dim3==dim3B);

    const unsigned int d0d1d2d3 = rank1==4 ? dim0*dim1*dim2*dim3 :
                                   rank1==3 ? dim1*dim2*dim3 :
                                   rank1==2 ? dim2*dim3 :
                                   dim3;

    const unsigned int packCount = DivCeil<unsigned int>(d0d1d2d3, CONFIG_M_AXI_WIDTH);

    MemoryPackK_t outputBuff;

    LoopMain: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=327680 max=327680

        if(mode==0)//Add
        {
            outputBuff = inputTn1[iter] + inputTn2[iter];
        }
        else if(mode==1)//Sub
        {
            outputBuff = inputTn1[iter] - inputTn2[iter];
        }
        else if(mode==2)//Mul (element wise)
        {
            outputBuff = inputTn1[iter] * inputTn2[iter];
        }
        else if(mode==3)//Div (element wise)
        {
            outputBuff = inputTn1[iter] / inputTn2[iter];
        }

        //-----------------------------------------------------
        outputTn[iter] = outputBuff;
    }
}


/**
 * @brief      MatOps for the first tensor of rank four and the second tensor of
 *             rank one. The latency will be reported for 5x1024x1024 and 1024
 *
 * @param[in]  inputTn1  input tensor 1
 * @param[in]  inputTn2  input tensor 2
 * @param[out] outputTn  output tensor
 * @param[in]  dim0      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim1      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim2      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim3      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim0B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim1B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim2B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  dim3B     inputTn2's shape aligned by last element of the shape vector to dim3B
 * @param[in]  rank1     inputTn1's rank
 * @param[in]  rank2     inputTn2's rank
 * @param[in]  mode      operation, 0:Add, 1:Sub, 2:Mul, 3:Div
 */
void matops_nd_1d(
        MemoryPackK_t *inputTn1,
        MemoryPackK_t *inputTn2,
        MemoryPackK_t *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int rank1,
        const int rank2,
        const int mode){
    assert(rank1!=rank2);
    assert(rank1>=1 && rank1<=4);
    assert(rank2==1);
    assert(dim3 == dim3B);

    const unsigned int d0d1d2d3 = rank1==4 ? dim0*dim1*dim2*dim3 :
                                   rank1==3 ? dim1*dim2*dim3 :
                                   rank1==2 ? dim2*dim3 :
                                   dim3;

    const unsigned int packCount = DivCeil<unsigned int>(d0d1d2d3, CONFIG_M_AXI_WIDTH);

    unsigned int indxS1;
    unsigned int d3, buff_d3;
    MemoryPackK_t outputBuff, tempBuff, tempBuff2;

    CONFIG_DTYPE buff[CONFIG_ND_1D_MAX_SLICE];
#pragma HLS array_partition variable=buff complete dim=0
    
    LoopReadTn2: for(int i=0; i < DivCeil<unsigned int>(dim3, CONFIG_M_AXI_WIDTH); i++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
        tempBuff = inputTn2[i];
        for(int j=0; j<CONFIG_M_AXI_WIDTH; j++){
#pragma HLS UNROLL
            buff[i*CONFIG_M_AXI_WIDTH + j] = tempBuff[j];
        }
        
    }

    LoopMain: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=327680 max=327680

        indxS1 = iter * CONFIG_M_AXI_WIDTH;
        d3 = indxS1 % dim3;

        LoopFill: for(int i=0; i<CONFIG_M_AXI_WIDTH; i++){
#pragma HLS UNROLL
            buff_d3 = (d3+i) % dim3;
            tempBuff2[i] = buff[buff_d3];
        }

        if(mode==0)//Add
        {
            outputBuff = inputTn1[iter] + tempBuff2;
        }
        else if(mode==1)//Sub
        {
            outputBuff = inputTn1[iter] - tempBuff2;
        }
        else if(mode==2)//Mul (element wise)
        {
            outputBuff = inputTn1[iter] * tempBuff2;
        }
        else if(mode==3)//Div (element wise)
        {
            outputBuff = inputTn1[iter] / tempBuff2;
        }

        //-----------------------------------------------------
        outputTn[iter] = outputBuff;
    }
}

/**
 * @brief      MatOps for the first input tensor of rank 1<=r<=4 and the second tensor of rank one and shape one, meaning a constant value.
 *             The latency will be reported for 5x1024x1024 and 1
 *
 * @param      inputTn1  input tensor 1
 * @param      inputTn2  input tensor 2
 * @param      outputTn  output tensor
 * @param[in]  dim0      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim1      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim2      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim3      inputTn1's shape aligned by last element of the shape vector to dim3
 * @param[in]  dim0B     inputTn2's shape aligned by last element of the shape vector to dim3(should be zero)
 * @param[in]  dim1B     inputTn2's shape aligned by last element of the shape vector to dim3(should be zero)
 * @param[in]  dim2B     inputTn2's shape aligned by last element of the shape vector to dim3(should be zero)
 * @param[in]  dim3B     inputTn2's shape aligned by last element of the shape vector to dim3(should be one)
 * @param[in]  rank1     inputTn1's rank
 * @param[in]  rank2     inputTn2's rank(should be 1)
 * @param[in]  mode      operation, 0:Add, 1:Sub, 2:Mul, 3:Div
 */
void matops_nd_1d_constant(
        MemoryPackK_t *inputTn1,
        MemoryPackK_t *inputTn2,
        MemoryPackK_t *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int rank1,
        const int rank2,
        const int mode){
    assert(rank1!=rank2 || (rank1==rank2 && rank2==1 && dim3B==1));
    assert(rank1>=1 && rank1<=4);
    assert(rank2==1);
    assert(dim3B == 1);

    const unsigned int d0d1d2d3 = rank1==4 ? dim0*dim1*dim2*dim3 :
                                   rank1==3 ? dim1*dim2*dim3 :
                                   rank1==2 ? dim2*dim3 :
                                   dim3;

    const unsigned int packCount = DivCeil<unsigned int>(d0d1d2d3, CONFIG_M_AXI_WIDTH);

    unsigned int indxS1;
    unsigned int d3, buff_d3;
    MemoryPackK_t outputBuff;
    
    CONFIG_DTYPE cte = (inputTn2[0])[0];
    MemoryPackK_t tempBuff(cte);

    LoopMain: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=327680 max=327680

        if(mode==0)//Add
        {
            outputBuff = inputTn1[iter] + tempBuff;
        }
        else if(mode==1)//Sub
        {
            outputBuff = inputTn1[iter] - tempBuff;
        }
        else if(mode==2)//Mul (element wise)
        {
            outputBuff = inputTn1[iter] * tempBuff;
        }
        else if(mode==3)//Div (element wise)
        {
            outputBuff = inputTn1[iter] / tempBuff;
        }

        //-----------------------------------------------------
        outputTn[iter] = outputBuff;
    }
}

extern "C" {
void task_matops(
        MemoryPackK_t *inputTn1,
        MemoryPackK_t *inputTn2,
        MemoryPackK_t *outputTn,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const int dim0B_IsNotZero,
        const int dim1B_IsNotZero,
        const int dim2B_IsNotZero,
        const int dim3B_IsNotZero,
        const int rank1,
        const int rank2,
        const int mode){

#pragma HLS INTERFACE m_axi     port=inputTn1           offset=slave    bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2           offset=slave    bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn           offset=slave    bundle=gmem3
#pragma HLS INTERFACE s_axilite port=inputTn1           bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2           bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn           bundle=control

#pragma HLS INTERFACE s_axilite port=dim0               bundle=control
#pragma HLS INTERFACE s_axilite port=dim1               bundle=control
#pragma HLS INTERFACE s_axilite port=dim2               bundle=control
#pragma HLS INTERFACE s_axilite port=dim3               bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B              bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B              bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B              bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B              bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B_IsNotZero    bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B_IsNotZero    bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B_IsNotZero    bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B_IsNotZero    bundle=control

#pragma HLS INTERFACE s_axilite port=rank1              bundle=control
#pragma HLS INTERFACE s_axilite port=rank2              bundle=control
#pragma HLS INTERFACE s_axilite port=mode               bundle=control

#pragma HLS INTERFACE s_axilite port=return             bundle=control

    if(rank1 == rank2 && !(rank2==1 && dim3B==1)){
        //printf("Selected Sub-kernel: matops_nd_nd\n");
        matops_nd_nd(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dim3,
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            rank1,
            rank2,
            mode);
    }else if(rank2 == 1 && dim3B!=1){
        //printf("Selected Sub-kernel: matops_nd_1d\n");
        matops_nd_1d(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dim3,
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            rank1,
            rank2,
            mode);
    }else if(rank2 == 1 && dim3B==1){
        //printf("Selected Sub-kernel: matops_nd_1d_constant\n");
        matops_nd_1d_constant(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dim3,
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            rank1,
            rank2,
            mode);
    }



}
}
