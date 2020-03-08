/*
Shape1=5x1024x1x128x,   , Shape2=5x1024x1x64x, 
Shape1=5x1024x1x192x,   , Shape2=5x1024x1x128x, 
Shape1=5x1024x1x64x,    , Shape2=5x1024x1x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,  
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,
*/

#include <cassert>
#include <stdio.h> 
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using MemoryPack_t = hlslib::DataPack<CONFIG_DTYPE, CONFIG_M_AXI_WIDTH>;
using MemoryPackWithFlags_t = hlslib::DataPack<CONFIG_DTYPE, CONFIG_M_AXI_WIDTH+1>;
using hlslib::Stream;
#define CONFIG_MAX_SLICE 192
#define CONFIG_STREAM1_DEPTH 2
#define CONFIG_STREAM2_DEPTH 2
#define CONFIG_STREAMo_DEPTH 2

typedef struct FlagsWord FlagsWord_t;

struct FlagsWord
{
    unsigned char flag1;
    unsigned char flag2;
    unsigned char flag3;
    unsigned char flag4;
};

FlagsWord_t concat2_decode_flags(MemoryPackWithFlags_t data){
#pragma HLS INLINE
    FlagsWord_t *flags;
    CONFIG_DTYPE flagsWord = data[16];
    flags = reinterpret_cast<struct FlagsWord*>(&flagsWord);
    return *flags;
}

MemoryPackWithFlags_t concat2_encode_flags(MemoryPack_t data, unsigned char flag1, unsigned char flag2, unsigned char flag3, unsigned char flag4){
#pragma HLS INLINE

    struct FlagsWord flags;
    flags.flag1 = flag1;
    flags.flag2 = flag2;
    flags.flag3 = flag3;
    flags.flag4 = flag4;
    MemoryPackWithFlags_t dataWithFlags;
    LoopEncode: for(int i=0; i<CONFIG_M_AXI_WIDTH; i++){
#pragma HLS UNROLL
        dataWithFlags[i] = data[i];
    }
    dataWithFlags[CONFIG_M_AXI_WIDTH] = *reinterpret_cast<CONFIG_DTYPE *>(&flags);
    return dataWithFlags;
}

void concat2_read1_rank4_over3(
    MemoryPack_t *inputTn1,
    Stream<MemoryPackWithFlags_t, CONFIG_STREAM1_DEPTH> &outStream,
    unsigned int dimA0,
    unsigned int dimA1,
    unsigned int dimA2,
    unsigned int dimA3,
    unsigned int dimA3Padded){

    MemoryPack_t tempBuff;
    MemoryPackWithFlags_t outBuff;
    const unsigned int packsPerSlice = (dimA3Padded/CONFIG_M_AXI_WIDTH);
    const unsigned int packCount = dimA0*dimA1*dimA2*packsPerSlice;
    unsigned char isLastPackOfSlice, wordCount;

    LoopRead: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=0

        isLastPackOfSlice = (iter+1)%packsPerSlice==0;
        wordCount = (isLastPackOfSlice==0) ? CONFIG_M_AXI_WIDTH : (dimA3 % CONFIG_M_AXI_WIDTH);

        tempBuff = inputTn1[iter]; 

        // flag1 = is this the last pack of the current slice (0: no, 1: yes)
        // flag2 = word count in the current pack(16: if this is not the last pack, dim3%AXI: otw)
        outBuff = concat2_encode_flags(tempBuff, isLastPackOfSlice, wordCount,0,0); 

        outStream.Push(outBuff);
    }
}

void concat2_read2_rank4_over3(
    MemoryPack_t *inputTn2,
    Stream<MemoryPackWithFlags_t, CONFIG_STREAM2_DEPTH> &outStream,
    unsigned int dimB0,
    unsigned int dimB1,
    unsigned int dimB2,
    unsigned int dimB3,
    unsigned int dimB3Padded){

    MemoryPack_t tempBuff;
    MemoryPackWithFlags_t outBuff;
    const unsigned int packsPerSlice = (dimB3Padded/CONFIG_M_AXI_WIDTH);
    const unsigned int packCount = dimB0*dimB1*dimB2*packsPerSlice;
    unsigned char isLastPackOfSlice, wordCount;

    LoopRead: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=0

        isLastPackOfSlice = (iter+1)%packsPerSlice==0;
        wordCount = (isLastPackOfSlice==0) ? CONFIG_M_AXI_WIDTH : (dimB3 % CONFIG_M_AXI_WIDTH);

        tempBuff = inputTn2[iter]; 

        // flag1 = is this the last pack of the current slice (0: no, 1: yes)
        // flag2 = word count in the current pack(16: if this is not the last pack, dim3%AXI: otw)
        outBuff = concat2_encode_flags(tempBuff, isLastPackOfSlice, wordCount,0,0); 

        outStream.Push(outBuff);
    }
}

void concat2_compute_rank4_over3(
    Stream<MemoryPackWithFlags_t, CONFIG_STREAM1_DEPTH> &inStream1,
    Stream<MemoryPackWithFlags_t, CONFIG_STREAM2_DEPTH> &inStream2,
    Stream<MemoryPack_t, CONFIG_STREAMo_DEPTH> &outStream,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int dimA3, 
    unsigned int dimA3Padded, 
    unsigned int dimB3,
    unsigned int dimB3Padded,
    unsigned int dimR3,
    unsigned int dimR3Padded){

    const unsigned int d0d1d2 = dim0*dim1*dim2; // total number of slices in the inputs or the output tensor.
    const unsigned int packsPerSlice = (dimR3Padded/CONFIG_M_AXI_WIDTH);
    const unsigned int packCount = d0d1d2*packsPerSlice;

    MemoryPackWithFlags_t cacheInput1; 
    MemoryPackWithFlags_t cacheInput2; 
    MemoryPack_t cacheOutput;

    unsigned char isFinished1=0, isFirst2=0, offset=0;

    LoopMain: for(unsigned int iter=0; iter<packCount; iter++){
        if(isFinished1==0){
            cacheInput1 = inStream1.Pop();
            FlagsWord_t flagsA = concat2_decode_flags(cacheInput1);
            if(flagsA.flag1==1){
                // This is the last pack of the current slice
                isFinished1 = 1;
                isFirst2 = 1;
                offset = flagsA.flag2;
            }

            LoopCopy0: for(unsigned char i=0; i<CONFIG_M_AXI_WIDTH; i++){
                if(i<flagsA.flag2){
                    cacheOutput[i] = cacheInput1[i];
                }
            }
        }
        
        if(isFinished1==1){
            if(isFirst2==0){
                LoopCopy1: for(unsigned char i=0; i<CONFIG_M_AXI_WIDTH; i++){ //Over cacheInput2
                    if( i>=(CONFIG_M_AXI_WIDTH-offset) ){
                        cacheOutput[i-(CONFIG_M_AXI_WIDTH-offset)] = cacheInput2[i];
                    }
                }
            }
            cacheInput2 = inStream2.Pop();
            FlagsWord_t flagsB = concat2_decode_flags(cacheInput2);
            if(flagsB.flag1==1){
                // This is the last pack of the current slice
                isFinished1 = 0;
            }

            LoopCopy2: for(unsigned char i=0; i<CONFIG_M_AXI_WIDTH; i++){
                if( i<flagsB.flag2 && (i+offset)<CONFIG_M_AXI_WIDTH ){ //Over cacheInput2
                    cacheOutput[i+offset] = cacheInput2[i];
                }
            }
            isFirst2=0;

        }

        outStream.Push(cacheOutput);
    }
}

void concat2_write_rank4_over3(
    Stream<MemoryPack_t, CONFIG_STREAMo_DEPTH> &inStream,
    MemoryPack_t *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int dimR3,
    unsigned int dimR3Padded){

    const unsigned int packCount = dim0*dim1*dim2*(dimR3Padded/CONFIG_M_AXI_WIDTH);
    LoopRead: for(unsigned int iter=0; iter<packCount; iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
        outputTn[iter] = inStream.Pop();
    }
}

void concat2_dataflow_rank4_over3(
    MemoryPack_t *inputTn1,
    MemoryPack_t *inputTn2,
    MemoryPack_t *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int dimA3, 
    unsigned int dimA3Padded, 
    unsigned int dimB3,
    unsigned int dimB3Padded,
    unsigned int dimR3,
    unsigned int dimR3Padded){
#pragma HLS DATAFLOW

    Stream<MemoryPackWithFlags_t, CONFIG_STREAM1_DEPTH> inStream1;
    Stream<MemoryPackWithFlags_t, CONFIG_STREAM2_DEPTH> inStream2;
    Stream<MemoryPack_t, CONFIG_STREAMo_DEPTH> outStream;

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(concat2_read1_rank4_over3,
        inputTn1, inStream1, dim0, dim1, dim2, dimA3, dimA3Padded);
    HLSLIB_DATAFLOW_FUNCTION(concat2_read2_rank4_over3,
        inputTn2, inStream2, dim0, dim1, dim2, dimB3, dimB3Padded);
    HLSLIB_DATAFLOW_FUNCTION(concat2_compute_rank4_over3,
        inStream1, inStream2, outStream, dim0, dim1, dim2, dimA3, dimA3Padded, dimB3, dimB3Padded, dimR3, dimR3Padded);
    HLSLIB_DATAFLOW_FUNCTION(concat2_write_rank4_over3,
        outStream, outputTn, dim0, dim1, dim2, dimR3, dimR3Padded);

    HLSLIB_DATAFLOW_FINALIZE();
}

extern "C" {
void task_concat(
    MemoryPack_t *inputTn1,
    MemoryPack_t *inputTn2,
    MemoryPack_t *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int dimA3,
    unsigned int dimA3Padded,
    unsigned int dimB3,
    unsigned int dimB3Padded,
    unsigned int dimR3,
    unsigned int dimR3Padded,
    int concatDim){

#pragma HLS INTERFACE m_axi     port=inputTn1       offset=slave    bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2       offset=slave    bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave    bundle=gmem3
#pragma HLS INTERFACE s_axilite port=inputTn1       bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2       bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control
#pragma HLS INTERFACE s_axilite port=dim0           bundle=control
#pragma HLS INTERFACE s_axilite port=dim1           bundle=control
#pragma HLS INTERFACE s_axilite port=dim2           bundle=control
#pragma HLS INTERFACE s_axilite port=dimA3          bundle=control
#pragma HLS INTERFACE s_axilite port=dimA3Padded    bundle=control
#pragma HLS INTERFACE s_axilite port=dimB3          bundle=control 
#pragma HLS INTERFACE s_axilite port=dimB3Padded    bundle=control 
#pragma HLS INTERFACE s_axilite port=dimR3          bundle=control 
#pragma HLS INTERFACE s_axilite port=dimR3Padded    bundle=control 
#pragma HLS INTERFACE s_axilite port=concatDim      bundle=control 
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    if(concatDim==3){
        concat2_dataflow_rank4_over3(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dimA3,
            dimA3Padded,
            dimB3,
            dimB3Padded,
            dimR3,
            dimR3Padded);
    }

}
}
