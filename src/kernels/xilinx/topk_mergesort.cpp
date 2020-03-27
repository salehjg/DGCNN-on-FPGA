#include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskTopK;
using hlslib::Stream;

// https://github.com/KastnerRG/pp4fpgas/blob/master/examples/merge_sort_loop_merged.cpp
void MergeSortWithIndices(
    CONFIG_DTYPE *inLocalBuff,
    unsigned *inLocalIndices){
#pragma HLS INLINE

    CONFIG_DTYPE temp[MaxSliceLen];
    unsigned indicesTemp[MaxSliceLen];

    LoopStage:
    for (int width = 1; width < MaxSliceLen; width = 2 * width) {
        int f1 = 0;
        int f2 = width;
        int i2 = width;
        int i3 = 2*width;
        if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
        if(i3 >= MaxSliceLen) i3 = MaxSliceLen;

        LoopMergeArrays:
        for (int i = 0; i < MaxSliceLen; i++) {
            #pragma HLS pipeline II=1

            CONFIG_DTYPE t1 = inLocalBuff[f1];
            CONFIG_DTYPE t2 = (f2 == i3) ? 0 : inLocalBuff[f2];

            if(f2 == i3 || (f1 < i2 && t1 <= t2)) {
                //if(f2 != i3)
                //{
                    indicesTemp[i] = inLocalIndices[f1];
                //}
                temp[i] = t1;
                f1++;
            } else {
                indicesTemp[i] = inLocalIndices[f2];
                assert(f2 < i3);
                temp[i] = t2;
                f2++;
            }
            if(f1 == i2 && f2 == i3) {
                f1 = i3;
                i2 += 2*width;
                i3 += 2*width;
                if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
                if(i3 >= MaxSliceLen) i3 = MaxSliceLen;
                f2 = i2;
            }
        }

        copy:
        for(int i = 0; i < MaxSliceLen; i++) {
            #pragma HLS pipeline II=1

            inLocalBuff[i] = temp[i];
            inLocalIndices[i] = indicesTemp[i];
        }

    }

    /*for(unsigned k=0; k<kVal; k++){
        std::cout<< "UDT.val"<<A[k]<<std::endl;
        std::cout<< "UDT.indices["<<k<<"]="<< indices[k]<<std::endl;
    }
    std::cout<< "==========================\n";*/
}



void UnitReadInput(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, PipeDepth> &streamInputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    unsigned indxS;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=UnitCount){
        LoopVecsPerPE:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            LoopPEs:
            for(unsigned iPE=0; iPE<UnitCount; iPE++){
                indxS = (batch+iPE)*vecsPerSlice + iVec;
                streamInputTn.Push(inputTn[indxS]);
            }
        }
    }

}

void UnitWriteOutput(
    MemoryPackI_t *indicesSplitedTn,
    Stream<MemoryPackI_t, PipeDepth> &streamIndices,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);

    unsigned indxD;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=UnitCount){
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                indxD = (batch+iPE)*vecsPerOutputSlice + iVec;
                indicesSplitedTn[indxD] = streamIndices.Pop();
            }
        }
    }
}

//latency reported for [5x1024]x1024, k=20, unitcount=8, m_axi_width=16, pipe_depth=2
void UnitProcessingElement(
    Stream<MemoryPackF_t, PipeDepth> &streamDataIn,
    Stream<MemoryPackF_t, PipeDepth> &streamDataOut,
    Stream<MemoryPackI_t, PipeDepth> &streamIndicesIn,
    Stream<MemoryPackI_t, PipeDepth> &streamIndicesOut,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice,
    const unsigned vecsPerOutputSlice,
    const unsigned kValue,
    const unsigned unitIndex){
    
    // Only divisible 'dim1' by maxi width is supported so far.
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    // Length of the PE's local buffers('MaxSliceLen') should be greater or equal to 'dim1'.
    assert(dim1<=MaxSliceLen);

    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);

    unsigned min_idx;
    unsigned indxS,indxD;

    CONFIG_DTYPE sliceData[MaxSliceLen];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceData cyclic factor=CONFIG_M_AXI_WIDTH dim=1)
    //DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceData complete dim=1)

    unsigned sliceIndices[MaxSliceLen];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceIndices cyclic factor=CONFIG_M_AXI_WIDTH dim=1)
    //DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceIndices complete dim=1)

    MemoryPackF_t sliceSubVec;
    MemoryPackI_t outputCache;
    unsigned outputCacheVecSubIdx;

    LoopMain: for(unsigned batch=0; batch<dim0; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=640 max=640
        ///TODO: Check for out of bound batch index 
        //--------------------------------------------------
        // 1. Read current slice and indices into local memory.
        LoopReadSlice: for(unsigned idx=0; idx<vecsPerSlice; idx++){
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            
            LoopInputPass01:
            for(unsigned iPE=0; iPE<UnitCount-unitIndex; iPE++){
                #pragma HLS PIPELINE II=1
                MemoryPackF_t vec = streamDataIn.Pop();
                if(iPE>0){
                    // Pass the data to other PEs and just keep the last one for this PE
                    if(unitIndex<(UnitCount-1)){
                        streamDataOut.Push(vec);
                    }
                }else{
                    sliceSubVec = vec;
                }
            }
            
            const unsigned offsetLocal = idx*CONFIG_M_AXI_WIDTH;
            LoopReadUnroll1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                const unsigned indxLocal = offsetLocal+i; 
                sliceData[indxLocal] = sliceSubVec[i];
                sliceIndices[indxLocal] = indxLocal;
            }
        }

        //--------------------------------------------------
        // 2. Run sorting algorithm on the local memory.
        MergeSortWithIndices(sliceData, sliceIndices);

        LoopPushTheResults:
        for(unsigned i=0; i<kValue; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=20 max=20
            outputCacheVecSubIdx = i % CONFIG_M_AXI_WIDTH;
            outputCache[outputCacheVecSubIdx] = sliceIndices[i];
            if (outputCacheVecSubIdx == (CONFIG_M_AXI_WIDTH - 1) || i == (kValue - 1)) {
                streamIndicesOut.Push(outputCache);
#ifdef KERNEL_LOGS
                cout << "PE" << unitIndex << ": " << " Sorted Vec i=" << i << endl;
#endif
            }
        }
        
        //--------------------------------------------------
        
        // 3. Handle incoming data of streamIndicesIn from other PEs.
        const unsigned _len2 = UnitCount-unitIndex-1;
        LoopHandleOtherPEsOutput:
        for(unsigned iPE=0; iPE<_len2; iPE++){
            #pragma HLS LOOP_TRIPCOUNT min=8 max=8
            ForOutputVecsPerPEs:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                if(unitIndex<(UnitCount-1)){
                    streamIndicesOut.Push(streamIndicesIn.Pop());
                }
#ifdef KERNEL_LOGS
                cout<<"*PE"<<unitIndex<<": "<<"Handling Other PE Results, "<<"Pop'ed streamIndicesIn"<<endl;
#endif
            }
        }
    }
#ifdef KERNEL_LOGS
    cout<<"==PE"<<unitIndex<<": "<<"FINISHED"<<endl;
#endif
}

extern "C"{
void task_topk(
        const MemoryPackF_t *inputTn,
        MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=indicesSplitedTn offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=kValue bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerSlice bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerOutputSlice bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    Stream<MemoryPackF_t, PipeDepth> streamsData[UnitCount+1];
#pragma HLS STREAM variable=streamsData depth=PipeDepth

    Stream<MemoryPackI_t, PipeDepth> streamsIndices[UnitCount+1];
#pragma HLS STREAM variable=streamsIndices depth=PipeDepth

#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < UnitCount+1; i++) {
        streamsData[i].set_name(("streamsData[" + std::to_string(i) + "]").c_str());
    }
    for (unsigned n = 0; n < UnitCount+1; n++) {
        streamsIndices[n].set_name(("streamsIndices[" + std::to_string(n) + "]").c_str());
    }
#endif
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(UnitReadInput, inputTn, streamsData[0], dim0, dim1, vecsPerSlice);
    
    for (unsigned iPE = 0; iPE < UnitCount; iPE++) {
#pragma HLS UNROLL
        HLSLIB_DATAFLOW_FUNCTION(UnitProcessingElement,
            streamsData[iPE],
            streamsData[iPE+1],
            streamsIndices[iPE+1],
            streamsIndices[iPE],
            dim0,
            dim1,
            vecsPerSlice,
            vecsPerOutputSlice,
            kValue,
            iPE);
    }
    
    HLSLIB_DATAFLOW_FUNCTION(UnitWriteOutput, indicesSplitedTn, streamsIndices[0], dim0, dim1, vecsPerOutputSlice);

    HLSLIB_DATAFLOW_FINALIZE();
    

}
}
