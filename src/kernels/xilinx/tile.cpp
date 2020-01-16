#include "AxiHelper.h"
#include <cassert>
#include <stdio.h>

#define CONFIG_RANK3_AXIS1_MAX_SLICE_SIZE 1024

template<typename DType, int VecDepth>
void TileRank3Axis2(
    PackedArray<DType, VecDepth> *inputTn,
    PackedArray<DType, VecDepth> *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int tileCount){

    assert(dim2==1);

    unsigned long indxS,indxD;
    const unsigned long d0d1d2 = dim0*dim1*tileCount;
    int d0,d1,d2;

    unsigned long elementCacheVecIdx, lastElementCacheVecIdx;
    PackedArray<DType, VecDepth> elementCache;
    PackedArray<DType, VecDepth> buff;
#pragma HLS array_partition variable=elementCache complete dim=0
#pragma HLS array_partition variable=buff complete dim=0

    unsigned long outputVecIdx, outputSubVecIdx;

    d0=0; elementCacheVecIdx=-1;
    d1=0; lastElementCacheVecIdx=-1;
    d2=0;

    //printf("dim0:%u, dim1:%u, dim2:%u, tileCount:%u\n",dim0,dim1,dim2,tileCount);

    LoopMain: for(unsigned long iter=0;iter<d0d1d2;iter++){
        //printf("d0:%d, d1:%d, d2:%d\n",d0,d1,d2);
        //------------------------------------------------------
        indxS = (d0)*dim1 + d1;
        indxD = (d0)*dim1*tileCount + (d1)*tileCount + (d2);
        //printf("indxS:%d, indxD:%d\n",(int)indxS,(int)indxD);

        //------------------------------------------------------
        elementCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        //printf("elementCacheVecIdx:%d, elementCacheVecSubIdx:%d\n",(int)elementCacheVecIdx,(int)FlatIdx_to_VecSubIdx(VecDepth, indxS));
        if(elementCacheVecIdx!=lastElementCacheVecIdx){
            elementCache = inputTn[elementCacheVecIdx];
            //printf("****input1 read vId=%d\n", (int)elementCacheVecIdx);
        }
        lastElementCacheVecIdx = elementCacheVecIdx;

        //------------------------------------------------------
        outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
        outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
        //printf("outputVecIdx:%d, outputSubVecIdx:%d\n",(int)outputVecIdx,(int)outputSubVecIdx);
        buff.vec[outputSubVecIdx] = elementCache.vec[FlatIdx_to_VecSubIdx(VecDepth, indxS)];

        //------------------------------------------------------
        if(outputSubVecIdx==(VecDepth-1) || iter == (d0d1d2-1)){
            // Write output buff when it's ready.
            // Write output buff when it's not ready yet but this is the last iteration.
            outputTn[FlatIdx_to_VecIdx(VecDepth, indxD)] = buff;
            //printf("####output write vId=%d\n", (int)FlatIdx_to_VecIdx(VecDepth, indxD));
        }

        //=====================================================
        if( d2 == (tileCount-1) ){
            d2=0;
            if( d1 == (dim1-1) ){
                d0++;
                d1=0;
            }else{
                d1++;
            }

        }else{
            d2++;
        }
    }

}

template<typename DType, int VecDepth>
void TileRank3Axis1(
    PackedArray<DType, VecDepth> *inputTn,
    PackedArray<DType, VecDepth> *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int tileCount){

    assert(dim1==1);
    unsigned long indxS,indxD;
    const unsigned long d0d1d2 = dim0*tileCount*dim2;

    unsigned long sliceBuffIndex;
    float sliceBuff[CONFIG_RANK3_AXIS1_MAX_SLICE_SIZE];
    DO_PRAGMA(HLS array_partition variable=sliceBuff cyclic factor=VecDepth dim=0)


    unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
    unsigned long outputVecIdx, outputSubVecIdx;
    PackedArray<DType, VecDepth> inputCache;
    PackedArray<DType, VecDepth> outputCache;
#pragma HLS array_partition variable=inputCache complete dim=0
#pragma HLS array_partition variable=outputCache complete dim=0

    int d0,d1,d2;

    d0=0; 
    d1=0; 
    d2=0;
    inputCacheVecIdx=-1;
    lastInputCacheVecIdx=-1;
    sliceBuffIndex=0;

    //printf("dim0:%u, dim1:%u, dim2:%u, tileCount:%u\n",dim0,dim1,dim2,tileCount);

    LoopMain: for(unsigned long iter=0;iter<d0d1d2;iter++){
        //printf("\n\nd0:%d, d1:%d, d2:%d\n",d0,d1,d2);

        indxS = d0*dim2 + d2; //d0, d1, and d2 are for outputTn, not inputTn!
        indxD = d0*tileCount*dim2 + d1*dim2 + d2;
        
        //printf("indxS:%d, indxD:%d\n",(int)indxS,(int)indxD);


        inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        //printf("inputCacheVecIdx:%d\n",(int)inputCacheVecIdx);
        if(inputCacheVecIdx != lastInputCacheVecIdx && sliceBuffIndex<dim2){
            //printf("****input1 read vId=%d\n", (int)inputCacheVecIdx);
            inputCache = inputTn[inputCacheVecIdx];
            for(int i=0;i<VecDepth;i++){
                if(sliceBuffIndex<dim2){
                    //printf("==========>> sliceBuffIndex = %d\n", sliceBuffIndex);
                    sliceBuff[sliceBuffIndex] = inputCache.vec[i];
                    sliceBuffIndex++;
                }
            }
        }else{
            // Use current cached inputTn to fill the next dim2 slice into the slice buffer.
            // Pay attention to the boundaries of the for loop.
            if(sliceBuffIndex==0){
                inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS);
                //printf("$$$$ Reusing cached input from subVecIndex=%d\n", inputCacheVecSubIdx);
                for(int i=inputCacheVecSubIdx;i<VecDepth;i++){
                    if(sliceBuffIndex<dim2){ 
                        //printf("==========>> sliceBuffIndex = %d\n", sliceBuffIndex);
                        sliceBuff[sliceBuffIndex] = inputCache.vec[i];
                        sliceBuffIndex++;
                    }
                }
            }
        }
        lastInputCacheVecIdx = inputCacheVecIdx;

        outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
        outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
        //printf("outputVecIdx:%d, outputSubVecIdx:%d\n",(int)outputVecIdx,(int)outputSubVecIdx);
        outputCache.vec[outputSubVecIdx] = sliceBuff[d2];
        
        if(outputSubVecIdx==(VecDepth-1) || iter == (d0d1d2-1)){
            // Write output buff when it's ready.
            outputTn[outputVecIdx] = outputCache;
            //printf("####output write vId=%d\n", (int)outputVecIdx);
        }
        //=====================================================
        if( d2 == (dim2-1) ){
            d2=0;
            if( d1 == (tileCount-1) ){
                d0++;
                d1=0;
                sliceBuffIndex=0;
                //printf("==========>> *** slice buffer index is CLEARED.\n");
            }else{
                d1++;
            }

        }else{
            d2++;
        }
    }

}

extern "C" {
void task_tile(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        PackedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int dim3,
        unsigned int rank,
        unsigned int tileAxis,
        unsigned int tileCount){

#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control
#pragma HLS INTERFACE s_axilite port=dim3       bundle=control
#pragma HLS INTERFACE s_axilite port=rank       bundle=control
#pragma HLS INTERFACE s_axilite port=tileAxis   bundle=control
#pragma HLS INTERFACE s_axilite port=tileCount  bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=outputTn

    if(rank==3 && tileAxis==2) {
        TileRank3Axis2<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, dim0, dim1, dim2, tileCount);
    }

    
    if(rank==3 && tileAxis==1) {
        TileRank3Axis1<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, dim0, dim1, dim2, tileCount);
    }


}
}
