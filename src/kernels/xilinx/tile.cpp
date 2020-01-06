#include <cassert>
#include "VectorizationHelper.h"
#include <stdio.h>

template<typename DType, int VecDepth>
void TileRank3Axis2(
    VectorizedArray<DType, VecDepth> *inputTn,
    VectorizedArray<DType, VecDepth> *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int tileCount){

    assert(dim2==1);

    unsigned long indxS,indxD;
    const unsigned long d0d1d2 = dim0*dim1*tileCount;
    int d0,d1,d2;

    unsigned long elementCacheVecIdx, lastElementCacheVecIdx;
    VectorizedArray<DType, VecDepth> elementCache;
    VectorizedArray<DType, VecDepth> buff;
#pragma HLS array_partition variable=elementCache complete dim=0
#pragma HLS array_partition variable=buff complete dim=0

    unsigned long outputVecIdx, outputSubVecIdx;

    d0=0; elementCacheVecIdx=-1;
    d1=0; lastElementCacheVecIdx=-1;
    d2=0;

    printf("dim0:%u, dim1:%u, dim2:%u, tileCount:%u\n",dim0,dim1,dim2,tileCount);

    for(unsigned long iter=0;iter<d0d1d2;iter++){
        printf("d0:%d, d1:%d, d2:%d\n",d0,d1,d2);
        //------------------------------------------------------
        indxS = (d0)*dim1 + d1;
        indxD = (d0)*dim1*tileCount + (d1)*tileCount + (d2);
        printf("indxS:%d, indxD:%d\n",(int)indxS,(int)indxD);

        //------------------------------------------------------
        elementCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
        printf("elementCacheVecIdx:%d, elementCacheVecSubIdx:%d\n",(int)elementCacheVecIdx,(int)FlatIdx_to_VecSubIdx(VecDepth, indxS));
        if(elementCacheVecIdx!=lastElementCacheVecIdx){
            elementCache = inputTn[elementCacheVecIdx];
            printf("****input1 read vId=%d\n", (int)elementCacheVecIdx);
        }
        lastElementCacheVecIdx = elementCacheVecIdx;

        //------------------------------------------------------
        outputVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
        outputSubVecIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
        printf("outputVecIdx:%d, outputSubVecIdx:%d\n",(int)outputVecIdx,(int)outputSubVecIdx);
        buff.vec[outputSubVecIdx] = elementCache.vec[FlatIdx_to_VecSubIdx(VecDepth, indxS)];

        //------------------------------------------------------
        if(outputSubVecIdx==(VecDepth-1)){
            // Write output buff when it's ready.
            outputTn[FlatIdx_to_VecIdx(VecDepth, indxD)] = buff;
            printf("####output write vId=%d\n", (int)FlatIdx_to_VecIdx(VecDepth, indxD));
        }

        //=====================================================
        if( d2 == ((unsigned long)tileCount-1) ){
            d2=0;
            if( d1 == ((unsigned long)dim1-1) ){
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


/*
void TileRank3Axis1(
    const float *inputTn,
    float *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int tileCount){
#pragma HLS INLINE

    assert(dim1==1);
    float buff[CONFIG_SLICE_SIZE];
    unsigned long indxS,indxD;

    for(int d0=0; d0<dim0; d0++){

        LoopBurstReadSlice: for(int d2=0; d2<dim2; d2++){
#pragma HLS PIPELINE II=1
            indxS = d0*dim2 + d2;
            buff[d2] = inputTn[indxS];
        }
        //-------------------------------------------

        LoopTile: for(int k=0; k<tileCount; k++){
            LoopBurstWriteSlice: for(int d2=0; d2<dim2; d2++){
#pragma HLS PIPELINE II=1
                indxD = (d0)*tileCount*dim2 + (k)*dim2 + (d2);
                outputTn[indxD] = buff[d2];
            }
        }
        //-------------------------------------------

    }


}
*/

/*
void TileRank4Axis2(
    const float *inputTn,
    float *outputTn,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int dim3,
    unsigned int tileCount){
#pragma HLS INLINE

    assert(dim2==1);
    float buff[CONFIG_SLICE_SIZE];
    unsigned long indxS,indxD;
    const unsigned long d0d1 = dim0*dim1;

    int d0,d1;
    d0=0;
    d1=0;
    //Fused loop for dim0 dim1 and dim3 (dim2 equals one).
    for(unsigned long iter=0;iter<d0d1;iter++){

        LoopBurstReadSlice: for(int d3=0; d3<dim3; d3++){
#pragma HLS PIPELINE II=1
            indxS = (d0)*dim1*dim3 + (d1)*dim3 + d3;
            buff[d3] = inputTn[indxS];
        }
        //-------------------------------------------

        LoopTile: for(int k=0; k<tileCount; k++){
            LoopBurstWriteSlice: for(int d3=0; d3<dim3; d3++){
#pragma HLS PIPELINE II=1
                indxD = (d0)*dim1*tileCount*dim3 + (d1)*tileCount*dim3 + (k)*dim3 + d3;
                outputTn[indxD] = buff[d3];
            }
        }

        //=====================================
        if(iter==dim1-1){
            d1=0;
            d0++;
        }else{
            d1++;
        }
    }
}
*/


extern "C" {
void task_tile(
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
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


    /*
    if(rank==4 && tileAxis==2) {
        TileRank4Axis2(inputTn,outputTn,dim0,dim1,dim2,dim3,tileCount);
    }*/

    if(rank==3 && tileAxis==2) {
        TileRank3Axis2<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, dim0, dim1, dim2, tileCount);
    }

    /*
    if(rank==3 && tileAxis==1) {
        TileRank3Axis1(inputTn,outputTn,dim0,dim1,dim2,tileCount);
    }*/


}
}
