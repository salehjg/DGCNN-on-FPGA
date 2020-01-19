#include "AxiHelper.h"
#include "xilinx/config.h"
#include <cassert>
#include <hls_stream.h>
#include <stdio.h>

#define CONFIG_N 1024
#define CONFIG_K 20

// Inline Sub-function
template<typename DTypeData, typename DTypeIndices, int UnitCount>
static void SortingUnit(
        hls::stream<DTypeData> &streamIn,
        hls::stream<DTypeIndices> &streamOut,
        int dim0,
        int dim1,
        int dim2,
        int kValue,
        int unitIndex){
//#pragma HLS INLINE

    int min_idx;
    unsigned long indxS,indxD;
    DTypeData sliceData[CONFIG_N];
    DTypeIndices sliceIndices[CONFIG_N];

    const int _len = dim0*dim1;
    const int _tripMain = dim0 * dim1 / UnitCount;
    For_Main: for(int batch=0; batch<_len; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=_tripMain max=_tripMain
        if(unitIndex+batch<_len){
            //--------------------------------------------------
            // 1. Read current slice and indices into local memory.
            For_Read: for(int idx=0; idx<dim2; idx++){
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                sliceData[idx] = streamIn.read();
                sliceIndices[idx] = idx;
            }

            //--------------------------------------------------
            // 2. Run sorting algorithm on the local memory.

            int i,j;

            const int _len = ((CONFIG_K)*( (CONFIG_N-1) + (CONFIG_N-CONFIG_K) ))/2;

            i = 0;
            j = 1;
            min_idx = 0;
            FusedLoopSort:for(int iter=0; iter<_len;iter++){

                if (sliceData[j] < sliceData[min_idx]){
                    min_idx = j;
                }

                //------------------------------
                //Fused loop's house keeping stuff
                if(j==CONFIG_N-1){
                    //if(min_idx != i)
                    {
                        //Commented lines are for avoid unnecessary memory accesses.
                        //They don't affect the REQUIRED output of this compute unit.

                        //float tmp = sliceData[min_idx];
                        sliceData[min_idx] = sliceData[i];
                        //sliceData[i] = tmp;
                        //--------------------------------
                        int tmp2 = sliceIndices[min_idx];
                        sliceIndices[min_idx] = sliceIndices[i];
                        sliceIndices[i] = tmp2;
                        streamOut<<tmp2;
                    }
                    //--------------------------
                    i++;
                    j=i+1;
                    //--------------------------
                    min_idx = i;
                }else{
                    j++;
                }
            }

            //--------------------------------------------------
            /*
            // 3. Write back the results of the current slice into the global memory
            ///TODO: It might be faster to directly write results to global memory inside FusedLoopSort.
            For_Write: for(i = 0; i < kValue; i++){
        #pragma HLS LOOP_TRIPCOUNT min=20 max=20
        #pragma HLS PIPELINE II=1
                indxD = batchIndex*kValue + i;

                indicesSplitedTn[indxD] = sliceIndices[i];
            }
            */
        }
    }
}

template<typename DTypeData, int UnitCount, int VecDepth>
static void ReadUnit(
        PackedArray<DTypeData, VecDepth> *inputTn,
        hls::stream<DTypeData> (&streamIn)[UnitCount], //https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
        int dim0,
        int dim1,
        int dim2){

    const int _len = dim0 * dim1;
    const int _tripMain = dim0 * dim1 / UnitCount;

    For_Main: for(int batch=0; batch<_len; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=_tripMain max=_tripMain
        For_Units: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL
            //-----------------------------------------------------------
            unsigned long indxS;
            int batchIndex;
            unsigned long inputCacheVecIdx, inputCacheVecSubIdx, lastInputCacheVecIdx;
            PackedArray<DTypeData, VecDepth> inputCache;
#pragma HLS array_partition variable=inputCache complete dim=0
            lastInputCacheVecIdx=-1;
            // Considering that dim2 slices are 1024 words long and VecDepth is 16 words long,
            // there won't be any wasted words in vectorized memory access.

            //-----------------------------------------------------------
            For_BurstRead: for(int d2=0; d2<dim2; d2++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

                batchIndex = batch+uc;
                indxS = batchIndex*dim2+d2;

                inputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxS);
                inputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxS);
                if(inputCacheVecIdx!=lastInputCacheVecIdx){
                    inputCache = inputTn[inputCacheVecIdx];
                }
                lastInputCacheVecIdx = inputCacheVecIdx;

                if(batchIndex<_len){
                    streamIn[uc] << inputCache.vec[inputCacheVecSubIdx];
                }
            }

            //-----------------------------------------------------------
        }
    }
}


template<typename DTypeIndices, int UnitCount, int VecDepth>
static void WriteUnit(
        PackedArray<DTypeIndices, VecDepth> *indicesSplitedTn,
        hls::stream<DTypeIndices> (&streamOut)[UnitCount], //https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
        int dim0,
        int dim1,
        int dim2,
        int kValue){

    const int _len = dim0 * dim1;
    const int _tripMain = dim0 * dim1 / UnitCount;

    For_Main: for(int batch=0; batch<_len; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=_tripMain max=_tripMain
        For_Units: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL
            //-----------------------------------------------------------
            int batchIndex; 
            unsigned long indxD;
            unsigned long outputCacheVecIdx, outputCacheVecSubIdx;
            PackedArray<DTypeIndices, VecDepth> outputCache;
#pragma HLS array_partition variable=outputCache complete dim=0

            //-----------------------------------------------------------
            For_BurstRead: for(int d2=0; d2<kValue; d2++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                batchIndex = batch+uc;
                indxD = batchIndex * kValue + d2;
                outputCacheVecIdx = FlatIdx_to_VecIdx(VecDepth, indxD);
                outputCacheVecSubIdx = FlatIdx_to_VecSubIdx(VecDepth, indxD);
                if(batchIndex<_len){
                    outputCache.vec[outputCacheVecSubIdx] = streamOut[uc].read();
                }

                // To avoid loss of data by half-full vectors, kValue should be devidable to VecDepth.
                if(outputCacheVecSubIdx == (VecDepth-1) ){
                    indicesSplitedTn[outputCacheVecIdx] = outputCache;
                }
            }

            //-----------------------------------------------------------
        }
    }
}

// Fused loop version, parallel batch processing
// Try04
template<typename DTypeData, typename DTypeIndices, int UnitCount, int VecDepthIn, int VecDepthOut>
void BatchSelectionSortTopK(
        PackedArray<DTypeData, VecDepthIn> *inputTn,
        PackedArray<DTypeIndices, VecDepthOut> *indicesSplitedTn,
        int dim0,
        int dim1,
        int dim2,
        int kValue){

    hls::stream<DTypeData>   streamIn[UnitCount];
    hls::stream<DTypeIndices> streamOut[UnitCount];
#pragma HLS STREAM variable=streamIn depth=10
#pragma HLS STREAM variable=streamOut depth=6

#pragma HLS dataflow

    ReadUnit<DTypeData,UnitCount,VecDepthIn>(inputTn, streamIn, dim0, dim1, dim2);

    For_Compute: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL

        SortingUnit<DTypeData,DTypeIndices,UnitCount>(streamIn[uc], streamOut[uc], dim0, dim1, dim2, kValue, uc);
    }

    WriteUnit<DTypeIndices,UnitCount,VecDepthOut>(indicesSplitedTn, streamOut, dim0, dim1, dim2, kValue);

}

extern "C"{
void task_topk(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        PackedArray<int, CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH> *indicesSplitedTn,
        int dim0,
        int dim1,
        int dim2,
        int kValue){

#pragma HLS INTERFACE m_axi     port=inputTn                offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesSplitedTn       offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=inputTn            bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control

#pragma HLS INTERFACE s_axilite port=kValue     bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=indicesSplitedTn

    BatchSelectionSortTopK<float, int, 4, CONFIG_M_AXI_WIDTH, CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH>(
        inputTn, 
        indicesSplitedTn, 
        dim0, 
        dim1, 
        dim2, 
        kValue);

}
}
