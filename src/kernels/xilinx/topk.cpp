#include <cassert>
#include <hls_stream.h>

#define CONFIG_N 1024
#define CONFIG_K 20

/*
//Naive version
template<typename DTypeData, typename DTypeIndices>
void BatchSelectionSortTopK_try01(
    const DTypeData* inputTn,
    DTypeIndices* indicesSplitedTn,
    int dim0,
    int dim1,
    int dim2,
    int kValue){

    int i, j, min_idx;
    unsigned long indxS,indxD;
    assert(kValue<dim2); 

    DTypeData sliceData[CONFIG_N];
    DTypeIndices sliceIndices[CONFIG_N];


    For_Main:for(int batch=0; batch<dim0*dim1; batch++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120

        //--------------------------------------------------
        // 1. Read current slice and indices into local memory.
        For_Read: for(int idx=0; idx<dim2; idx++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            indxS = batch*dim2 + idx;
            sliceIndices[idx] = idx;
            sliceData[idx] = inputTn[indxS];
        }

        //--------------------------------------------------
        // 2. Run sorting algorithm on the local memory.
        For_Sort1: for (i = 0; i < kValue; i++){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS UNROLL factor=1
            // Find the maximum element in unsorted array
            min_idx = i;
            For_Sort2: for (j = i+1; j < dim2; j++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1004 max=1024
                if (sliceData[j] < sliceData[min_idx])
                    min_idx = j;
            }

            // Swap the found maximum element with the first element
            if(min_idx != i){
                float tmp = sliceData[min_idx];
                sliceData[min_idx] = sliceData[i];
                sliceData[i] = tmp;
                //-----------------------------
                int tmpi = sliceIndices[min_idx];
                sliceIndices[min_idx] = sliceIndices[i];
                sliceIndices[i] = tmpi;
            }
        }

        //--------------------------------------------------
        // 3. Write back the results of the current slice into the global memory
        For_Write: for(i = 0; i < kValue; i++){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS PIPELINE II=1
            indxD = batch*kValue + i;

            indicesSplitedTn[indxD] = sliceIndices[i];
        }

        //--------------------------------------------------
    }
}

//Fused loop version
template<typename DTypeData, typename DTypeIndices>
void BatchSelectionSortTopK_try02(
    const DTypeData* inputTn,
    DTypeIndices* indicesSplitedTn,
    int dim0,
    int dim1,
    int dim2,
    int kValue){

    int min_idx;
    unsigned long indxS,indxD;

    DTypeData sliceData[CONFIG_N];
    DTypeIndices sliceIndices[CONFIG_N];


    For_Main:for(int batch=0; batch<dim0*dim1; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        //--------------------------------------------------
        // 1. Read current slice and indices into local memory.
        For_Read: for(int idx=0; idx<dim2; idx++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            indxS = batch*dim2 + idx;
            sliceIndices[idx] = idx;
            sliceData[idx] = inputTn[indxS];
        }

        //--------------------------------------------------
        // 2. Run sorting algorithm on the local memory.

        int i,j;

        const int _len = CONFIG_K * ( 2*(CONFIG_N-1)*(CONFIG_N-CONFIG_K) );
        i = 0;
        j = 1;
        min_idx = 0;
        for(int iter=0; iter<_len;iter++){

            if (sliceData[j] < sliceData[min_idx]){
                min_idx = j;
            }

            //------------------------------
            //Fused loop's house keeping stuff
            if(j==CONFIG_N-1){
                if(min_idx != i){
                    float tmp = sliceData[min_idx];
                    sliceData[min_idx] = sliceData[i];
                    sliceData[i] = tmp;
                    //-----------------------------
                    int tmpi = sliceIndices[min_idx];
                    sliceIndices[min_idx] = sliceIndices[i];
                    sliceIndices[i] = tmpi;
                }
                //--------------------------
                i++;
                j++;
                //--------------------------
                min_idx = i;
            }
        }

        //--------------------------------------------------
        // 3. Write back the results of the current slice into the global memory
        For_Write: for(i = 0; i < kValue; i++){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS PIPELINE II=1
            indxD = batch*kValue + i;

            indicesSplitedTn[indxD] = sliceIndices[i];
        }

        //--------------------------------------------------
    }
}


//Dataflow version
template<typename DTypeData, typename DTypeIndices>
void BatchSelectionSortTopK_try03(
    const DTypeData* inputTn,
    DTypeIndices* indicesSplitedTn,
    int dim0,
    int dim1,
    int dim2,
    int kValue){
#pragma HLS DATAFLOW

    int min_idx;
    unsigned long indxS,indxD;

    DTypeData sliceData[CONFIG_N];
    DTypeIndices sliceIndices[CONFIG_N];

    hls::stream<DTypeData> streamData;
    hls::stream<DTypeIndices> streamIndices;
#pragma HLS STREAM variable=streamData depth=2048 dim=1
#pragma HLS STREAM variable=streamIndices depth=40 dim=1


    For_Read1: for(int batch=0; batch<dim0*dim1; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        For_Read: for(int idx=0; idx<dim2; idx++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            indxS = batch*dim2 + idx;


            streamData << inputTn[indxS];
        }
    }



    For_Sort0:for(int batch=0; batch<dim0*dim1; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        For_Sort_StreamRead: for(int k=0;k<CONFIG_N;k++){
            sliceIndices[k] = k;
            sliceData[k] = streamData.read();
        }

        For_Sort1: for (int i = 0; i < kValue; i++){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS UNROLL factor=1
            // Find the maximum element in unsorted array
            min_idx = i;
            For_Sort2: for (int j = i+1; j < dim2; j++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1004 max=1024
                if (sliceData[j] < sliceData[min_idx])
                    min_idx = j;
            }

            // Swap the found maximum element with the first element
            if(min_idx != i){
                float tmp = sliceData[min_idx];
                sliceData[min_idx] = sliceData[i];
                sliceData[i] = tmp;
                //-----------------------------
                int tmpi = sliceIndices[min_idx];
                sliceIndices[min_idx] = sliceIndices[i];
                sliceIndices[i] = tmpi;
            }
        }


        For_Sort_StreamWrite: for(int k=0;k<CONFIG_K;k++){
            streamIndices<<sliceIndices[k];
        }

    }



    For_Write1:for(int batch=0; batch<dim0*dim1; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        For_Write2: for(int i = 0; i < kValue; i++){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20
#pragma HLS PIPELINE II=1
            indxD = batch*kValue + i;

            indicesSplitedTn[indxD] = streamIndices.read();
        }
    }

}
*/



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

template<typename DTypeData, typename DTypeIndices, int UnitCount>
static void ReadUnit(
        const DTypeData* inputTn,
        hls::stream<DTypeData> (&streamIn)[UnitCount], //https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
        int dim0,
        int dim1,
        int dim2){

    const int _len = dim0 * dim1;
    const int _tripMain = dim0 * dim1 / UnitCount;

    int batchIndex;

    For_Main: for(int batch=0; batch<_len; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=_tripMain max=_tripMain
        For_Units: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL
            For_BurstRead: for(int d2=0; d2<dim2; d2++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                batchIndex = batch+uc;
                if(batchIndex<_len){
                    streamIn[uc]<< inputTn[batchIndex*dim2+d2];
                }
            }
        }
    }
}


template<typename DTypeData, typename DTypeIndices, int UnitCount>
static void WriteUnit(
        DTypeIndices* indicesSplitedTn,
        hls::stream<DTypeIndices> (&streamOut)[UnitCount], //https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
        int dim0,
        int dim1,
        int dim2,
        int kValue){

    const int _len = dim0 * dim1;
    const int _tripMain = dim0 * dim1 / UnitCount;

    int batchIndex;

    For_Main: for(int batch=0; batch<_len; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=_tripMain max=_tripMain
        For_Units: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL
            For_BurstRead: for(int d2=0; d2<kValue; d2++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                batchIndex = batch+uc;
                if(batchIndex<_len){
                    indicesSplitedTn[batchIndex * kValue + d2] = streamOut[uc].read();
                }
            }
        }
    }
}

//Fused loop version, parallel batch processing
template<typename DTypeData, typename DTypeIndices, int UnitCount>
void BatchSelectionSortTopK_try04(
        const DTypeData* inputTn,
        DTypeIndices* indicesSplitedTn,
        int dim0,
        int dim1,
        int dim2,
        int kValue){

    hls::stream<DTypeData>   streamIn[UnitCount];
    hls::stream<DTypeIndices> streamOut[UnitCount];
#pragma HLS STREAM variable=streamIn depth=10
#pragma HLS STREAM variable=streamOut depth=6

#pragma HLS dataflow

    ReadUnit<DTypeData,DTypeIndices,UnitCount>(inputTn, streamIn, dim0, dim1, dim2);

    For_Compute: for(int uc=0; uc<UnitCount; uc++){
#pragma HLS UNROLL

        SortingUnit<DTypeData,DTypeIndices,UnitCount>(streamIn[uc], streamOut[uc], dim0, dim1, dim2, kValue, uc);
    }

    WriteUnit<DTypeData,DTypeIndices,UnitCount>(indicesSplitedTn, streamOut, dim0, dim1, dim2, kValue);

}

extern "C"{
void task_topk(
        const float* inputTn,
        int* indicesSplitedTn,
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

    BatchSelectionSortTopK_try04<float, int, 4>(inputTn, indicesSplitedTn, dim0, dim1, dim2, kValue);

}
}
