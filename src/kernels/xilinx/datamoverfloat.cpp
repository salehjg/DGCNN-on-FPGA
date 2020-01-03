#include "VectorizationHelper.h"
#include "stdio.h"
/*
_task_datamover:
Copies data of length 'len' :
    reverseSwitch=0 --> from srcBuff to dstBuff, so, from bankA to bankB
    reverseSwitch=1 --> from dstBuff to srcBuff, so, from bankB to bankA

/!\ srcBuff should ALWAYS be on bankA.
/!\ dstBuff should ALWAYS be on bankB.
/!\ len is number of (VecDepth*sizeof(DType)-bytes)words 
*/

template<typename DType, int VecDepth>
static void _task_datamover(
        VectorizedArray<DType, VecDepth> *srcBuff,
        VectorizedArray<DType, VecDepth> *dstBuff,
        int reverseSwitch,
        const unsigned long len){
#pragma HLS inline

    VectorizedArray<DType, VecDepth> buff;
    

    if(reverseSwitch==0){
        for(unsigned long i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            buff = srcBuff[i];
            dstBuff[i] = buff;
        }
    }else{
        for(unsigned long i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            buff = dstBuff[i];
            srcBuff[i] = buff;
        }
    }


}

extern "C" {

// The --sp option decides which bank is bankA and which is bankB
// Currently, bankA is bank1 & bankB is bank2 and no SLR assignment is done.
void task_datamover_mod1_float(
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *srcBuff,
        VectorizedArray<float, CONFIG_M_AXI_WIDTH> *dstBuff,
        int reverseSwitch,
        const unsigned long len){
#pragma HLS INTERFACE m_axi     port=srcBuff            offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=dstBuff            offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=srcBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len                bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control
#pragma HLS data_pack variable=srcBuff
#pragma HLS data_pack variable=dstBuff
    _task_datamover<float, CONFIG_M_AXI_WIDTH>(srcBuff, dstBuff, reverseSwitch, len);
}

}
