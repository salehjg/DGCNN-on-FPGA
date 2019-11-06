/*
_task_datamover:
Copies data of length 'len' :
    reverseSwitch=0 --> from srcBuff to dstBuff, so, from bankA to bankB
    reverseSwitch=1 --> from dstBuff to srcBuff, so, from bankB to bankA

/!\ srcBuff should ALWAYS be on bankA.
/!\ dstBuff should ALWAYS be on bankB.
*/

template<typename DType>
static void _task_datamover(
        DType *srcBuff,
        DType *dstBuff,
        int reverseSwitch,
        const unsigned long len){
#pragma HLS inline

    if(reverseSwitch==0){
        for(unsigned long i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            dstBuff[i] = srcBuff[i];
        }
    }else{
        for(unsigned long i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            srcBuff[i] = dstBuff[i];
        }
    }


}

extern "C" {
/*
void task_datamover_b0_to_b1_integer(
        int *srcBuff,
        int *dstBuff,
        int reverseSwitch,
        const unsigned long len){
#pragma HLS INTERFACE m_axi     port=srcBuff            offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=dstBuff            offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=srcBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len                bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    _task_datamover_b0_to_b1<int>(srcBuff, dstBuff, reverseSwitch, len);
}
*/

// The --sp option decides which bank is bankA and which is bankB
// Currently, bankA is bank1 & bankB is bank2 and no SLR assignment is done.
void task_datamover_mod1_int(
        int *srcBuff,
        int *dstBuff,
        int reverseSwitch,
        const unsigned long len){
#pragma HLS INTERFACE m_axi     port=srcBuff            offset=slave bundle=gmemi0
#pragma HLS INTERFACE m_axi     port=dstBuff            offset=slave bundle=gmemi1
#pragma HLS INTERFACE s_axilite port=srcBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len                bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    _task_datamover<int>(srcBuff, dstBuff, reverseSwitch, len);
}

}
