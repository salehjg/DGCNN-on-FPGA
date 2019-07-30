/*
_task_datamover_b0_to_b1:
Copies data of length 'len' :
	reverseSwitch=0 --> from srcBuff to dstBuff, so, from bank0 to bank1
	reverseSwitch=1 --> from dstBuff to srcBuff, so, from bank1 to bank0

/!\ srcBuff should ALWAYS be on bank0.
/!\ dstBuff should ALWAYS be on bank1.
*/

template<typename DType>
static void _task_datamover_b0_to_b1(
		DType *srcBuff,
		DType *dstBuff,
		int reverseSwitch,
		const unsigned long len){


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
#pragma HLS INTERFACE m_axi     port=srcBuff    		offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=dstBuff   			offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=srcBuff    		bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff   			bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len        		bundle=control
#pragma HLS INTERFACE s_axilite port=return     		bundle=control

	_task_datamover_b0_to_b1<int>(srcBuff, dstBuff, reverseSwitch, len);
}
*/
void task_datamover_b0_to_b1_float(
		float *srcBuff,
		float *dstBuff,
		int reverseSwitch,
		const unsigned long len){
#pragma HLS INTERFACE m_axi     port=srcBuff    		offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=dstBuff   			offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=srcBuff    		bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff   			bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len        		bundle=control
#pragma HLS INTERFACE s_axilite port=return     		bundle=control

	_task_datamover_b0_to_b1<float>(srcBuff, dstBuff, reverseSwitch, len);
}

}
