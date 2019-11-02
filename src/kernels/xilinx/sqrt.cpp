#include <hls_math.h>

#define CONFIG_CHUNK_SIZE 32

extern "C" {
void task_sqrt(
        const float* inputTn,
        float* outputTn,
        unsigned long len){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=len        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    Loop1:for(unsigned long i=0;i<len;i++){
#pragma HLS PIPELINE II=1
    	outputTn[i] = sqrt(inputTn[i]);
    }
}
}