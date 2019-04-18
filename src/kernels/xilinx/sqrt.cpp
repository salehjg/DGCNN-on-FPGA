#include <stdio.h>
#include <string.h>
#include <hls_math.h>

#define CONFIG_CHUNK_SIZE 32

extern "C" {
void task_sqrt(
        float* inputTn,
        float* outputTn,
        unsigned long len){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=len        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

	float buff[CONFIG_CHUNK_SIZE];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=1

    Loop1:for(unsigned long i=0;i<len;i+=CONFIG_CHUNK_SIZE){
#pragma HLS PIPELINE off
    	int safe_chunk = CONFIG_CHUNK_SIZE;
    	if(i + CONFIG_CHUNK_SIZE > len){
    		safe_chunk = len - i;
    	}
    	memcpy(buff,&inputTn[i], safe_chunk*sizeof(float));

        Loop2:for(int j=0; j<CONFIG_CHUNK_SIZE; j++){
#pragma HLS UNROLL
        	if(j<safe_chunk){
        		buff[j] = sqrt(buff[j]);
        	}
        }

        memcpy(&outputTn[i], buff, safe_chunk*sizeof(float));
    }

}
}
