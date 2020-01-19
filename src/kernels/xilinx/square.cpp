#include "AxiHelper.h"
#include "xilinx/config.h"

template<typename DType, int VecDepth>
void _task_square(
        PackedArray<DType, VecDepth> *inputTn,
        PackedArray<DType, VecDepth> *outputTn,
        unsigned long len){

	PackedArray<DType, VecDepth> buff;
#pragma HLS array_partition variable=buff complete dim=0

    for(unsigned long i=0;i<len;i++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
#pragma HLS PIPELINE II=1
    	buff = inputTn[i];
    	for(int j=0; j<VecDepth; j++){
#pragma HLS UNROLL
    		buff.vec[j] = buff.vec[j] * buff.vec[j];
    	}
        outputTn[i] = buff;
    }

}

extern "C" {
void task_square(
        PackedArray<float, CONFIG_M_AXI_WIDTH> *inputTn,
        PackedArray<float, CONFIG_M_AXI_WIDTH> *outputTn,
        const unsigned long len){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=len        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS data_pack variable=inputTn
#pragma HLS data_pack variable=outputTn

    _task_square<float, CONFIG_M_AXI_WIDTH>(inputTn, outputTn, len);
}
}

