#include <iostream>
#include <cassert>

//The latency is reported for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20

template<typename DType>
void Gather(
	const DType* inputTn,
	const int*   indicesTn,
	DType* outputTn,
	int indices_axis,
	int inputDim0,
	int inputDim1,
	int inputDim2,
	int indicesDim0,
	int indicesDim1,
	int indicesDim2){

	assert(inputDim0 == indicesDim0);
	assert(inputDim1 == indicesDim1);
	assert(indices_axis == 1);

	unsigned long indxS1, indxS2, indxD;
	unsigned long BxNxKxD = indicesDim0 * indicesDim1 * indicesDim2 * inputDim2;
	int d0idx, d1idx, d2idx,d2input;
	d0idx = 0;
	d1idx = 0;
	d2idx = 0;
	d2input = 0;

	//Nested loop for B,N,K,D
	LoopIter: for(unsigned long iter=0; iter<BxNxKxD; iter++){
#pragma HLS LOOP_TRIPCOUNT min=6553600 max=6553600

		// Only calculate this on start of the loop D
		if(d2input==0){
			indxS1 = d0idx*indicesDim1*indicesDim2 + d1idx*indicesDim2 + d2idx;
		}

		indxD = d0idx*indicesDim1*indicesDim2*inputDim2 + d1idx*indicesDim2*inputDim2 + d2idx*inputDim2 + d2input;
		indxS2 = d0idx*indicesDim1*inputDim2 + indicesTn[indxS1]*inputDim2 + d2input;
		outputTn[indxD] = inputTn[indxS2];
		//========================================
		if(d2input==inputDim2-1){
			d2input = 0;
			if(d2idx==indicesDim2-1){
				d2idx = 0;
				if(d1idx==indicesDim1-1){
					d1idx=0;
					d0idx++;
				}else{
					d1idx++;
				}
			}else{
				d2idx++;
			}
		}else{
			d2input++;
		}
	}
	/*
	unsigned int
		B = dimA0,
		N = dimA1,
		K = dimB2,
		D = dimA2;

	for(int b=0;b<B;b++){
		for(int n=0;n<N;n++){
			for(int k=0;k<K;k++){
				indxS1 = b*N*K + n*K + k;
				for(int d=0;d<D;d++)
				{
					indxD = b*N*K*D + n*K*D + k*D + d;
					indxS2 = b*N*D +
							 inputTn2[indxS1]*D +
							 d;
					outputTn[indxD] = inputTn1[indxS2];
				}
			}
		}
	}
	*/
}

extern "C"{
void task_gather(
	const float* inputTn,
	const int*   indicesTn,
	float* outputTn,
	int indices_axis,
	int inputDim0,
	int inputDim1,
	int inputDim2,
	int indicesDim0,
	int indicesDim1,
	int indicesDim2){
#pragma HLS INTERFACE m_axi     port=inputTn   		offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesTn  	offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn  		offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn   		bundle=control
#pragma HLS INTERFACE s_axilite port=indicesTn  	bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  		bundle=control

#pragma HLS INTERFACE s_axilite port=indices_axis   bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim0      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim1      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim2      bundle=control

#pragma HLS INTERFACE s_axilite port=indicesDim0    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim1    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim2    bundle=control

#pragma HLS INTERFACE s_axilite port=return     	bundle=control

	Gather<float>(inputTn,indicesTn,outputTn,indices_axis,inputDim0,inputDim1,inputDim2,indicesDim0,indicesDim1,indicesDim2);
}
}
