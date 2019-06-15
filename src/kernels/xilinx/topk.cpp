#include <cassert>

void BatchSelectionSortTopK(
	const float* inputTn,
	int* indicesTn,
	int* indicesSplitedTn,
	float* outputTn,
	int dim0,
	int dim1,
	int dim2,
	int kValue){

	int i, j, max_idx;
	unsigned long indxS,indxD;
	assert(kValue<dim2);

	// 1. Copy inputTn into outputTn, so sorting algorithm could be
	//    run on outputTn without editing inputTn.
	for(unsigned long i = 0; i<dim0*dim1*dim2; i++){
		outputTn[i] = inputTn[i];
	}

	// 2. Initializing indicesTn for each of k-element slices of it.
	for(int batch=0; batch<dim0*dim1; batch++){
		for(int idx=0; idx<dim2; idx++){
			indicesTn[batch*dim2 + idx] = idx;
		}
	}

	// 3. Running descending selection sort only for first k elements of
	//    each of dim2 slices of outputTn(which is a clone of inputTn).
	for(int batch=0; batch<dim0*dim1; batch++){

		// Run selection sort on current slice of dim2.
		for (i = 0; i < kValue; i++)
		{
			// Find the maximum element in unsorted array
			max_idx = i;
			for (j = i+1; j < dim2; j++){
				if (outputTn[batch*dim2 + j] > outputTn[batch*dim2 + max_idx])
					max_idx = j;
			}

			// Swap the found maximum element with the first element
			if(max_idx != i){
				float tmp = outputTn[batch*dim2 + max_idx];
				outputTn[batch*dim2 + max_idx] = outputTn[batch*dim2 + i];
				outputTn[batch*dim2 + i] = tmp;
				//------------------------------------------------------------
				int tmpi = indicesTn[batch*dim2 + max_idx];
				indicesTn[batch*dim2 + max_idx] = indicesTn[batch*dim2 + i];
				indicesTn[batch*dim2 + i] = tmpi;
			}
		}

	}

	// 4. Splitting indicesTn which is of shape BxNxN into BxNxK
	for(int batch=0; batch<dim0*dim1; batch++){

		// Run selection sort on current slice of dim2.
		for (i = 0; i < kValue; i++){
			indxS = batch*dim2 + i;
			indxD = batch*kValue + i;

			indicesSplitedTn[indxD] = indicesTn[indxS];
		}
	}

}

extern "C"{
void task_topk(
	const float* inputTn,
	int* indicesTn,
	int* indicesSplitedTn,
	float* outputTn,
	int dim0,
	int dim1,
	int dim2,
	int kValue){
#pragma HLS INTERFACE m_axi     port=inputTn   				offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesTn  			offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=indicesSplitedTn  		offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   			offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=inputTn   			bundle=control
#pragma HLS INTERFACE s_axilite port=indicesTn  		bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn  	bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   		bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

	BatchSelectionSortTopK(inputTn, indicesTn, indicesSplitedTn, outputTn, dim0, dim1, dim2, kValue);

}
}
