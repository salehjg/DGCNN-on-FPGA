kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_gather(
		__global float* inputTn1, //Points : dimA0 x dimA1 x dimA2 ,(BxNxD) meaning B batches, N points with D features
		__global int* 	inputTn2, //Indices: dimA0 x dimA1 x dimB2 ,(BxNxK) meaning B batches, N points, each of them with K neighbors with D features
		__global float* outputTn, //Output : dimA0 x dimA1 x dimB2 x dimA2 ,(BxNxKxD)

		unsigned int dimA0,
		unsigned int dimA1,
		unsigned int dimA2,
		unsigned int dimB2){

	unsigned long indxS1, indxS2, indxD;
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

}
