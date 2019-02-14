kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_tile(
		global const float * __restrict__ g_idata,
		global float * __restrict__ g_odata,
		unsigned int dim0,
		unsigned int dim1,
		unsigned int dim2,
		unsigned int dim3,
		unsigned int rank,
		unsigned int tileAxis,
		unsigned int tileCount){
	/*
		Rank=4 : dim0~dim3 :: shape[0]~shape[3]
		Rank=3 : dim0~dim2 :: shape[0]~shape[2]
	*/

	unsigned long indxS1,indxD;
	if(rank==4 && tileAxis==2) {
		unsigned int B,N,K,D;
		B = dim0;
		N = dim1;
		D = dim3;
		K = tileCount;

		//tile ing input of shape BxNxD into BxNxKxD.
		for (int b = 0; b < B; b++) {
			for (int n = 0; n < N; n++) {
				indxS1 = b * N * D + n * D + 0; //beginning of dim2 of input
				for (int k = 0; k < K; k++) {
					indxD = b * N * K * D + n * K * D + k * D + 0;
					for(int i=0;i<D;i++){
						g_odata[indxD+i] = g_idata[indxS1+i];
					}
					//std::copy(inputTn->_buff + indxS1,
					//		  inputTn->_buff + indxS1 + D,
					//		  rsltTn->_buff + indxD);
				}
			}
		}
	}

	if(rank==3 && tileAxis==2) { //BxN = BxNx1   ------->  BxNxK  (PAGE 221 of my design notebook)
		unsigned int B,N,K,D;
		B = dim0;
		N = dim1;
		K = tileCount;

		//tile ing input of shape BxN or BxNx1 into BxNxK.
		for (int b = 0; b < B; b++) {
			for (int n = 0; n < N; n++) {
				indxS1 = b*N + n;
				for(int k=0;k<K;k++){
					indxD = b*N*K + n*K + k;
					g_odata[indxD] = g_idata[indxS1];
				}
			}
		}
	}

	if(rank==3 && tileAxis==1) { //BxN = Bx1xN   ------->  BxKxN  (PAGE 221 of my design notebook)
		unsigned int B,N,K,D;
		B = dim0;
		N = dim2;
		K = tileCount;

		//tile ing input of shape BxN or Bx1xN into BxKxN.
		for (int b = 0; b < B; b++) {
			for(int k=0;k<K;k++){
				for (int n = 0; n < N; n++) {
					indxD  = b*K*N + k*N + n;
					indxS1 = b*1*N + n;
					g_odata[indxD] = g_idata[indxS1];
				}
			}
		}
	}

	if(rank==2 && tileAxis==0) {
		unsigned int K,D;
		D = dim1;
		K = tileCount;

		//tile ing input of shape BxNxD into BxNxKxD.
		for (int k = 0; k < K; k++) {
			indxD = k * D + 0;
			for(int i=0;i<D;i++){
				g_odata[indxD+i] = g_idata[i];
			}
			//std::copy(inputTn->_buff ,
			//		  inputTn->_buff + D,
			//		  rsltTn->_buff + indxD);
		}
	}
}
