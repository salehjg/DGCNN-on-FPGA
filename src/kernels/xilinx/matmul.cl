kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_matmul(
		global const float *g_idataA,
		global const float *g_idataB,
		global float * g_odata,
		const unsigned int dim0A,
		const unsigned int dim1A,
		const unsigned int dim2A,
		const unsigned int dim0B,
		const unsigned int dim1B,
		const unsigned int dim2B){

    unsigned int batchsize = dim0A;
    unsigned int matrixH1  = dim1A;
    unsigned int matrixW1  = dim2A;
    unsigned int matrixH2  = dim1B;
    unsigned int matrixW2  = dim2B;
    unsigned long indxS1,indxS2,indxD;

	for(int b=0;b<batchsize;b++){
		// for element of output of matrixH1 x matrixW2
		for(int j=0;j<matrixH1;j++){
			for(int i=0;i<matrixW2;i++){
				//mat1: select row j
				//mat2: select col i
				float sum=0;
				for(int mat1_x=0;mat1_x<matrixW1;mat1_x++)
				{
					indxS1 = b*matrixH1*matrixW1 +
							 j*matrixW1 + mat1_x;
					/*indxS2 = b*matrixH2*matrixW2 +
							 mat1_x*matrixW1 + j;*/
					indxS2 = b*matrixH2*matrixW2 +
							 mat1_x*matrixW2 + i;

					sum += g_idataA[indxS1] * g_idataB[indxS2];
				}
				// for element of output of matrixH1 x matrixW2
				indxD = b*matrixH1*matrixW2 +
						j*matrixW2 + i;
				g_odata[indxD] = sum;

			}
		}
	}
}
