kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_transpose(
		global const float * __restrict__ g_idata,
		global float * __restrict__ g_odata,
		const unsigned int dim0,
		const unsigned int dim1,
		const unsigned int dim2){
	
	unsigned long indxS=0;
	unsigned long indxD=0;

	for(int b=0;b<dim0;b++)
	{
		for (int j = 0; j < dim1; j++) {
			for (int i = 0; i < dim2 ; i++) {
				indxS = b * dim1 * dim2 + j * dim2 + i;
				indxD = b * dim1 * dim2 + i * dim1 + j;
				g_odata[indxD] = g_idata[indxS];
			}
		}
	}
}

