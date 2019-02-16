kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_matops(
		global const float *g_idata1,
		global const float *g_idata2,
		global float * g_odata,
		const unsigned int dim0,
		const unsigned int dim1,
		const unsigned int dim2,
		const unsigned int dim3,
		const unsigned int dim0B,
		const unsigned int dim1B,
		const unsigned int dim2B,
		const unsigned int dim3B,
		const int dim0B_IsNotZero,
		const int dim1B_IsNotZero,
		const int dim2B_IsNotZero,
		const int dim3B_IsNotZero,
		const int mode){

	unsigned long indxS1,indxS2;
	for(int d0=0;d0<dim0;d0++){
		for(int d1=0;d1<dim1;d1++){
			for(int d2=0;d2<dim2;d2++){
				for(int d3=0;d3<dim3;d3++){
					indxS1 = d0*dim1*dim2*dim3+
							 d1*dim2*dim3+
							 d2*dim3+
							 d3;
					indxS2 = d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
							 d1 * dim2B * dim3B * dim1B_IsNotZero +
							 d2 * dim3B * dim2B_IsNotZero +
							 d3 * dim3B_IsNotZero;

					if(mode==0)//Add
					{
						g_odata[indxS1] = g_idata1[indxS1] + g_idata2[indxS2];
					}
					else if(mode==1)//Sub
					{
						g_odata[indxS1] = g_idata1[indxS1] - g_idata2[indxS2];
					}
					else if(mode==2)//Mul (element wise)
					{
						g_odata[indxS1] = g_idata1[indxS1] * g_idata2[indxS2];
					}
					else if(mode==3)//Div (element wise)
					{
						g_odata[indxS1] = g_idata1[indxS1] / g_idata2[indxS2];
					}
				}
			}
		}
	}

}
