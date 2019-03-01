kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_reducesum(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2){
	unsigned long indxS=0;
	unsigned long indxD=0;
	//-----------------------------------------------------TTT:
	if(overaxis0==true &&
	   overaxis1==true &&
	   overaxis2==true ){
		unsigned long limit = dim0*dim1*dim2;
		for(unsigned long b=0;b<limit;b++) {
			g_odata[0] += g_idata[b];
		}
	}
	//-----------------------------------------------------TFF:
	if(overaxis0==true &&
	   overaxis1==false &&
	   overaxis2==false ){
		float sum=0;
		for(int d1=0; d1<dim1;d1++){
			for(int d2=0;d2<dim2;d2++){
				sum=0;
				indxD = d1 * dim2 + d2;
				//sum over dim of interest
				for(int dx=0;dx<dim0;dx++)
				{
					indxS = dx * dim1*dim2 + d1 * dim2 + d2;
					sum += g_idata[indxS] ;
				}
				g_odata[indxD] = sum;
			}
		}
	}
	//-----------------------------------------------------FTF:
	if(overaxis0==false &&
	   overaxis1==true &&
	   overaxis2==false ){
		float sum=0;
		for(int d0=0; d0<dim0;d0++){
			for(int d2=0;d2<dim2;d2++){
				sum=0;
				indxD = d0 *dim2 + d2;
				//sum over dim of interest
				for(int dx=0;dx<dim1;dx++)
				{
					indxS = d0 * dim1*dim2 + dx * dim2 + d2;
					sum+=g_idata[indxS] ;
				}
				g_odata[indxD] = sum;
			}
		}
	}
	//-----------------------------------------------------FFT:
	if(overaxis0==false &&
	   overaxis1==false &&
	   overaxis2==true ){
		float sum=0;
		for(int d0=0; d0<dim0;d0++){
			for(int d1=0;d1<dim1;d1++){
				sum=0;
				indxD = d0 * dim1 + d1;
				//sum over dim of interest
				for(int dx=0;dx<dim2;dx++)
				{
					indxS = d0 * dim1*dim2 + d1 * dim2 + dx;
					sum+=g_idata[indxS] ;
				}
				g_odata[indxD] = sum;
			}
		}
	}
	//-----------------------------------------------------
}
