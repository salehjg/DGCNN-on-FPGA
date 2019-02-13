kernel void ndrange_reducemax(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2)
{
    // WANING : dim0 means dim2 and dim2 means dim0
    if (!overaxis2 && overaxis1 && !overaxis0)
    {
        // Case 2 - sums in Y-direction
        // each thread is responsible for a separate Y-column sum
        unsigned int idx = (get_group_id(0) * get_local_size(0) + get_local_id(0));
        if (idx < (dim0*dim2))
        {
            unsigned int tidx = idx%dim0 + (idx/dim0)*(dim0*dim1); //indices over input tensor (begining of axis1 slices)

            float tMax = FLT_MIN;
            float gval ;

            for (unsigned int i = 0; i < dim1; i++)
            {
                //printf("idx: %03d \t\t tidx: %03d\n",idx,tidx);
                gval = g_idata[tidx];
                if(gval > tMax)tMax = gval;
                tidx += dim0;
            }

            g_odata[idx] = tMax;
        }
    }
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_reducemax(
		global const float * __restrict__  g_idata,
		global float * __restrict__  g_odata,
		const unsigned int dim0,
		const unsigned int dim1,
		const unsigned int dim2,
		const int overaxis0,
		const int overaxis1,
		const int overaxis2){
    if(!overaxis2 && overaxis1 && !overaxis0){
        unsigned long indxS,indxD;
        float max_cte= FLT_MIN;
        float max= FLT_MIN;

        //output is of shape dim0xdim2
        for(int d0=0;d0<dim0;d0++){
            for(int d2=0;d2<dim2;d2++){
                indxD = d0*dim2 + d2;
                max = max_cte;
                for(int d1=0;d1<dim1;d1++){
                    indxS = d0*dim1*dim2 + d1*dim2 + d2;
					if(max < g_idata[indxS]){
						max = g_idata[indxS];
					}
                }
                g_odata[indxD]=max;
            }
        }
    }
}
