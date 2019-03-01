kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_split_integer(
		global const int *g_idata,
        global int *g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int new_dim2) {

    unsigned long indxS,indxD;

	for(int d0=0;d0<dim0;d0++){
        for(int d1=0;d1<dim1;d1++){
            for(int d2=0;d2<new_dim2;d2++){
                indxS = d0*dim1*dim2 +
                        d1*dim2 +
                        d2;
                indxD = d0*dim1*new_dim2 +
                        d1*new_dim2 +
                        d2;   

                g_odata[indxD] = g_idata[indxS];        
            }
        }
	}

}