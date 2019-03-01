kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_square(
		global const float * __restrict__ g_idata,
		global float * __restrict__ g_odata,
		const unsigned long len){
    for(unsigned long i=0;i<len;i++){
    	g_odata[i] = g_idata[i] * g_idata[i];
    }
}

