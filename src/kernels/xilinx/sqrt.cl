kernel void ndrange_sqrt(global const float * __restrict__ g_idata, global float * __restrict__ g_odata, const unsigned long len){
    unsigned long idx = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<len){
        g_odata[idx] = sqrt(g_idata[idx]);
    }
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_sqrt(global const float * __restrict__ g_idata, global float * __restrict__ g_odata, const unsigned long len){
    for(unsigned long d=0;d<len;d++) {
    	g_odata[d] = sqrt(g_idata[d]);
    }
}

