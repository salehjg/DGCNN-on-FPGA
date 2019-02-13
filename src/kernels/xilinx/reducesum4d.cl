//kernel_reduce_sum_4d_try05
kernel void ndrange_reducesum4d(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        local float *smem_buff,
        const int pow_y,
        const unsigned long slice_count,
        const unsigned long dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO)
{
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned long TPG = dim3; //threads per group
        unsigned long GroupIndex = get_group_id(0) * TGPB + get_local_id(0) / TPG;
        unsigned long _limit = slice_count*dim3;
        unsigned long GroupIndexWithinBlock = (GroupIndex - get_group_id(0) *TGPB);

        {
            // Fill unused shared mem with zeros! - Share memory is NOT initialized to zero
            // https://stackoverflow.com/questions/22172881/why-cuda-shared-memory-is-initialized-to-zero?noredirect=1&lq=1
            smem_buff[get_local_id(0)] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);


        // Ignore incomplete groups at the end of each thread block.
        if( GroupIndexWithinBlock < TGPB && GroupIndex < TGC){
            //Thread Index In Thread Group
            unsigned long TIITG = (get_local_id(0) - (GroupIndexWithinBlock*TPG));
            float thread_sum = 0.0; // sum for current thread in the thread group

            //------------------------------------------------------------
            for(unsigned long iSPT=0;iSPT<SPT;iSPT++){
                unsigned long gidx =  TGO*GroupIndex + iSPT*dim3 + TIITG;
                if(gidx < _limit ){

                    //if(blockIdx.x==35)
                    //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, _limit:%06ld, gidx: %06ld\n",
                    //        blockIdx.x, threadIdx.x, GroupIndex, TIITG,_limit, gidx);
                    float pow_rslt=g_idata[gidx];
                    for(int ipwr=0;ipwr<pow_y-1;ipwr++){
                        pow_rslt = pow_rslt * pow_rslt;
                    }
                    thread_sum += pow_rslt;
                }
            }

            //------------------------------------------------------------
            // |---element0s---|-----element1s----|------....|
            smem_buff[TIITG*TGPB + GroupIndexWithinBlock] = thread_sum;
        }


        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction of current block's shared memory buffer
        unsigned int thid = get_local_id(0);
        while(thid<TGPB){ // block stride loop

            for(unsigned long stride=TGPB/2; stride>0; stride >>= 1){
                if (thid < stride){
                    for(int d3=0;d3<TPG;d3++){
                        smem_buff[d3*TGPB + (thid)] += smem_buff[d3*TGPB + (thid+stride)];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            // -------------------
            thid += get_local_size(0);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(get_local_id(0)==0) {
            for (int d3 = 0; d3 < TPG; d3++) {
                g_odata[get_group_id(0) * dim3 + d3] = smem_buff[d3 * TGPB];
                //printf("** bid: %06d, tid: %06d, GroupIndex: %06ld, g_out_idx: %06ld, val: %f\n",
                //       blockIdx.x, threadIdx.x, GroupIndex, blockIdx.x * dim3 + d3,smem_buff[d3 * TGPB]);
            }
        }

    }

}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_reducesum4d(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        const int pow_y,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3){
	float sum=0;
    unsigned long indxS=0;
    unsigned long indxD=0;
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
		for (int d3 = 0; d3 < dim3; d3++){
			sum=0;
			indxD = d3;
			for (int d0 = 0; d0 < dim0; d0++) {
				for (int d1 = 0; d1 < dim1; d1++) {
					for (int d2 = 0; d2 < dim2; d2++) {

						indxS = d0*dim1*dim2*dim3+
						d1*dim2*dim3+
						d2*dim3+
						d3;

						float pow_rslt = g_idata[indxS];
						for(int ipwr=0;ipwr<pow_y-1;ipwr++){
							pow_rslt = pow_rslt * pow_rslt;
						}
						sum += pow_rslt;
					}
				}
			}

			g_odata[indxD] = sum;
		}

    }
}
