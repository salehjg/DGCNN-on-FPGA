// COPYRIGHT TO CHARLESQ34 @ GitHub : PointNet++
// input: k (1), distance matrix dist (b,m,n)
// output: outi (b,m,n), out (b,m,n)
// only the top k results within n are useful
kernel void ndrange_topk(
        global const float * __restrict__ dist,
        global int * __restrict__ outi,
        global float * __restrict__ out,
        const int b,
        const int n,
        const int m,
        const int k) {
    //              get_group_id(uint dimindx)      and      blockIdx. [xyz]
    //              get_local_size(uint dimindx)    and      blockDim. [xyz]
    //              get_local_id(uint dimindx)      and      threadIdx.[xyz]
    //              get_num_groups(uint dimindx)    and      gridDim.  [xyz]
    int batch_index = get_group_id(0);
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = get_local_id(0);
    int stride = get_local_size(0);

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    global float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s;
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_topk(
        global const float *  dist, 	//dim0xdim1xdim2
        global int *  outi,				//dim0xdim1xdim2 (only first K elements over dim2 are valid)
        global float *  out,			//dim0xdim1xdim2 (only first K elements over dim2 are valid)
		const int dim0,
		const int dim1,
		const int dim2,
		const int k) {

	unsigned long indxS,indxD;

	for(int d0=0;d0<dim0;d0++){
		for(int d1=0;d1<dim1;d1++){
			indxS = d0*dim1*dim2 + d1*dim2 + 0;
			//copy input data into output and outputI

			for(int q=0;q<dim2;q++){
				out[indxS+q]  = dist[indxS+q];
				outi[indxS+q] = q;
			}
 

			//https://en.wikipedia.org/wiki/Selection_sort

			// a[0] to a[n-1] is the array to sort
			int i,j;
			int n=dim2; // initialise to a's length

			// advance the position through the entire array
			//   (could do j < n-1 because single element is also min element)
			for (j = 0; j < n-1; j++){
				// find the min element in the unsorted a[j .. n-1]
				// assume the min is the first element
				int iMin = j;
				// test against elements after j to find the smallest
				for (i = j+1; i < n; i++){
					// if this element is less, then it is the new minimum
					if (out[indxS+i] < out[indxS+iMin]){
						// found new minimum; remember its index
						iMin = i;
					}
				}

				if (iMin != j){

					{ 
						float val = out[indxS+j];
						out[indxS+j] = out[indxS+iMin];
						out[indxS+iMin] = val;
					}

					{
						int val = outi[indxS+j];
						outi[indxS+j] = outi[indxS+iMin];
						outi[indxS+iMin] = val;
					}

				} 
			}


		}
	}







}
