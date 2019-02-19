kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_conv2d_mlp(
		global const float *gi_input,
		global const float *gi_weight,
		global const float *gi_bias,
		global float * go_data,
		const unsigned int B,
		const unsigned int N,
		const unsigned int K,
		const unsigned int D,
		const unsigned int ChOut){

	unsigned long indxD,indxS1,indxS2;

	for(int b=0;b<B;b++){
		for(int n=0;n<N;n++){
			for(int k=0;k<K;k++){
				indxS1 = b*N*K*D + n*K*D + k*D + 0;
				for(int ch=0;ch<ChOut;ch++){
					float sum=0;
					for(int d=0;d<D;d++){
						indxS2 = d*ChOut + ch;
						sum += gi_input[indxS1+d] * gi_weight[indxS2];
					}
					indxD=b*N*K*ChOut+ n*K*ChOut+ k*ChOut+ ch;
					go_data[indxD] = sum + gi_bias[ch];
				}
			}
		}
	}
	
}
