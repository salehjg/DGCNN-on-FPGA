/*
** ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
** ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x128x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=2, Shape1=5x1024x20x64x, 
** ReduceMax: reductionDim=1, Shape1=5x1024x1x1024x, 
**
** ReduceMax: reductionDim=1, DIM2 SHOULD BE EQUALS TO ONE, ARGS(0,1,2)=DIM0x(DIM1)xDIM3,
** ReduceMax: reductionDim=2,                             , ARGS(0,1,2)=[DIM0*DIM1]x(DIM2)xDIM3
*/
#include <stdio.h>
#include <string.h>
#include <hls_math.h>

#define CONFIG_SLICE_SIZE  1024
#define CONFIG_SLICE_COUNT 2

extern "C" {
void task_reducemax(
        float* inputTn,
        float* outputTn,
		const unsigned int dim0,
		const unsigned int dim1,
		const unsigned int dim2,
		const int overaxis0,
		const int overaxis1,
		const int overaxis2){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control

#pragma HLS INTERFACE s_axilite port=overaxis0	bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1	bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2	bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

    float buff[CONFIG_SLICE_COUNT+1][CONFIG_SLICE_SIZE];

    if(!overaxis2 && overaxis1 && !overaxis0){
        unsigned long indxS,indxD;

		for(int d0=0;d0<dim0;d0++){
			for(int d1=0;d1<dim1;d1+=CONFIG_SLICE_COUNT){

				//reading slices to local buffers while checking bounds before burst reading.
				for(int s=0;s<CONFIG_SLICE_COUNT;s++){
					int d1_ex = d1 + s;
					if(d1_ex<dim1){
						indxS = d0*dim1*dim2 + (d1_ex)*dim2 + 0;
						//memcpy(dst,src,len in bytes)
						//buf[0] is for final results only.
						memcpy(&buff[1+s][0], &inputTn[indxS], dim2*sizeof(float));
					}
				}

				//comparing buff[1] and buff[2] and then comparing maxed result with buff[0]
				//so we have a maxing engine working parallel on "CONFIG_SLICE_COUNT" slices and 
				//then comparing the results with last run's results and updating final(buff[0]).
				for(int s=0;s<CONFIG_SLICE_COUNT;s+=2){
						//Keep nested loop perfect and avoid if statements here.

						//compare elements of slice0 and slice "s"
						//pay attention to the const bound of the loop.
						for(int d2=0;d2<CONFIG_SLICE_SIZE;d2++){
							int d1_ex = d1 + s;
							if(d1_ex<dim1){
								if(d2<dim2){
									float max = buff[1+s][d2] > buff[1+s+1][d2] ?
												buff[1+s][d2] 	:
												buff[1+s+1][d2] ;

									if(d1==0){
										//skip comparison of max val with buff[0] if buff[0] has not initialized yet.
										buff[0][d2] = max;
									}else{
										buff[0][d2] = max > buff[0][d2] ? max : buff[0][d2];
									}
								}
							}
						}
					
				}

				//Writing final results(buff[0]) to output tensor:
				indxD = d0*dim2 + 0;
				memcpy(&outputTn[indxD], &buff[0][0], dim2*sizeof(float));


			}
		}




    }
}
}
