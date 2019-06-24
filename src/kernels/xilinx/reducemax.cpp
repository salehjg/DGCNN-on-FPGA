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

#define CONFIG_SLICE_SIZE  		1024
//#define CONFIG_SLICE_COUNT 		5


extern "C" {



/*
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
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    if(!overaxis2 && overaxis1 && !overaxis0){
        unsigned long indxS,indxD;

		LoopD0:for(int d0=0;d0<dim0;d0++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
			LoopD1:for(int d1=0;d1<dim1;d1+=CONFIG_SLICE_COUNT){
#pragma HLS LOOP_TRIPCOUNT min=20 max=20

				//reading slices to local buffers while checking bounds before burst reading.
				LoopRead1:for(int s=0;s<CONFIG_SLICE_COUNT;s++){
					int d1_ex = d1 + s;
					if(d1_ex<dim1){
						indxS = d0*dim1*dim2 + (d1_ex)*dim2 + 0;
						//memcpy(dst,src,len in bytes)
						//buf[0] is for final results only.
						//memcpy(&buff[1+s][0], &inputTn[indxS], dim2*sizeof(float));
						//-------------------------------------------------------------------
						//Using for-loop for burst reading because trip-count pragma is not available for memcpy
						LoopCPY1:for(unsigned long i=0;i<dim2;i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
							buff[1+s][i] = inputTn[indxS+i];
						}
					}
				}

				//comparing buff[1] and buff[2] and then comparing maxed result with buff[0]
				//so we have a maxing engine working parallel on "CONFIG_SLICE_COUNT" slices and
				//then comparing the results with last run's results and updating final(buff[0]).
				LoopCMP1:for(int s=0;s<CONFIG_SLICE_COUNT;s+=2){
						//Keep nested loop perfect and avoid if statements here.

						//compare elements of slice0 and slice "s"
						//pay attention to the const bound of the loop.
						LoopCMP2:for(int d2=0;d2<CONFIG_SLICE_SIZE;d2++){
#pragma HLS UNROLL factor=4
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
				//memcpy(&outputTn[indxD], &buff[0][0], dim2*sizeof(float));
				//----------------------------------------------------------
				//using for-loop for burst writing because tripcount pragma is not available for memcpy
				LoopCPY2:for(unsigned long i=0;i<dim2;i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
					outputTn[indxD+i] = buff[0][i];
				}


			}
		}




    }
}
*/

void task_reducemax(
        const float* inputTn,
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

    float buff_tmp[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0
    float buff_rslt[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    if(!overaxis2 && overaxis1 && !overaxis0){
        unsigned long indxS,indxD;
        unsigned long d0d1 = dim0 * dim1;

        //Fused loops of dim0 and dim1:
        for(unsigned long iter=0, d0=0, d1=0 ; iter<d0d1; iter++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
        	//Content of loop dim1 should be here:
			
			indxS = d0*dim1*dim2 + (d1)*dim2 + 0;

        	//Read 1 slice of dim2 from input tensor(burst read):
        	if(d1==0){
        		//Read first slice into reduced buffer
        		LoopRead1:for(int i=0;i<dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
					buff_rslt[i] = inputTn[indxS+i];	
        		}
        	}else{
        		//Read others into temp buffer
        		LoopRead2:for(int i=0;i<dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
					buff_tmp[i] = inputTn[indxS+i];	
        		}
        	}
        	
        	//Only compare if cached slice is not the first one.
        	if(d1!=0){
	        	//Compare cached slice with reduced slice(buff_rslt)
	        	LoopCompare:for(int i=0;i<CONFIG_SLICE_SIZE;i++){
#pragma HLS PIPELINE
	        		if(i<dim2){
						if(buff_tmp[i]>buff_rslt[i]){
							buff_rslt[i] = buff_tmp[i];
						}
	        		}
	        	}
        	}

        	//=====================================================
        	//House keeping if-statements for fused loops:
        	if(d1==dim1-1){
   				//Content after loop dim1 should be here:

        		indxD = d0*dim2 + 0;

        		//After processing all dim2 slices within current dim1 slice, 
        		//write back reduced slice into output tensor
        		LoopWrite:for(int i=0;i<dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
					outputTn[indxD + i] = buff_rslt[i];	
        		}

        		//---------------------------------
        		d1=0;
        		d0++;
        	}else{
        		d1++;
        	}
        }


    }
}

}
