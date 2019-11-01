//Latency is for 5x1024x20x64 and 5x1024x20x64
extern "C" {
void task_matops(
		const float *inputTn1,
		const float *inputTn2,
		float * outputTn,
		const unsigned int dim0,
		const unsigned int dim1,
		const unsigned int dim2,
		const unsigned int dim3,
		const unsigned int dim0B,
		const unsigned int dim1B,
		const unsigned int dim2B,
		const unsigned int dim3B,
		const int dim0B_IsNotZero,
		const int dim1B_IsNotZero,
		const int dim2B_IsNotZero,
		const int dim3B_IsNotZero,
		const int mode){

#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control
#pragma HLS INTERFACE s_axilite port=dim3       bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B_IsNotZero  bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B_IsNotZero  bundle=control

#pragma HLS INTERFACE s_axilite port=mode  		bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

	unsigned long indxS1,indxS2;
	const unsigned long len = dim0*dim1*dim2*dim3;
	int d0,d1,d2,d3;
	d0=0;
	d1=0;
	d2=0;
	d3=0;

	//Fused loops for dim0 to dim3
	for(unsigned long iter=0; iter<len;iter++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=6553600 max=6553600
		//Content of loop dim3 should be here

		indxS1 = 	d0*dim1*dim2*dim3+
					d1*dim2*dim3+
					d2*dim3+
					d3;
		indxS2 = 	d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
				 	d1 * dim2B * dim3B * dim1B_IsNotZero +
					d2 * dim3B * dim2B_IsNotZero +
					d3 * dim3B_IsNotZero;

		if(mode==0)//Add
		{
			outputTn[indxS1] = inputTn1[indxS1] + inputTn2[indxS2];
		}
		else if(mode==1)//Sub
		{
			outputTn[indxS1] = inputTn1[indxS1] - inputTn2[indxS2];
		}
		else if(mode==2)//Mul (element wise)
		{
			outputTn[indxS1] = inputTn1[indxS1] * inputTn2[indxS2];
		}
		else if(mode==3)//Div (element wise)
		{
			outputTn[indxS1] = inputTn1[indxS1] / inputTn2[indxS2];
		}

        //=====================================================
        //House keeping if-statements for fused loops:
        if(d3==dim3-1){
        	d3=0;
            if(d2==dim2-1){
            	d2=0;
                if(d1==dim1-1){
                	d1=0;
                	d0++;
                }else{
                	d1++;
                }

            }else{
                d2++;
            }
        }else{
            d3++;
        }
	}

}
}
