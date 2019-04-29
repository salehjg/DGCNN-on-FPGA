/*
* Shape=5x1024x3    FFT
* Shape=5x1024x64   FFT
*/

#define CONFIG_SLICE_SIZE               64
#define CONFIG_OUTPUT_BUFF_SIZE         32


extern "C" {
void task_reducesum(
        const float * inputTn,
        float * outputTn,
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

#pragma HLS INTERFACE s_axilite port=overaxis0  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2  bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

	//-----------------------------------------------------FFT:
    unsigned long indxS,indxD;
    unsigned long d0d1 = dim0 * dim1;
    float buff[CONFIG_SLICE_SIZE];
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=32 dim=1
    float buff_output[CONFIG_OUTPUT_BUFF_SIZE];

    //Fused loops of dim0 and dim1:
    for(unsigned long iter=0, d0=0, d1=0 ; iter<d0d1; iter++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        //Content of loop dim1 should be here:

        indxS = d0*dim1*dim2 + (d1)*dim2 + 0;

        //Read 1 slice of dim2 from input tensor(burst read):
        LoopSetZero:for(int i=dim2;i<CONFIG_SLICE_SIZE;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            buff[i] = 0;
        }

        //Read first slice into reduced buffer
        LoopRead1:for(int i=0;i<dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            buff[i] = inputTn[indxS+i];
        }

  		//Parallel Reduction - Interleaved Addressing with Cyclic Partitioning of Local Buffer
        //Compare cached slice with reduced slice(buff_rslt)
        LoopReduction :for(int s=CONFIG_SLICE_SIZE/2;s>0;s>>=1){
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN off
#pragma HLS DEPENDENCE variable=buff array inter RAW true
        	LoopIteration:for(int i=0;i<CONFIG_SLICE_SIZE/2;i++){
#pragma HLS LOOP_FLATTEN off
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable=buff array inter RAW false

				if(i<s){
					buff[i] += buff[i+s];
				}

        	}
        }


        /*
        //Simple for-loop reduction
        for(int i = 1 ; i< CONFIG_SLICE_SIZE;i++){
        	buff[0] += buff[i];
        }
        */

        indxD = d0*dim1 + d1; //outputTn is of shape Dim0xDim1
        outputTn[indxD] = buff[0];

        //=====================================================
        //House keeping if-statements for fused loops:
        if(d1==dim1-1){
            //Content after loop dim1 should be here:
            //---------------------------------
            d1=0;
            d0++;
        }else{
            d1++;
        }
    }
}
}
