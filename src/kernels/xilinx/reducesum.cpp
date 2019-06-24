/*
* Shape=5x1024x3    FFT
* Shape=5x1024x64   FFT
*/

#include <hls_stream.h>

#define CONFIG_SLICE_SIZE               64
#define CONFIG_OUTPUT_BUFF_SIZE         32

/*
// Non-dataflow versions (simple loop and parallel reduction)

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

        //Set unused elements of local buffer to zero.
        LoopSetZero:for(int i=dim2;i<CONFIG_SLICE_SIZE;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            buff[i] = 0;
        }

        //Read 1 slice of dim2 from input tensor(burst read):
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



        ////Simple for-loop reduction
        //for(int i = 1 ; i< CONFIG_SLICE_SIZE;i++){
        //	buff[0] += buff[i];
        //}


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
*/

//=========================================================================================================
//=========================================================================================================

void SubfucSliceReadBurst(
		const float *inputTn,
		hls::stream<float> &inStream,
		unsigned int dim0,
		unsigned int dim1,
		unsigned int dim2,
		int d0,
		int d1){
    LoopRead: for(int i=0;i<dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
    	inStream << inputTn[(d0)*dim1*dim2 + (d1)*dim2 + i];
    }
}

void SubfuncSliceReduceSum(
		hls::stream<float> &inStream,
		hls::stream<float> &outStream,
		unsigned int dim2){
    //Simple for-loop reduction
	float sum = 0;
    LoopReduce: for(int i = 0 ; i< dim2;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
#pragma HLS PIPELINE
    	sum += inStream.read();
    }
    outStream<<sum;
}

void SubfuncSliceWrite(
		float* outputTn,
		hls::stream<float> &inStream,
		unsigned int dim1,
		int d0,
		int d1){
    outputTn[d0*dim1 + d1] = inStream.read();
}

// Dataflow Version
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
    hls::stream<float> datastream1;
#pragma HLS STREAM variable=datastream1  depth=32
    hls::stream<float> datastream2;
#pragma HLS STREAM variable=datastream2  depth=32

    LoopDim0: for(int d0=0; d0<dim0; d0++){
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
    	LoopDim1: for(int d1=0; d1<dim1; d1++){
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
#pragma HLS DATAFLOW

			//Read 1 slice of dim2 from input tensor(burst read):
			SubfucSliceReadBurst(inputTn, datastream1, dim0, dim1, dim2, d0, d1);

			////Simple for-loop reduction
			SubfuncSliceReduceSum(datastream1,datastream2, dim2);

			//outputTn is of shape Dim0xDim1
			SubfuncSliceWrite(outputTn, datastream2, dim1, d0, d1);
		}
    }
}
}
