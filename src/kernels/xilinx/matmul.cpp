#include <cassert>

#define CONFIG_BUFF_DEPTH           2
#define CONFIG_MAX_COMMON_DIM       1024

template <typename DType>
void BatchMatMulAXI32(
    DType *inputTn1, //Same as MatA
    DType *inputTn2, //Same as MatB
    DType *outputTn,

    unsigned int dim0A, //Batch
    unsigned int dim1A, //rows
    unsigned int dim2A, //cols

    unsigned int dim0B, //Batch
    unsigned int dim1B, //rows
    unsigned int dim2B  //cols
){
    assert(dim0A==dim0B);
    assert(dim2A==dim1B);

    int commonDim = dim2A;
    unsigned long indxS1,indxD;

    DType buff1[CONFIG_BUFF_DEPTH][CONFIG_MAX_COMMON_DIM]; //Row major buffer
    DType buff2[CONFIG_MAX_COMMON_DIM][CONFIG_BUFF_DEPTH]; //Row major buffer


    //Because burst reading a window of MatB is slower than MatA, outermost for-loop must belong to MatB to maximize data reuse for this matrix.
    LoopBatch:for(int batch=0; batch<dim0A; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopTiles1:for(int d2B=0; d2B<dim2B; d2B+=CONFIG_BUFF_DEPTH){
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
            LoopTiles2:for(int d1A=0; d1A<dim1A; d1A+=CONFIG_BUFF_DEPTH){
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
                //===============================================================================================================================
                //Sec. 1 : Burst read current window of matrix B only once in the begining of the loop. (TRYING TO MAXIMIZE DATA REUSE)
                //Avoiding logic between nested loops to make some optimization techniques possible.
                if(d1A==0){

                    int valid_w;
                    if(d2B+CONFIG_BUFF_DEPTH > dim2B){
                        valid_w = dim2B - d2B;
                    }else{
                        valid_w = CONFIG_BUFF_DEPTH;
                    }

                    ///TODO: INIT THE WHOLE LOCAL BUFF2 TO ZERO, BECAUSE DOING IT INSIDE BURST READ LOOP
                    ///      WILL CAUSE PROBLEMS AGAINST BURST READ OPERATION INFERMENT.
                    /*
                    LoopInitZero0:for(int j=0; j<CONFIG_MAX_COMMON_DIM; j++){
                        LoopInitZero1:for(int i=0; i<CONFIG_BUFF_DEPTH; i++){
                            buff2[j][i] = 0;
                        }
                    }
                    */
                    LoopInitZero0:for(int j=commonDim; j<CONFIG_MAX_COMMON_DIM; j++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                        LoopInitZero1:for(int i=valid_w; i<CONFIG_BUFF_DEPTH; i++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                            buff2[j][i] = 0;
                        }
                    }

                    //burst reading the separated rows in burst mode
                    LoopBurstReadA_J:for(int j=0; j<commonDim;j++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        LoopBurstReadA_I:for(int i=0; i<valid_w;i++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
                            indxS1 = (batch)*commonDim*dim2B + (j)*dim2B + (d2B+i);
                            buff2[j][i] = inputTn2[indxS1];
                        }
                    }
                }

                //===============================================================================================================================
                //Sec. 2 : Burst read different windows of matrix A. (LESS DATA REUSE)
                int valid_h,valid_len;
                if(d1A+CONFIG_BUFF_DEPTH > dim1A){
                    valid_h = dim1A - d1A;
                }else{
                    valid_h = CONFIG_BUFF_DEPTH;
                }
                valid_len = valid_h * commonDim;

                ///TODO: INIT THE WHOLE LOCAL BUFF1 TO ZERO, BECAUSE DOING IT INSIDE BURST READ LOOP
                ///      WILL CAUSE PROBLEMS AGAINST BURST READ OPERATION INFERMENT.
                /*
                 LoopInitZero2:for(int j=0; j<CONFIG_BUFF_DEPTH; j++){
                    LoopInitZero3:for(int i=0; i<CONFIG_MAX_COMMON_DIM; i++){
                        buff1[j][i] = 0;
                    }
                }
                */
                 LoopInitZero2:for(int j=valid_h; j<CONFIG_BUFF_DEPTH; j++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                    LoopInitZero3:for(int i=commonDim; i<CONFIG_MAX_COMMON_DIM; i++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
                        buff1[j][i] = 0;
                    }
                }

                //burst reading the whole buffer size worth of elements of matA
                LoopBurstReadB_J:for(int j=0; j<valid_h;j++){
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
                    LoopBurstReadB_I:for(int i=0; i<commonDim;i++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        indxS1 = (batch)*dim1A*commonDim + (d1A+j)*commonDim + (i);
                        buff1[j][i] = inputTn1[indxS1];
                    }
                }

                //=====================================================================================
                //Sec. 3: Process content of two local buffers and produce the output elements for (x, y) .
                int x,y;
                x=0;
                y=0;
                int xx,yy;
                //Fused loops for (x,y), pay attention that virtual x-loop is the inner loop. So gmem writes will be in order.
                LoopProcess1:for(int iter=0; iter<CONFIG_BUFF_DEPTH*CONFIG_BUFF_DEPTH; iter++){
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
                    DType sum=0;
                    LoopProcess2:for(int q=0; q<commonDim; q++){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        sum += buff1[y][q] * buff2[q][x];
                    }
                    //Writing output as row-major array.
                    xx = x+d2B;
                    yy = y+d1A;
                    indxD = (batch)*dim1A*dim2B + (yy)*dim2B + (xx);
                    if(yy<dim1A && xx<dim2B){
                        outputTn[indxD] = sum;
                    }

                    //=====================================
                    if(x == CONFIG_BUFF_DEPTH-1){
                        x=0;
                        y++;
                    }else{
                        x++;
                    }
                }

            }
        }
    }

}
extern "C"{
void task_matmul(
        float *inputTn1,
        float *inputTn2,
        float *outputTn,

        unsigned int dim0A,
        unsigned int dim1A,
        unsigned int dim2A,

        unsigned int dim0B,
        unsigned int dim1B,
        unsigned int dim2B){

#pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0A       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1A       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2A       bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

    BatchMatMulAXI32<float>(inputTn1, inputTn2, outputTn, dim0A, dim1A, dim2A, dim0B, dim1B, dim2B);
}
}
