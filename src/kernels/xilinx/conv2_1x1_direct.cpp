#include <cassert>
#include <hls_stream.h>

#define CONFIG_MAX_WEIGHT_BUF_D2    (328)
#define CONFIG_MAX_WEIGHT_BUF_D3    (1024)
#define CONFIG_MAX_WEIGHT_BUF_SIZE  (CONFIG_MAX_WEIGHT_BUF_D2*CONFIG_MAX_WEIGHT_BUF_D3)
/*
template <typename DType>
void conv2mlp_try01(
    const DType *inputTn,
    const DType *weightTn,
    const DType *biasTn,
    DType *outputTn,
    unsigned int dim0D,
    unsigned int dim1D,
    unsigned int dim2D,
    unsigned int dim3D,
    unsigned int dim0W,
    unsigned int dim1W,
    unsigned int dim2W,
    unsigned int dim3W,
    unsigned int dim0B
    ){
    unsigned int B,N,K,D,ch_out;
    unsigned long indxS1,indxS2,indxD;

    B = dim0D;
    N = dim1D;
    K = dim2D;
    D = dim3D;

    ch_out = dim3W;

    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            for(int k=0;k<K;k++){
                indxS1 = b*N*K*D + n*K*D + k*D + 0;
                for(int ch=0;ch<ch_out;ch++){
                    float sum=0;
                    for(int d=0;d<D;d++){
                        indxS2 = d*ch_out + ch;
                        sum += inputTn[indxS1+d] * weightTn[indxS2];
                    }
                    indxD=b*N*K*ch_out+ n*K*ch_out+ k*ch_out+ ch;
                    outputTn[indxD] = sum + biasTn[ch];
                }
            }
        }
    }
}
*/

// Latency report is for InputTn=5x1024x1x320 and WeightTn=1x1x320x1024
template <typename DType>
void conv2_1x1_direct(
    const DType *inputTn,
    const DType *weightTn,
    const DType *biasTn,
    DType *outputTn,
    unsigned int dim0D,
    unsigned int dim1D,
    unsigned int dim2D,
    unsigned int dim3D,
    unsigned int dim0W,
    unsigned int dim1W,
    unsigned int dim2W,
    unsigned int dim3W,
    unsigned int dim0B
    ){
    unsigned long indxS1,indxS2,indxD;
    DType buff_weight[CONFIG_MAX_WEIGHT_BUF_D2][CONFIG_MAX_WEIGHT_BUF_D3];
    DType buff_bias[CONFIG_MAX_WEIGHT_BUF_D3];
    //hls::stream<float> inputTnStream;

    //Only 1x1 kernels
    assert(dim0W==1);
    assert(dim1W==1);
    unsigned long lenWeight = dim2W*dim3W;
    assert(lenWeight<CONFIG_MAX_WEIGHT_BUF_SIZE);



    // 0. Loading entire biasTn on local memory considering it is small enough and of rank one.
    For_B1:for(int i3=0;i3<CONFIG_MAX_WEIGHT_BUF_D3;i3++){
        if(i3<dim3W){
            buff_bias[i3] = biasTn[i3];
        }else{
            //buff_bias[i3] = 0;
        }
    }



    // 1. Loading entire weightTn on local memory considering it is small enough and of shape 1x1x*x*.
    For_W1:for(int i2=0;i2<CONFIG_MAX_WEIGHT_BUF_D2;i2++){
        For_W2:for(int i3=0;i3<CONFIG_MAX_WEIGHT_BUF_D3;i3++){
            if(i2<dim2W && i3<dim3W){
                buff_weight[i2][i3] = weightTn[i2*dim3W + i3];
            }else{
                //buff_weight[i2][i3] = 0;
            }
        }
    }



    // 2. Doing actual 1x1 convolution to mimic shared mlp layer. loops for b,n and k are fused together.
    int b,n,k,ch;
    b=0;
    n=0;
    k=0;
    const unsigned long d0d1d2w3 = dim0D * dim1D * dim2D;
    For_Fused:for(unsigned long iter=0; iter<d0d1d2w3; iter++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120

        indxS1 = b*dim1D*dim2D*dim3D + n*dim2D*dim3D + k*dim3D + 0;
        For_ChOut:for(int ch=0;ch<CONFIG_MAX_WEIGHT_BUF_D3;ch++){
#pragma HLS UNROLL factor=64
            if(ch<dim3W){
                float sum=0;
                For_D:for(int d=0;d<dim3D;d++){
#pragma HLS LOOP_TRIPCOUNT min=320 max=320
#pragma HLS PIPELINE II=1
                    //indxS2 = d*dim3W + ch;
                    sum += inputTn[indxS1+d] * buff_weight[d][ch];//weightTn[indxS2];
                }
                indxD=b*dim1D*dim2D*dim3W+ n*dim2D*dim3W+ k*dim3W+ ch;
                outputTn[indxD] = sum + buff_bias[ch];
            }
        }


        //-------------------------------------------------------
        if(k == dim2D-1){
            k=0;
            if(n == dim1D-1){
                n=0;
                b++;
            }else{
                n++;
            }
        }else{
            k++;
        }
    }





}

extern "C"{
void task_conv2_1x1_direct(
    const float *inputTn,
    const float *weightTn,
    const float *biasTn,
    float *outputTn,
    unsigned int dim0D,
    unsigned int dim1D,
    unsigned int dim2D,
    unsigned int dim3D,
    unsigned int dim0W,
    unsigned int dim1W,
    unsigned int dim2W,
    unsigned int dim3W,
    unsigned int dim0B){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=weightTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=biasTn     offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=weightTn   bundle=control
#pragma HLS INTERFACE s_axilite port=biasTn     bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=dim0D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2D      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3D      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2W      bundle=control
#pragma HLS INTERFACE s_axilite port=dim3W      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0B      bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control
    conv2_1x1_direct<float>(inputTn, weightTn, biasTn, outputTn, dim0D, dim1D, dim2D, dim3D, dim0W, dim1W, dim2W, dim3W, dim0B);

}
}
