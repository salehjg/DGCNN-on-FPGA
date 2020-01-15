#include <iostream>

/*
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x64x, ,   Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x20x128x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
** ReduceSum4D: , Shape1=5x1024x1x1024x, ,  Combination=1-1-1-0-,
 */
#define CONFIG_SLICE_SIZE       1024
#define CONFIG_MAX_POW_Y        3

#define MAX_POW_Y_MINUS_ONE     (CONFIG_MAX_POW_Y-1)

extern "C" {

void task_reducesum4d(
        const float* inputTn,
        float* outputTn,
        const int pow_y,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3){

#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

#pragma HLS INTERFACE s_axilite port=pow_y      bundle=control

#pragma HLS INTERFACE s_axilite port=dim0       bundle=control
#pragma HLS INTERFACE s_axilite port=dim1       bundle=control
#pragma HLS INTERFACE s_axilite port=dim2       bundle=control
#pragma HLS INTERFACE s_axilite port=dim3       bundle=control

#pragma HLS INTERFACE s_axilite port=overaxis0  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis1  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis2  bundle=control
#pragma HLS INTERFACE s_axilite port=overaxis3  bundle=control

#pragma HLS INTERFACE s_axilite port=return     bundle=control

    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        float buff_tmp[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff_tmp complete dim=0
        float buff_rslt[CONFIG_SLICE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=buff_rslt complete dim=0

        unsigned long indxS;
        unsigned long d0d1d2 = dim0 * dim1 * dim2;
        int pow_y_minus_one = pow_y -1;

        //Fused loops of dim0 and dim1:
        for(unsigned long iter=0, d0=0, d1=0, d2=0 ; iter<d0d1d2; iter++){
#pragma HLS LOOP_TRIPCOUNT min=102400 max=102400
            //Content of loop dim1 should be here:

            indxS = (d0)*dim1*dim2*dim3 + (d1)*dim2*dim3 + (d2)*dim3 + 0;

            //Read 1 slice of dim2 from input tensor(burst read):
            if(d0==0 && d1==0 && d2==0){
                //Read first slice into reduced buffer
                LoopRead1:for(int i=0;i<dim3;i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
                    buff_rslt[i] = 0;
                    //std::cout<<"indxI="<<indxS+i << " ,buff_rslt="<< buff_rslt[i] <<std::endl;
                }
            }else{
                //Read others into temp buffer
                LoopRead2:for(int i=0;i<dim3;i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
                    buff_tmp[i] = inputTn[indxS+i];
                    //std::cout<<"indxI="<<indxS+i << " ,buff_tmp= "<< buff_tmp[i] <<std::endl;
                }
            }


            //add cached slice to reduced slice(buff_rslt)
            LoopReduction:for(int i=0;i<CONFIG_SLICE_SIZE;i++){
//#pragma HLS UNROLL
#pragma HLS PIPELINE
                if(i<dim3){
                    //std::cout<<"**indxI="<<i << " ,buff_tmp= "<< buff_tmp[i] << " ,buff_rslt= "<< buff_rslt[i] <<std::endl;
                    float pow_rslt = buff_tmp[i];
                    LoopPow:for(int ipwr=0;ipwr<(MAX_POW_Y_MINUS_ONE);ipwr++){
#pragma HLS UNROLL
                        if(ipwr<pow_y_minus_one){
                            pow_rslt = pow_rslt * pow_rslt;
                        }
                    }
                    //std::cout<<"**indxI="<<i << " ,pow_rslt= "<< pow_rslt <<std::endl;
                    buff_rslt[i] = buff_rslt[i] + pow_rslt;
                    //std::cout<<"**indxI="<<i << " ,Sum= "<< buff_rslt[i] <<std::endl;

                }
            }
            

            //=====================================================
            //House keeping if-statements for fused loops:
            if(d2==dim2-1){
                d2=0;
                if(d1==dim1-1){
                    d0++;
                    d1=0;
                }else{
                    d1++;
                }
            }else{
                d2++;
            }
        }

        //After processing all dim2 slices within the inputTn,
        //write back reduced slice into output tensor
        LoopWrite:for(int i=0;i<dim3;i++){
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
            outputTn[i] = buff_rslt[i];
        }

    }
}

}
