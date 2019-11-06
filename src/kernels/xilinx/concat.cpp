/*
Shape1=5x1024x1x128x,   , Shape2=5x1024x1x64x, 
Shape1=5x1024x1x192x,   , Shape2=5x1024x1x128x, 
Shape1=5x1024x1x64x,    , Shape2=5x1024x1x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x64x,   , Shape2=5x1024x20x64x, 
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,  
Shape1=5x1024x20x3x,    , Shape2=5x1024x20x3x,
*/

void concat2(
    float* inputTn1,
    float* inputTn2,
    float* outputTn,
    unsigned int dimA0,
    unsigned int dimA1,
    unsigned int dimA2,
    unsigned int dimA3,
    unsigned int dimB0,
    unsigned int dimB1,
    unsigned int dimB2,
    unsigned int dimB3){

    unsigned int  dimR0,dimR1,dimR2,dimR3;
    dimR0 = dimA0;
    dimR1 = dimA1;
    dimR2 = dimA2;
    dimR3 = dimA3 + dimB3;
    unsigned long indxS1,indxS2,indxD;

    Loop1: for(int d0=0;d0<dimA0;d0++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        Loop2: for(int d1=0;d1<dimA1;d1++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            Loop3: for(int d2=0;d2<dimA2;d2++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=20
                Loop4: for(int d3=0;d3<dimA3;d3++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=192
#pragma HLS PIPELINE II=1

                    indxS1 = d0*dimA1*dimA2*dimA3 +
                             d1*dimA2*dimA3+
                             d2*dimA3+
                             d3;
                    indxD = (d0)*dimR1*dimR2*dimR3 +
                            (d1)*dimR2*dimR3+
                            (d2)*dimR3+
                            (d3);
                    outputTn[indxD] = inputTn1[indxS1];
                }
            }
        }
    }

    Loop5: for(int d0=0;d0<dimB0;d0++){
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        Loop6: for(int d1=0;d1<dimB1;d1++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            Loop7: for(int d2=0;d2<dimB2;d2++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=20
                Loop8: for(int d3=0;d3<dimB3;d3++){
#pragma HLS LOOP_TRIPCOUNT min=3 max=128
#pragma HLS PIPELINE II=1

                    indxS2 = d0*dimB1*dimB2*dimB3 +
                             d1*dimB2*dimB3+
                             d2*dimB3+
                             d3;
                    indxD  = (d0+0)*dimR1*dimR2*dimR3 +
                             (d1+0)*dimR2*dimR3+
                             (d2+0)*dimR3+
                             (d3+dimA3);
                    outputTn[indxD] = inputTn2[indxS2];
                }
            }
        }
     }
}

extern "C" {
void task_concat(
        float* inputTn1,
        float* inputTn2,
        float* outputTn,

        unsigned int dimA0,
        unsigned int dimA1,
        unsigned int dimA2,
        unsigned int dimA3,

        unsigned int dimB0,
        unsigned int dimB1,
        unsigned int dimB2,
        unsigned int dimB3){

    #pragma HLS INTERFACE m_axi     port=inputTn1   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=inputTn2   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem2

    #pragma HLS INTERFACE s_axilite port=inputTn1   bundle=control
    #pragma HLS INTERFACE s_axilite port=inputTn2   bundle=control
    #pragma HLS INTERFACE s_axilite port=outputTn   bundle=control

    #pragma HLS INTERFACE s_axilite port=dimA0      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimA1      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimA2      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimA3      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimB0      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimB1      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimB2      bundle=control
    #pragma HLS INTERFACE s_axilite port=dimB3      bundle=control

    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    concat2(
            inputTn1,
            inputTn2,
            outputTn,
            dimA0,
            dimA1,
            dimA2,
            dimA3,
            dimB0,
            dimB1,
            dimB2,
            dimB3);

}
}
