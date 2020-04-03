#include <cassert>
#include <iostream>
#include <limits>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/TreeReduce.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskMatMul;

constexpr unsigned CONFIG_MAX_N = 1024;
constexpr unsigned CONFIG_MAX_K = 1024;
constexpr unsigned CONFIG_MAX_M = 1024;

// Architecture adopted from https://github.com/spcl/hls_tutorial_examples
template<unsigned D>
void MatmulReorderedVectorized_V1(
    const CONFIG_DTYPE* A,
    const MemoryPackF_t* B,
    MemoryPackF_t *C,
    const unsigned sizeBatch,
    const unsigned sizeN,
    const unsigned sizeK,
    const unsigned sizeM){

    // MatA's  shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeK] = [Batch, Height, Width]; Row-major
    // MatB's  shape = [dim0, dim1, dim2] = [batchSize, sizeK, sizeM] = [Batch, Height, Width]; Row-major
    // MatC=AB shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeM] = [Batch, Height, Width]; Row-major
    
    const unsigned lastDimPaddedA = MakeDivisible<unsigned>(sizeK, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedB = MakeDivisible<unsigned>(sizeM, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedC = lastDimPaddedB;


    const unsigned vecsPerSliceA = lastDimPaddedA/CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceB = lastDimPaddedB/CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceC = lastDimPaddedC/CONFIG_M_AXI_WIDTH;

    const unsigned boundLoopN = DivCeil<unsigned>(sizeN, D);

    LoopBatch:
    for(unsigned batch=0; batch<sizeBatch; batch++) {
        LoopN:
        for (unsigned n = 0; n < boundLoopN; n++) {
            MemoryPackF_t acc[D][CONFIG_MAX_M / CONFIG_M_AXI_WIDTH];
            #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

            LoopK:
            for (unsigned k = 0; k < sizeK; k++) {
                const unsigned kVecIndex = k / CONFIG_M_AXI_WIDTH;
                CONFIG_DTYPE a_buffer[D];
                LoopReadA:
                for (unsigned nd = 0; (nd<D)&&((n*D+nd)<sizeN); nd++) {
                    #pragma HLS PIPELINE II=1
                    // matrix A is padded on the last dimension but it is accessed by axi-32bits.
                    const unsigned indxS1 = (batch)*sizeN*lastDimPaddedA + (n*D+nd)*lastDimPaddedA + (k);
                    a_buffer[nd] = A[indxS1];
                }
                LoopM:
                for (unsigned m = 0; m < vecsPerSliceB; m++) {
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS2 = (batch)*sizeK*vecsPerSliceB + k*vecsPerSliceB + m;
                    const auto b_val = B[indxS2];
                    LoopUnrolled:
                    for (unsigned nd = 0; (nd<D)&&((n*D+nd)<sizeN); ++nd) {
                        #pragma HLS UNROLL
                        const auto prev = (k > 0) ? acc[nd][m] : MemoryPackF_t(0.);
                        acc[nd][m] = prev + a_buffer[nd] * b_val;
                        #pragma HLS DEPENDENCE variable=acc inter false
                    }
                }
            }
            LoopWriteD:
            for (unsigned nd = 0; (nd<D)&&((n*D+nd)<sizeN); ++nd) {
                LoopWriteM:
                for (unsigned m = 0; m < vecsPerSliceB; ++m) {
                    #pragma HLS LOOP_FLATTEN
                    #pragma HLS PIPELINE II=1
                    const unsigned indxD = (batch)*sizeN*vecsPerSliceC + (n*D+nd)*vecsPerSliceC + m;
                    C[indxD] = acc[nd][m];
                }
            }
        }
    }
}

extern "C"{
void task_matmul(
        const CONFIG_DTYPE *inputTn1,
        const MemoryPackF_t *inputTn2,
        MemoryPackF_t *outputTn,
        const unsigned sizeBatch,
        const unsigned sizeN,
        const unsigned sizeK,
        const unsigned sizeM){
#pragma HLS INTERFACE m_axi port=inputTn1 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=inputTn2 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=inputTn1 bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2 bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=sizeBatch bundle=control
#pragma HLS INTERFACE s_axilite port=sizeN bundle=control
#pragma HLS INTERFACE s_axilite port=sizeK bundle=control
#pragma HLS INTERFACE s_axilite port=sizeM bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    MatmulReorderedVectorized_V1<4>(
        inputTn1, inputTn2, outputTn, 
        sizeBatch, sizeN, sizeK, sizeM);

}
}
