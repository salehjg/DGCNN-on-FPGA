/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "PaddingCpu.h"
#include "Utility.h"
#include "Conv2D.h"
#include "Conv2Helper.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using namespace std;
using namespace ConfigTaskConv2;

extern "C"
void task_conv2_1x1_direct(
    MemoryPackK_t const a[],
    MemoryPackM_t const b[],
    MemoryPackM_t const e[],
    MemoryPackM_t c[],
    const unsigned size_n, 
    const unsigned size_k,
    const unsigned size_m);

int main(int argc, char **argv) {

    unsigned conv_b = 1;
    unsigned conv_n = 128;
    unsigned conv_k = 20;
    unsigned conv_din = 6;
    unsigned conv_dout = 128;

    if (argc < 6 || argc > 6) {
        cout << "Usage: ./TestSimulation Conv_B Conv_N Conv_K Conv_Din Conv_Dout" << endl;
        cout << "Running with the default parameters(B,N,K,Din,Dout)=("<< 
                conv_b << "," <<
                conv_n << "," <<
                conv_k << "," <<  
                conv_din << "," <<
                conv_dout << "," <<
                ")"<<endl;
    }else{
        conv_b = stoul(argv[1]);
        conv_n = stoul(argv[2]);
        conv_k = stoul(argv[3]);
        conv_din = stoul(argv[4]);
        conv_dout = stoul(argv[5]);
    }

    cout << "Convolution Shapes: Data(BxNxKxD1), Weight(D1xD2) :   " <<
            conv_b << "x" << conv_n << "x" << conv_k << "x" << conv_din << "  " <<
            conv_din << "x" << conv_dout << endl;

    const unsigned size_n = conv_b*conv_n*conv_k;
    const unsigned size_k = conv_din;
    const unsigned size_m = conv_dout;

    cout << "[N, K, M] = [" << size_n << ", " << size_k << ", " << size_m << "]" << endl;

    /*if (size_k % kMemoryWidthK != 0) {
    cerr << "K must be divisable by memory width." << endl;
    return 1;
    }*/

    if (size_m % kMemoryWidthM != 0) {
        cerr << "M must be divisable by memory width." << endl;
        return 1;
    }
    if (size_n % kOuterTileSizeN != 0) {
        cerr << "N must be divisable by the outer tile size in N."
        << endl;
        return 1;
    }
    if (size_m % kOuterTileSizeM != 0) {
        cerr << "M must be divisable by the outer tile size in M" << endl;
        return 1;
    }

    const auto size_k_padded = DivCeil<unsigned>(size_k, kTransposeWidth)*kTransposeWidth;

    vector<CONFIG_DTYPE> a(size_n * size_k); //input
    vector<CONFIG_DTYPE> b(size_k * size_m); //weight
    vector<CONFIG_DTYPE> e(size_m); //bias
    vector<CONFIG_DTYPE> cReference(size_n * size_m, 0);
    vector<CONFIG_DTYPE> cReferenceConv2(size_n * size_m, 0);

    default_random_engine rng(kSeed);
    typename conditional<
        is_integral<CONFIG_DTYPE>::value, uniform_int_distribution<unsigned long>,
        uniform_real_distribution<double>>::type dist(1, 10);

    for_each(a.begin(), a.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    for_each(b.begin(), b.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    for_each(e.begin(), e.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    vector<CONFIG_DTYPE> aPadded(size_n * size_k_padded);
    PadTensor<CONFIG_DTYPE>(a, aPadded, size_n, size_k, DivCeil<unsigned>(size_k, kTransposeWidth)*kTransposeWidth);

    const auto aKernel = Pack<kMemoryWidthA, float>(aPadded);
    const auto bKernel = Pack<kMemoryWidthM, float>(b);
    const auto eKernel = Pack<kMemoryWidthM, float>(e);
    auto cKernel = Pack<kMemoryWidthM, float>(cReference);

    ReferenceImplementation(a.data(), b.data(), cReference.data(), size_n, size_k, size_m);
    Conv2Kernel1x1CPU<CONFIG_DTYPE >(
        a.data(), b.data(), e.data(), cReferenceConv2.data(), 
        conv_b, conv_n, conv_k, conv_din, conv_dout);

    cout << "Running simulation...\n" << flush;

    task_conv2_1x1_direct(
        aKernel.data(), bKernel.data(), 
        eKernel.data(), cKernel.data(),
        size_n, size_k, size_m);
    cout << "Verifying results...\n" << flush;

    const auto cTest = Unpack<kMemoryWidthM, float>(cKernel);

    for (unsigned i = 0; i < size_n; ++i) {
        for (unsigned j = 0; j < size_m; ++j) {
            const auto testVal = make_signed<CONFIG_DTYPE>(cTest[i * size_m + j]);
            const auto refVal = make_signed<CONFIG_DTYPE>(cReference[i * size_m + j]);
            const auto refValConv2 = make_signed<CONFIG_DTYPE>(cReferenceConv2[i * size_m + j]);
            //const CONFIG_DTYPE diff = abs(testVal - refVal);
            const CONFIG_DTYPE diff2 = abs(testVal - refValConv2);
            /*if (diff > static_cast<CONFIG_DTYPE>(1e-3)) {
            cerr << "Mismatch detected(Kernel vs. CPU MM) at (" << i << ", " << j
            << "): " << testVal << " vs. " << refVal << "\n";
            return 1;
            }*/
            if (diff2 /*/ refValConv2*/ > static_cast<CONFIG_DTYPE>(1e-2)) {
                cerr << "Mismatch detected(Kernel vs. CPU Conv2) at (" << i << ", " << j
                    << "): " << testVal << " vs. " << refValConv2 << "\n";
                return 2;
            }
        }
    }
    //cout << "Matrix-matrix multiplication successfully verified.\n";
    cout << "Conv2D 1x1 successfully verified.\n";

    return 0;
}
