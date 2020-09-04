#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>
#include "Utility.h"
#include "AxiHelper.h"

using namespace std;
using namespace ConfigTaskTopK;

extern "C"
void task_topk(
        const MemoryPackF_t *inputTn,
        //MemoryPackI_t *indicesSplitedTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice);

void CpuGoldSort(
    const CONFIG_DTYPE *inputBuff,
    CONFIG_DTYPE *outputBuff,
    const unsigned dim0,
    const unsigned dim1){

    CONFIG_DTYPE tmp_array[dim1]; 

    for(unsigned b=0; b<dim0; b++){
        for(unsigned i=0; i<dim1; i++){
            tmp_array[i] = inputBuff[b*dim1+i];
        }

        sort(tmp_array, tmp_array+dim1);

        for(unsigned i=0; i<dim1; i++){
            outputBuff[b*dim1+i] = tmp_array[i];
        }
    }
}

template<unsigned vecSize>
int TestTopk(
    const string& testName,
    const unsigned dim0, 
    const unsigned dim1){

    assert(dim1%vecSize==0);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;
    cout<<"InputShape: "<<dim0<<"x"<<dim1<<endl;
    cout<<"OutputShape: "<<dim0<<"x"<<dim1<<endl;

    const unsigned lenInput = dim0*dim1;
    const unsigned lenOutput = dim0*dim1;
    const unsigned vecsPerSlice = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerOutputSlice = vecsPerSlice;

    // UDT outputs are padded in the last dimension to be divisible by maxi width
    const unsigned lenOutputUdt = dim0*(vecsPerOutputSlice*CONFIG_M_AXI_WIDTH);

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);
    std::vector<CONFIG_DTYPE> hostUDT(lenOutputUdt);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    /*
    for(unsigned i=0; i<lenInput; i++){
        hostInputTn[i] = (CONFIG_DTYPE)(lenInput - i-1);
    }*/

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTn);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    // The kernel writes the results in the output tensor with padding on the last dimension.
    task_topk(deviceInputTn.data(), deviceOutputTn.data(), dim0, dim1, 0, vecsPerSlice, 0);
    CpuGoldSort(hostInputTn.data(), hostGold.data(), dim0, dim1);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    bool rslt = true;

    for(unsigned b=0;b<dim0;b++){
        for(unsigned d1=0;d1<dim1;d1++){
            // The kernel writes the results in the output tensor with padding on the last dimension.
            unsigned indx = b*dim1+d1;
            float Cpu = hostGold[indx];
            float Udt = hostOutputTn[indx];

            if(Cpu!=Udt){
                printf("Sorted array mismatch, d0:%d, d1:%d, Gold:%f, UDT:%f\n",
                       b, d1,
                       Cpu,Udt);
                rslt=false;
            }
        }
    }

    std::cout<<std::endl;

    if(rslt){
        std::cout<<"Test \""<<testName<<"\" is successfully verified."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int main(int argc, char **argv) {
    int rslt0 = 0;
    rslt0 += TestTopk<16>("MergeSortDF_SoloSlice", 1, MaxSliceLen);
    return rslt0;
}
