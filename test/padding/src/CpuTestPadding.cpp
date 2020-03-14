#include "PaddingCpu.h"
#include "Utility.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using namespace std;

extern "C"
void task_pad_last_dim(
    const MemoryPack_t *inputTn,
    MemoryPack_t *outputTn,
    const int reverseSwitch,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm);

int main(int argc, char **argv) {

    const unsigned int dim0 = 1024;
    const unsigned int dim1 = 6;
    const unsigned int dim1Padded = 16;
    const unsigned int vecSize = 16;

    unsigned int lenInput  = dim0*dim1 + (vecSize - dim0*dim1%vecSize);
    unsigned int lenOutput = dim0*dim1Padded + (vecSize - dim0*dim1Padded%vecSize);

    const unsigned int gcd = __gcd(dim1, vecSize);
    const unsigned int lcm = (dim1*vecSize)/(gcd);

    cout<<"InputShape: "<<dim0<<"x"<<dim1<<endl;
    cout<<"Padding Dimension: "<<"1"<<endl;
    cout<<"The Dimension Before Padding: "<<dim1<<endl;
    cout<<"The Dimension After Padding: "<<dim1Padded<<endl;
    cout<<"Vector Size: "<<vecSize<<endl;
    cout<<"GCD(dim1,vecSize): "<<gcd<<endl;
    cout<<"LCM(dim1,vecSize): "<<lcm<<endl;

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    const auto deviceInputTn = Pack<vecSize>(hostInputTn);
    auto deviceOutputTn = Pack<vecSize>(hostGold);

    task_pad_last_dim(deviceInputTn.data(), deviceOutputTn.data(), 0, dim0, dim1, dim1Padded, lcm);
    PadTensor<CONFIG_DTYPE>(hostInputTn, hostGold, dim0, dim1, dim1Padded);

    const auto hostOutputTn = Unpack<vecSize>(deviceOutputTn);
    bool rslt = true;

    for(int d0=0; d0<dim0; d0++){
        for(int d1=0; d1<dim1Padded; d1++){
            unsigned int indx = d0*dim1Padded+d1;
            CONFIG_DTYPE diff = (hostOutputTn[indx] - hostGold[indx]);
            diff = diff>=0? diff: -1*diff;
            if(diff>1e-03){
                std::cout<<"Mismatch at d0: "<< d0 <<", d1: "<< d1 << std::endl;
                std::cout<<"Value: "<< hostOutputTn[indx] << std::endl;
                std::cout<<"Gold: "<< hostGold[indx] << std::endl;
                rslt = false;
            }
        }
    }

    std::cout<<std::endl;

    if(rslt){
        std::cout<<"Sub-vector padding successfully verified."<<std::endl;
    }

    return (rslt)? 0 : 1;
}
