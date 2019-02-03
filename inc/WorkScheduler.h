//
// Created by saleh on 8/27/18.
//

#ifndef DEEPPOINTV1_WORKSCHEDULER_H
#define DEEPPOINTV1_WORKSCHEDULER_H

#include <build_config.h>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
#include <thread>
class WorkScheduler {
public:
#ifdef USE_CUDA
    cudaStream_t cudaStream;
#endif

private:

};


#endif //DEEPPOINTV1_WORKSCHEDULER_H
