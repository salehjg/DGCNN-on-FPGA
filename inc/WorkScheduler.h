//
// Created by saleh on 8/27/18.
//

#pragma once

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
