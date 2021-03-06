//
// Created by saleh on 8/30/18.
//

#pragma once

#include <build_config.h>
#include "../inc/TensorF.h"

#ifdef USE_CUDA
#include "../inc/cuda_imp/CudaTensorF.h"
#endif

#ifdef USE_OCL
#include "../inc/ocl_imp/OclTensorF.h"
#endif

#include "cnpy.h"
#include <vector>
#include <string>
using namespace std;

#ifndef USE_OCL
    struct cl_context{

    };
    struct cl_command_queue{

    };
#endif

class WeightsLoader {
public:
    WeightsLoader(vector<PLATFORMS> neededPlatforms);
#ifdef USE_OCL
    void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList, cl::Context *oclContex, cl::CommandQueue *oclQueue);
#else
    void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList) ;
#endif

    TensorF* AccessWeights(PLATFORMS platform, string name);

private:
    
    int ResolveMemoryBank(PLATFORMS platform, string name);
    int _ResolveMemoryBankOclXilinx(string name);
    string _ResolveTensorTagOclXilinx(string name);

    map<string,int> strToIndexMap;
    vector<cnpy::NpyArray> _cnpyBuff;
    TensorF** weightsCPU;
    TensorF** weightsCUDA;
    TensorF** weightsOCL;

    bool _isUsedCPU  = false;
    bool _isUsedCUDA = false;
    bool _isUsedOCL  = false; 

};
