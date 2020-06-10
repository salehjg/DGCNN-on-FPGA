//
// Created by saleh on 8/22/18.
//

#include "build_config.h"
#include "../inc/PlatformSelector.h"
#include "../inc/cpu_imp/CpuImplementation.h"

#ifdef USE_CUDA
#include <cuda_imp/CudaMemHelper.h>
#include <cuda_imp/common.h>
#include "../inc/cuda_imp/CudaImplementation.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include "../inc/cuda_imp/CudaTensorI.h"
#endif

#ifdef USE_OCL
#include "ocl_imp/xilinx/XilinxImplementation.h"
#include "../inc/ocl_imp/OclTensorF.h"
#include "../inc/ocl_imp/OclTensorI.h"
#endif

PlatformSelector::PlatformSelector(PLATFORMS defaultPlatform, vector<PLATFORMS> neededPlatforms,bool loadWeights) {
    this->defaultPlatform = defaultPlatform;

    for(std::vector<PLATFORMS>::iterator it = neededPlatforms.begin(); it != neededPlatforms.end(); ++it) {
        switch(*it){
            case PLATFORMS::CPU : {
                cpuPlatformClass = new CpuImplementation();
                break;
            }
#ifdef USE_CUDA
            case PLATFORMS::GPU_CUDA : {
                cudaPlatformClass = new CudaImplementation(11);
                break;
            }
#endif
#ifdef USE_OCL
            case PLATFORMS::GPU_OCL : {
                openclPlatformClass = new XilinxImplementation(11);
                break;
            }
#endif
        }
    }

    weightsLoader = new WeightsLoader(neededPlatforms);
    if(!loadWeights) SPDLOG_LOGGER_WARN(logger,"Weights are not loaded to the device memory");
#ifdef USE_OCL
    if(loadWeights){
        std::string wDir = globalArgDataPath; wDir.append("/weights/");
        std::string wFileList = globalArgDataPath; wFileList.append("/weights/filelist.txt");
        SPDLOG_LOGGER_TRACE(logger,"Weights Dir: {}", wDir);
        SPDLOG_LOGGER_TRACE(logger,"Weights File List Path: {}", wFileList);
        weightsLoader->LoadFromDisk(wDir.c_str() ,
                                    wFileList.c_str() ,
                                    openclPlatformClass->getContext(),
                                    openclPlatformClass->getQueue());
    }
#else
    if(loadWeights){
        weightsLoader->LoadFromDisk(REPO_DIR "/data/weights/",
                                    REPO_DIR "/data/weights/filelist.txt" );
    }
#endif
}

PlatformSelector::~PlatformSelector(){
    SPDLOG_LOGGER_TRACE(logger,"~PlatformSelector");
    delete(weightsLoader);
    delete(cpuPlatformClass);
    delete(openclPlatformClass);
}

TensorF* PlatformSelector::CrossThePlatform(TensorF *srcTn, PLATFORMS platform) {
    assert(srcTn->getShape()[srcTn->getRank()-1]!=1); //Just to make sure nothing will go wrong because of padding
    switch(srcTn->getPlatform()){
        case PLATFORMS::CPU :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    return srcTn;
                    break;
                }
#ifdef USE_CUDA
                case PLATFORMS::GPU_CUDA :{
                    CudaTensorF *rsltTn = new CudaTensorF();
                    rsltTn->InitWithHostData(srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
#ifdef USE_OCL
                case PLATFORMS::GPU_OCL :{
                    OclTensorF *rsltTn = new OclTensorF();
                    rsltTn->InitWithHostData(openclPlatformClass->getContext(), openclPlatformClass->getQueue(), srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
            }
            break;

        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    CudaTensorF * srcTensor = (CudaTensorF*)srcTn;
                    TensorF* rsltTn = srcTensor->TransferToHost();
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    return srcTn;
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    throw "Not Implemented.";
                    break;
                }
            }
            break;

        }
#endif

#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    OclTensorF* srcTensor = (OclTensorF*)srcTn;
                    TensorF* rsltTn = srcTensor->TransferToHost(openclPlatformClass->getQueue());
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    throw "Not Implemented.";
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    return srcTn;
                    break;
                }
            }
            break;
        }
#endif
    }
}


TensorI* PlatformSelector::CrossThePlatform(TensorI *srcTn, PLATFORMS platform) {
    assert(srcTn->getShape()[srcTn->getRank()-1]!=1); //Just to make sure nothing will go wrong because of padding
    switch(srcTn->getPlatform()){
        case PLATFORMS::CPU :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    return srcTn;
                    break;
                }
#ifdef USE_CUDA
                case PLATFORMS::GPU_CUDA :{
                    CudaTensorI *rsltTn = new CudaTensorI();
                    rsltTn->InitWithHostData(srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
#ifdef USE_OCL
                case PLATFORMS::GPU_OCL :{
                    OclTensorI *rsltTn = new OclTensorI();
                    rsltTn->InitWithHostData(openclPlatformClass->getContext(), openclPlatformClass->getQueue(), srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                }
#endif
            }
            break;

        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    CudaTensorI * srcTensor = (CudaTensorI*)srcTn;
                    return srcTensor->TransferToHost();
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    return srcTn;
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    throw "Not Implemented.";
                    break;
                }
            }
            break;

        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    OclTensorI* srcTensor = (OclTensorI*)srcTn;
                    TensorI* rsltTn = srcTensor->TransferToHost(openclPlatformClass->getQueue());
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    throw "Not Implemented.";
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    return srcTn;
                    break;
                }
            }
            break;
        }
#endif
    }
}

TensorF* PlatformSelector::Transpose(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat){
    TensorF *__batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Transpose(scheduler, __batchedMat);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Transpose(scheduler,__batchedMat);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Transpose(scheduler,__batchedMat);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2){
    TensorF* __batchedMat1 = CrossThePlatform(batchedMat1, platform);
    TensorF* __batchedMat2 = CrossThePlatform(batchedMat2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Square(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat){
    TensorF* __batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceSum(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceSum(scheduler, __inputTn, over_axis0, over_axis1, over_axis2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->ReduceSum(scheduler, __inputTn,over_axis0, over_axis1, over_axis2);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceSum(scheduler, __inputTn, over_axis0, over_axis1, over_axis2);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceSum4D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceSum4D(scheduler, __inputTn, over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->ReduceSum4D(scheduler, __inputTn,over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceSum4D(scheduler, __inputTn, over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Mean(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Variance(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Sqrt(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Concat2(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceMax(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, int reductionDim){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceMax(scheduler, __inputTn, reductionDim);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->ReduceMax(scheduler, __inputTn,reductionDim);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceMax(scheduler, __inputTn, reductionDim);
            break;
        }
#endif
    }
    return nullptr;
}


TensorI* PlatformSelector::TopK(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    TensorF* __batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Gather(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    TensorI* __indices = CrossThePlatform(indices, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#endif
    }
    return nullptr;
}


TensorF* PlatformSelector::Conv2D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    TensorF* __weights = CrossThePlatform(weights, platform);
    TensorF* __biases = CrossThePlatform(biases, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReLU(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Tile(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            break;
        }
#endif
    }
    return nullptr;
}

void PlatformSelector::DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn,npy_dir);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            TensorF* __inputTn2 = CrossThePlatform(inputTn, PLATFORMS::CPU);
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn2,npy_dir);
            break;
        }
#endif
    }
    return;
}

void PlatformSelector::DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir){
    TensorI* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn,npy_dir);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            TensorI* __inputTn2 = CrossThePlatform(inputTn, PLATFORMS::CPU);
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn2,npy_dir);
            break;
        }
#endif
    }
    return;
}

bool PlatformSelector::CompareTensors(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->CompareTensors(scheduler, __inputTn1,__inputTn2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
#endif
    }
    return false;
}

bool PlatformSelector::CompareTensorsInteger(PLATFORMS platform, WorkScheduler scheduler, TensorI *inputTn1, TensorI *inputTn2) {
    TensorI* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorI* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->CompareTensorsInteger(scheduler, __inputTn1,__inputTn2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
#endif
    }
    return false;
}

TensorF* PlatformSelector::PadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimPadded){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->PadLastDim(scheduler, __inputTn, lastDimPadded);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->PadLastDim(scheduler, __inputTn, lastDimPadded);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::UnpadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimUnpadded){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->UnpadLastDim(scheduler, __inputTn, lastDimUnpadded);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->UnpadLastDim(scheduler, __inputTn, lastDimUnpadded);
            break;
        }
#endif
    }
    return nullptr;
}

void PlatformSelector::DumpImplementationSpecificLogs(PLATFORMS platform){
    switch(platform){
        case PLATFORMS::CPU :{
            assert(0);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            assert(0);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            openclPlatformClass->DumpDataMoverLaunchLogs();
            break;
        }
#endif
    }
}