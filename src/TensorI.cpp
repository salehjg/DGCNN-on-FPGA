//
// Created by saleh on 8/23/18.
//

#include <cassert>
#include "TensorI.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include <iostream>
#ifdef USE_CUDA
#include <cuda_runtime.h>

#ifndef cudaCheckErrors
#define cudaCheckErrors(msg) \
        do { \
            cudaError_t __err = cudaGetLastError(); \
            if (__err != cudaSuccess) { \
                fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
                fprintf(stderr, "*** FAILED - ABORTING\n"); \
            } \
        } while (0)
#endif
#endif

TensorI::TensorI() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorI::TensorI(std::vector<unsigned> shape) {
    Init(shape);
}

TensorI::TensorI(std::vector<unsigned> shape, int *buff) {
    Init(shape,buff);
}

void TensorI::Init(std::vector<unsigned> shape) {
    if(initialized){
        std::cout<<"--- TensorI: buffer deleted.\n";
        delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    initialized = true;
    _buff = new int[getLength()];
    platform = PLATFORMS::CPU;
}

void TensorI::Init(std::vector<unsigned> shape, int* buff){
    if(initialized){
        std::cout<<"--- TensorI: buffer deleted.\n";
        delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    _buff = buff;
    initialized = true;
    platform = PLATFORMS::CPU;
}

std::vector<unsigned> TensorI::getShape(){
    return shape;
}

int TensorI::getRank() {
    return rank;
}


void TensorI::ExpandDims(int axis) {
    // DOES NOT AFFECT DATA PADDING PATTERNS
    assert((axis>=0 && axis<=getRank()) || axis==-1);
    if((axis==-1||axis==shape.size()) && (platform==PLATFORMS::GPU_OCL) ){
        //Just making sure that padded last dim policy wont cause any problems.
        assert(shape[shape.size()-1]==1);
    }

    if(axis==-1) axis=(int)shape.size();
    shape.insert(shape.begin()+axis,1);
    this->rank++;
}

void TensorI::ExpandDimZero(){
    // DOES NOT AFFECT DATA PADDING PATTERNS
    ExpandDims(0);
}

void TensorI::SqueezeDims() {
    // DOES NOT AFFECT DATA PADDING PATTERNS

    //Just making sure that padded last dim policy wont cause any problems.
    if(platform==PLATFORMS::GPU_OCL) assert(shape[shape.size()-1]!=1);

    std::vector<unsigned> shapeNew;

    for (int i = 0; i < shape.size(); i++) {
        if(shape[i]!=1) shapeNew.push_back(shape[i]);
    }
    shape = shapeNew;
    rank = (int)shape.size();
}

void TensorI::SqueezeDimZero(){
    // DOES NOT AFFECT DATA PADDING PATTERNS
    if(shape[0]==1){
        shape.erase(shape.begin());
        rank--;
    }
}

void TensorI::Reshape(std::vector<unsigned> newShape){
    unsigned len = 1;
    for (int i = 0; i < newShape.size(); i++) {
        len = len * newShape[i];
    }
    assert(len==getLength());

    //Just making sure that padded last dim policy wont cause any problems.
    if(platform==PLATFORMS::GPU_OCL) assert(newShape[newShape.size()-1]==shape[shape.size()-1]);

    shape = newShape;
    rank = (int)shape.size();
}

PLATFORMS TensorI::getPlatform(){
    return platform;
}

unsigned TensorI::getLength() {
    if(initialized) {
        unsigned len = 1;
        for (int i = 0; i < shape.size(); i++) {
            len = len * shape[i];
        }
        return len;
    }else{
        return 0;
    }
}

unsigned TensorI::getLengthBytes() {
    if(initialized) {
        unsigned len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(int);
    }else{
        return 0;
    }
}

unsigned TensorI::getLengthPadded(int vectorWords){
    if(initialized) {
        unsigned len = 1;
        std::vector<unsigned> paddedShape = PadShape(shape, vectorWords);
        for (int i = 0; i < paddedShape.size(); i++) {
            len = len * paddedShape[i];
        }

        return len;
    }else{
        return 0;
    }
}

std::vector<unsigned> TensorI::PadShape(std::vector<unsigned> &actualShape, int vectorWords){
    std::vector<unsigned> paddedShape = actualShape;
    // always pad the last dimension.
    unsigned lastDim = paddedShape[paddedShape.size()-1];
    paddedShape[paddedShape.size()-1] = MakeDivisible<unsigned>(lastDim, vectorWords);

    return paddedShape;
}

unsigned TensorI::getLengthBytesPadded(int vectorWords){
    assert(vectorWords>0);
    return getLengthPadded(vectorWords) * sizeof(int);
}

unsigned TensorI::getVectorCountPadded(int vectorWords){
    assert(vectorWords>0);
    unsigned len = getLengthPadded(vectorWords);
    return len / (unsigned)vectorWords;
}

TensorI::~TensorI() {
    if(platform == PLATFORMS::CPU){
        if(initialized){
            //std::cout<<"--- TensorI: destructed.\n";
            delete(_buff);
        }
    }/*else if(platform == PLATFORMS::GPU_CUDA){

#ifdef USE_CUDA
        cudaError_t cuda_stat;
        if(initialized){
            //std::cout<<"--- CudaTensorI: buffer deleted.\n";
            cuda_stat = cudaFree(_buff);
            assert(cuda_stat==cudaSuccess);
        }
        cudaCheckErrors("~CudaTensorI@CudaTensorI: ERR04");
#endif
    }
    */
        
}