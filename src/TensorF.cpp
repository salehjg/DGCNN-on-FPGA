//
// Created by saleh on 8/23/18.
//

#include <cassert>
#include "../inc/TensorF.h"
#include <vector>
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
TensorF::TensorF() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorF::TensorF(std::vector<unsigned int> shape) {
    Init(shape);
}

TensorF::TensorF(std::vector<unsigned int> shape, float *buff) {
    Init(shape,buff);
}

void TensorF::Init(std::vector<unsigned int> shape) {
    if(initialized){
        std::cout<<"--- TensorF: buffer deleted.\n";
        delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)(shape.size());
    initialized = true;
    _buff = new float[getLength()];
    platform = PLATFORMS::CPU;
}

void TensorF::Init(std::vector<unsigned int> shape, float* buff){
    if(initialized){
        std::cout<<"--- TensorF: buffer deleted.\n";
        delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)(shape.size());
    _buff = buff;
    initialized = true;
    platform = PLATFORMS::CPU;
}

std::vector<unsigned int> TensorF::getShape(){
    return shape;
}

int TensorF::getRank() {
    return rank;
}

void TensorF::ExpandDims(int axis) {
    assert((axis>=0 && axis<=getRank()) || axis==-1);
    if(axis==-1) axis=(int)shape.size();
    shape.insert(shape.begin()+axis,1);
    this->rank++;
}

void TensorF::ExpandDimZero(){
    ExpandDims(0);
}

void TensorF::SqueezeDimZero(){
    if(shape[0]==1){
        shape.erase(shape.begin());
        rank--;
    }
}

void TensorF::SqueezeDims() {
    std::vector<unsigned int> shapeNew;

    for (int i = 0; i < shape.size(); i++) {
        if(shape[i]!=1) shapeNew.push_back(shape[i]);
    }
    shape = shapeNew;
    rank = (int)shape.size();
}

void TensorF::Reshape(std::vector<unsigned int> newShape){
    unsigned long len = 1;
    for (int i = 0; i < newShape.size(); i++) {
        len = len * newShape[i];
    }
    assert(len==getLength());
    shape = newShape;
    rank = (int)shape.size();
}

PLATFORMS TensorF::getPlatform(){
    return platform;
}

unsigned long TensorF::getLength() {
    ///TODO: Change the if statement, because if we init a Tensor instance with an external buffer,
    /// on destruction, the destructor will delete that external buffer which isn't right
    /// so 'initialized' should represent the presence of internal allocated buffer. NOT an external one!
    if(initialized) {
        unsigned long len = 1;
        for (int i = 0; i < shape.size(); i++) {
            len = len * shape[i];
        }
        return len;
    }else{
        return 0;
    }
}

unsigned long TensorF::getLengthBytes() {
    if(initialized) {
        unsigned long len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(float);
    }else{
        return 0;
    }
}

// https://stackoverflow.com/questions/9331561/why-does-my-classs-destructor-get-called-when-i-add-instances-to-a-vector
TensorF::~TensorF() {
    if(platform == PLATFORMS::CPU){
        if(initialized){
            //std::cout<<"--- TensorF: destructed.\n";
            delete(_buff);
        }
    } /*else if(platform == PLATFORMS::GPU_CUDA){

#ifdef USE_CUDA
        cudaError_t cuda_stat;
        if(initialized){
            //std::cout<<"--- CudaTensorF: buffer deleted.\n";
            cuda_stat = cudaFree(_buff);
            assert(cuda_stat==cudaSuccess);
        }
        cudaCheckErrors("~TensorF-CUDA@TensorF: ERR04");
#endif
    }else if (platform == PLATFORMS::GPU_OCL){

    }*/

}