//
// Created by saleh on 8/23/18.
//

#include <cassert>
#include "../inc/TensorI.h"
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

TensorI::TensorI(std::vector<unsigned int> shape) {
    Init(shape);
}

TensorI::TensorI(std::vector<unsigned int> shape, int *buff) {
    Init(shape,buff);
}

void TensorI::Init(std::vector<unsigned int> shape) {
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

void TensorI::Init(std::vector<unsigned int> shape, int* buff){
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

std::vector<unsigned int> TensorI::getShape(){
    return shape;
}

int TensorI::getRank() {
    return rank;
}


void TensorI::ExpandDims(int axis) {
    assert((axis>=0 && axis<=getRank()) || axis==-1);
    if(axis==-1) axis=(int)shape.size();
    shape.insert(shape.begin()+axis,1);
    this->rank++;
}

void TensorI::ExpandDimZero(){
    ExpandDims(0);
}

void TensorI::SqueezeDims() {
    std::vector<unsigned int> shapeNew;

    for (int i = 0; i < shape.size(); i++) {
        if(shape[i]!=1) shapeNew.push_back(shape[i]);
    }
    shape = shapeNew;
    rank = (int)shape.size();
}

void TensorI::SqueezeDimZero(){
    if(shape[0]==1){
        shape.erase(shape.begin());
        rank--;
    }
}

void TensorI::Reshape(std::vector<unsigned int> newShape){
    unsigned long len = 1;
    for (int i = 0; i < newShape.size(); i++) {
        len = len * newShape[i];
    }
    assert(len==getLength());
    shape = newShape;
    rank = (int)shape.size();
}

PLATFORMS TensorI::getPlatform(){
    return platform;
}

unsigned long TensorI::getLength() {
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

unsigned long TensorI::getLengthBytes() {
    if(initialized) {
        unsigned long len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(int);
    }else{
        return 0;
    }
}

unsigned long TensorI::getLengthPadded(int vectorWords){
    assert(vectorWords>0);
    unsigned long len = getLength();
    int pad = len % vectorWords;
    pad = vectorWords - pad;
    return len + pad;
}

unsigned long TensorI::getLengthBytesPadded(int vectorWords){
    assert(vectorWords>0);
    return getLengthPadded(vectorWords) * sizeof(int);
}

unsigned long TensorI::getVectorCountPadded(int vectorWords){
    assert(vectorWords>0);
    unsigned long len = getLengthPadded(vectorWords);
    return len / (unsigned long)vectorWords;
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