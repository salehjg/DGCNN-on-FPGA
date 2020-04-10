//
// Created by saleh on 8/23/18.
//

#include <cassert>
#include "TensorF.h"
#include "ocl_imp/xilinx/AxiHelper.h"
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

void TensorF::ExpandDimZero(){
    // DOES NOT AFFECT DATA PADDING PATTERNS
    ExpandDims(0);
}

void TensorF::SqueezeDimZero(){
    // DOES NOT AFFECT DATA PADDING PATTERNS
    if(shape[0]==1){
        shape.erase(shape.begin());
        rank--;
    }
}

void TensorF::SqueezeDims() {
    // DOES NOT AFFECT DATA PADDING PATTERNS

    //Just making sure that padded last dim policy wont cause any problems.
    if(platform==PLATFORMS::GPU_OCL) assert(shape[shape.size()-1]!=1);

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

    //Just making sure that padded last dim policy wont cause any problems.
    if(platform==PLATFORMS::GPU_OCL) assert(newShape[newShape.size()-1]==shape[shape.size()-1]);

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

unsigned long TensorF::getLengthPadded(int vectorWords){
    if(initialized) {
        unsigned long len = 1;
        std::vector<unsigned int> paddedShape = PadShape(shape, vectorWords);
        for (int i = 0; i < paddedShape.size(); i++) {
            len = len * paddedShape[i];
        }

        return len;
    }else{
        return 0;
    }
}

std::vector<unsigned int> TensorF::PadShape(std::vector<unsigned int> &actualShape, int vectorWords){
    std::vector<unsigned int> paddedShape = actualShape;
    // always pad the last dimension.
    unsigned int lastDim = paddedShape[paddedShape.size()-1];
    paddedShape[paddedShape.size()-1] = MakeDivisible<unsigned int>(lastDim, vectorWords);

    return paddedShape;
}

unsigned long TensorF::getLengthBytesPadded(int vectorWords){
    assert(vectorWords>0);
    return getLengthPadded(vectorWords) * sizeof(float);
}

unsigned long TensorF::getVectorCountPadded(int vectorWords){
    assert(vectorWords>0);
    unsigned long len = getLengthPadded(vectorWords);
    return len / (unsigned long)vectorWords;
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