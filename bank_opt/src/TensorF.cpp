#include <cassert>
#include <vector>
#include <iostream>
#include "TensorF.h"
#include "Helper.h"

TensorF::TensorF() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorF::TensorF(std::vector<unsigned> shape) {
    Init(shape);
}

TensorF::TensorF(std::vector<unsigned> shape, float *buff) {
    Init(shape,buff);
}

void TensorF::Init(std::vector<unsigned> shape) {
    if(initialized){
        //delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)(shape.size());
    initialized = true;
    platform = PLATFORMS::CPU;
}

void TensorF::Init(std::vector<unsigned> shape, float* buff){
    if(initialized){
        //delete(_buff);
    }
    this->shape = shape;
    this->rank = (int)(shape.size());
    initialized = true;
    platform = PLATFORMS::CPU;
}

std::vector<unsigned> TensorF::getShape(){
    return shape;
}

int TensorF::getRank() {
    return rank;
}

void TensorF::ExpandDims(int axis) {
    // DOES NOT AFFECT DATA PADDING PATTERNS
    assert((axis>=0 && axis<=getRank()) || axis==-1);

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

    std::vector<unsigned> shapeNew;

    for (int i = 0; i < shape.size(); i++) {
        if(shape[i]!=1) shapeNew.push_back(shape[i]);
    }
    shape = shapeNew;
    rank = (int)shape.size();
}

void TensorF::Reshape(std::vector<unsigned> newShape){
    unsigned len = 1;
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

unsigned TensorF::getLength() {
    ///TODO: Change the if statement, because if we init a Tensor instance with an external buffer,
    /// on destruction, the destructor will delete that external buffer which isn't right
    /// so 'initialized' should represent the presence of internal allocated buffer. NOT an external one!
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

unsigned TensorF::getLengthBytes() {
    if(initialized) {
        unsigned len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(float);
    }else{
        return 0;
    }
}

unsigned TensorF::getLengthPadded(int vectorWords){
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

std::vector<unsigned> TensorF::PadShape(std::vector<unsigned> &actualShape, int vectorWords){
    std::vector<unsigned> paddedShape = actualShape;
    // always pad the last dimension.
    unsigned lastDim = paddedShape[paddedShape.size()-1];
    paddedShape[paddedShape.size()-1] = MakeDivisible<unsigned>(lastDim, vectorWords);

    return paddedShape;
}

unsigned TensorF::getLengthBytesPadded(int vectorWords){
    assert(vectorWords>0);
    return getLengthPadded(vectorWords) * sizeof(float);
}

unsigned TensorF::getVectorCountPadded(int vectorWords){
    assert(vectorWords>0);
    unsigned len = getLengthPadded(vectorWords);
    return len / (unsigned)vectorWords;
}

// https://stackoverflow.com/questions/9331561/why-does-my-classs-destructor-get-called-when-i-add-instances-to-a-vector
TensorF::~TensorF() {
    if(platform == PLATFORMS::CPU){
        if(initialized){
            //delete(_buff);
        }
    }
}