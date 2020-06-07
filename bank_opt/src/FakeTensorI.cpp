#include <cassert>
#include <iostream>
#include "FakeTensorI.h"
#include "Helper.h"

TensorI::TensorI() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorI::TensorI(std::vector<unsigned> shape, int bank, string _tag) {
    Init(shape, bank, _tag);
}

TensorI::TensorI(std::vector<unsigned> shape, int *buff, int bank, string _tag) {
    Init(shape,buff, bank, _tag);
}

void TensorI::Init(std::vector<unsigned> shape, int bank, string _tag) {
    if(initialized){
        //delete(_buff);
    }
    this->bank = bank;
    this->tag = _tag;
    this->shape = shape;
    this->rank = (int)shape.size();
    initialized = true;
    platform = PLATFORMS::CPU;
}

void TensorI::Init(std::vector<unsigned> shape, int* buff, int bank, string _tag){
    if(initialized){
        //delete(_buff);
    }
    this->bank = bank;
    this->tag = _tag;
    this->shape = shape;
    this->rank = (int)shape.size();
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
            //delete(_buff);
        }
    }
}