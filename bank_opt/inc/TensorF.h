#pragma once

#include <vector>
#include "Helper.h"

class TensorF {
public:
    TensorF();
    TensorF(std::vector<unsigned> shape);
    TensorF(std::vector<unsigned> shape, float* buff);
    virtual void Init(std::vector<unsigned> shape);
    virtual void Init(std::vector<unsigned> shape, float* buff);
    std::vector<unsigned> getShape();
    int getRank();
    void ExpandDims(int axis);
    void SqueezeDims();
    void ExpandDimZero();
    void SqueezeDimZero();
    void Reshape(std::vector<unsigned> newShape);
    PLATFORMS getPlatform();
    unsigned getLength();
    unsigned getLengthBytes();
    unsigned getLengthPadded(int vectorWords);
    unsigned getLengthBytesPadded(int vectorWords);
    unsigned getVectorCountPadded(int vectorWords);
    static std::vector<unsigned> PadShape(std::vector<unsigned> &actualShape, int vectorWords);
    virtual ~TensorF();

    float* _buff;

protected:
    std::vector<unsigned> shape;
    int rank;
    bool initialized=false;
    PLATFORMS platform;
};
