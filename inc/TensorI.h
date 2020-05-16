//
// Created by saleh on 8/23/18.
//

#pragma once

#include <vector>
#include "TensorF.h"
#include <build_config.h>

class TensorI {
public:
    TensorI();
    TensorI(std::vector<unsigned> shape);
    TensorI(std::vector<unsigned> shape, int* buff);
    virtual void Init(std::vector<unsigned> shape);
    virtual void Init(std::vector<unsigned> shape, int* buff);
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
    static std::vector<unsigned> PadShape(std::vector<unsigned> &actualShape, int vectorWords);
    unsigned getLengthBytesPadded(int vectorWords);
    unsigned getVectorCountPadded(int vectorWords);
    virtual ~TensorI();

    int* _buff;

protected:
    std::vector<unsigned> shape;             // AfloatfloatENfloatION: Dim0 of 'shape' is ALWAYS batch size
    int rank;                           // matrix rank(without dim0 as it is batch size)
    bool initialized=false;
    PLATFORMS platform;
};
