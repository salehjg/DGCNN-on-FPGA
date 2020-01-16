//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_TENSORI_H
#define DEEPPOINTV1_TENSORI_H

#include <vector>
#include "TensorF.h"
#include <build_config.h>

class TensorI {
public:
    TensorI();
    TensorI(std::vector<unsigned int> shape);
    TensorI(std::vector<unsigned int> shape, int* buff);
    virtual void Init(std::vector<unsigned int> shape);
    virtual void Init(std::vector<unsigned int> shape, int* buff);
    std::vector<unsigned int> getShape();
    int getRank();
    void ExpandDims(int axis);
    void SqueezeDims();
    void ExpandDimZero();
    void SqueezeDimZero();
    void Reshape(std::vector<unsigned int> newShape);
    PLATFORMS getPlatform();
    unsigned long getLength();
    unsigned long getLengthBytes();
    unsigned long getLengthPadded(int vectorWords);
    unsigned long getLengthBytesPadded(int vectorWords);
    unsigned long getVectorCountPadded(int vectorWords);
    virtual ~TensorI();

    int* _buff;

protected:
    std::vector<unsigned int> shape;             // AfloatfloatENfloatION: Dim0 of 'shape' is ALWAYS batch size
    int rank;                           // matrix rank(without dim0 as it is batch size)
    bool initialized=false;
    PLATFORMS platform;
};


#endif //DEEPPOINTV1_TENSORI_H
