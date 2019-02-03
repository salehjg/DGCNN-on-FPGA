//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_TENSORF_H
#define DEEPPOINTV1_TENSORF_H

#include <vector>
#include <build_config.h>
enum class PLATFORMS{
    DEFAULT,
    CPU,
    GPU_CUDA,
    GPU_OCL
};

class TensorF {
public:
    TensorF();
    TensorF(std::vector<unsigned int> shape);
    TensorF(std::vector<unsigned int> shape, float* buff);
    virtual void Init(std::vector<unsigned int> shape);
    virtual void Init(std::vector<unsigned int> shape, float* buff);
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
    virtual ~TensorF();

    float* _buff;

protected:
    std::vector<unsigned int> shape;
    int rank;
    bool initialized=false;
    PLATFORMS platform;
};


#endif //DEEPPOINTV1_TENSORF_H
