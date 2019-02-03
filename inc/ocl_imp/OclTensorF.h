//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORF_H
#define DEEPPOINTV1_OCLTENSORF_H

#include <ocl_imp/xilinx/xcl2.hpp>
#include "../../inc/TensorF.h"

class OclTensorF: public TensorF {
public:
    OclTensorF();
    OclTensorF(cl::Context *context, std::vector<unsigned int> shape);
    OclTensorF(std::vector<unsigned int> shape, cl::Buffer *clBuff);
    void Init(cl::Context *context, std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, cl::Buffer *clBuff);
    void InitWithHostData(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned int> shape, float *hostBuff);
    TensorF* TransferToHost(cl::CommandQueue *queue);
    virtual ~OclTensorF();
    cl::Buffer *ocl_buff;
private:


};


#endif //DEEPPOINTV1_OCLTENSORF_H
