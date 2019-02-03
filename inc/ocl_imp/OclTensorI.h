//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORI_H
#define DEEPPOINTV1_OCLTENSORI_H

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"
#include <ocl_imp/xilinx/xcl2.hpp>

class OclTensorI: public TensorI {
public:
    OclTensorI();

    OclTensorI(cl::Context *context, std::vector<unsigned int> shape);
    OclTensorI( std::vector<unsigned int> shape, cl::Buffer *clBuff);
    void Init(cl::Context *context, std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, cl::Buffer *clBuff);
    void InitWithHostData(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned int> shape, int *hostBuff);
    TensorI* TransferToHost(cl::CommandQueue *queue);
    virtual ~OclTensorI();
    cl::Buffer *ocl_buff;
private:


};


#endif //DEEPPOINTV1_OCLTENSORI_H
