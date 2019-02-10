//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORI_H
#define DEEPPOINTV1_OCLTENSORI_H

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"
#include <ocl_imp/xilinx/xcl.h>

class OclTensorI: public TensorI {
public:
    OclTensorI();

    OclTensorI(cl_context context, std::vector<unsigned int> shape);
    OclTensorI( std::vector<unsigned int> shape, cl_mem clBuff);
    void Init(cl_context context, std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, cl_mem clBuff);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, int *hostBuff);
    TensorI* TransferToHost(cl_command_queue queue);
    virtual ~OclTensorI();

    cl_mem ocl_buff;
private:


};


#endif //DEEPPOINTV1_OCLTENSORI_H
