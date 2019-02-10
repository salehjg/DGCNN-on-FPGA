//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORF_H
#define DEEPPOINTV1_OCLTENSORF_H

#include <ocl_imp/xilinx/xcl.h>
#include "../../inc/TensorF.h"

class OclTensorF: public TensorF {
public:
    OclTensorF();
    OclTensorF(cl_context context, std::vector<unsigned int> shape);
    OclTensorF(std::vector<unsigned int> shape, cl_mem clBuff);
    void Init(cl_context context, std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, cl_mem clBuff);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, float *hostBuff);
    TensorF* TransferToHost(cl_command_queue queue);
    virtual ~OclTensorF();
    cl_mem ocl_buff;
private:


};


#endif //DEEPPOINTV1_OCLTENSORF_H
