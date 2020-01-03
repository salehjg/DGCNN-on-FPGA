//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORF_H
#define DEEPPOINTV1_OCLTENSORF_H

#include <ocl_imp/xilinx/xcl.h>
#include "../../inc/TensorF.h"
#include "../../inc/ocl_imp/xilinx/VectorizationHelper.h"

#define DATAMOVER_KERNEL_BANK_A_INDEX   1
#define DATAMOVER_KERNEL_BANK_B_INDEX   2

class OclTensorF: public TensorF {
public:
    OclTensorF(int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorF(cl_context context, std::vector<unsigned int> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorF(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void Init(cl_context context, std::vector<unsigned int> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    void Init(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, float *hostBuff, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    int getDramBank();
    void ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank=-1);
    TensorF* CloneToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank);
    TensorF* TransferToHost(cl_command_queue queue);
    virtual ~OclTensorF();
    cl_mem ocl_buff;
private:
    int LaunchDataMover(cl_program program, cl_command_queue queue, int srcBank, int dstBank, cl_mem srcBuff, cl_mem dstBuff, unsigned long len);
    int TranslateBankIndex(int bankIndex);

    //If bank arg were not specified, tensor would be allocated
    //on default bank which is default value of 'dramBank'
    int dramBank = DATAMOVER_KERNEL_BANK_A_INDEX;
    int vectorWords = -1;
};


#endif //DEEPPOINTV1_OCLTENSORF_H
