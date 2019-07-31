//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_OCLTENSORI_H
#define DEEPPOINTV1_OCLTENSORI_H

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"
#include <ocl_imp/xilinx/xcl.h>

// These two are defined in OclTensorF.h
//#define DATAMOVER_KERNEL_BANK_A_INDEX	1
//#define DATAMOVER_KERNEL_BANK_B_INDEX	2

class OclTensorI: public TensorI {
public:
    OclTensorI();

    OclTensorI(cl_context context, std::vector<unsigned int> shape, int bank=-1);
    OclTensorI(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void Init(cl_context context, std::vector<unsigned int> shape, int bank=-1);
    void Init(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, int *hostBuff, int bank=-1);
    int getDramBank();
    int ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank=-1);
    TensorI* TransferToHost(cl_command_queue queue);
    virtual ~OclTensorI();
    cl_mem ocl_buff;
private:
	int LaunchDataMover(cl_program program, cl_command_queue queue, int srcBank, int dstBank, cl_mem srcBuff, cl_mem dstBuff, unsigned long len);
	int TranslateBankIndex(int bankIndex);

	//If bank arg were not specified, tensor would be allocated
	//on default bank which is default value of 'dramBank'
	int dramBank = DATAMOVER_KERNEL_BANK_A_INDEX;
};


#endif //DEEPPOINTV1_OCLTENSORI_H
