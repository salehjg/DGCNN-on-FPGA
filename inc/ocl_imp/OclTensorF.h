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
    OclTensorF(cl_context context, std::vector<unsigned int> shape, int bank=-1);// Should be changed 
    OclTensorF(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void Init(cl_context context, std::vector<unsigned int> shape, int bank=-1);// Should be changed 
    void Init(std::vector<unsigned int> shape, cl_mem clBuff, int bank=-1);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, float *hostBuff, int bank=-1);// Should be changed 
    int getDramBank();
    int ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank=-1);
    TensorF* TransferToHost(cl_command_queue queue);
    virtual ~OclTensorF();
    cl_mem ocl_buff;
private:
	int LaunchDataMover(cl_program program, cl_command_queue queue, int srcBank, int dstBank, cl_mem srcBuff, cl_mem dstBuff, unsigned long len);
	int TranslateBankIndex(int bankIndex);

	//If bank arg were not specified, tensor would be allocated
	//on default bank which is default value of 'dramBank'
	int dramBank = 0;

};


#endif //DEEPPOINTV1_OCLTENSORF_H
