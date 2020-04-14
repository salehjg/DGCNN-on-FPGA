//
// Created by saleh on 8/23/18.
//

#pragma once

#include <exception>
#include <ocl_imp/xilinx/xcl.h>
#include "TensorF.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"

struct SameBankException : public std::exception {
   const char * what () const throw () {
      return "Requested to clone to the same memory bank.";
   }
};

constexpr int DATAMOVER_KERNEL_BANK_A_INDEX = 1;
constexpr int DATAMOVER_KERNEL_BANK_B_INDEX = 2;

class OclTensorF: public TensorF {
public:
    OclTensorF(int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorF(cl_context context, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorF(std::vector<unsigned> shape, cl_mem clBuff, int bank=-1);
    void Init(cl_context context, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    void Init(std::vector<unsigned> shape, cl_mem clBuff, int bank=-1);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned> shape, float *hostBuff, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    int getDramBank();
    void ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank=-1);
    TensorF* CloneToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank);
    TensorF* CloneIfNeededToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank);
    TensorF* TransferToHost(cl_command_queue queue);
    static float* PadHostBuffer(std::vector<unsigned> actualShape, float *hostSrcBuff, int vectorWords);
    static float* UnPadHostBuffer(std::vector<unsigned> actualShape, float *hostSrcBuff, int vectorWords);
    unsigned getPaddedLastDim();
    virtual ~OclTensorF();
    cl_mem ocl_buff;
private:
    int LaunchDataMover(cl_program program, cl_command_queue queue, int srcBank, int dstBank, cl_mem srcBuff, cl_mem dstBuff, unsigned len);
    int TranslateBankIndex(int bankIndex);



    //If bank arg were not specified, tensor would be allocated
    //on default bank which is default value of 'dramBank'
    int dramBank = 1;
    int vectorWords = -1;
};
