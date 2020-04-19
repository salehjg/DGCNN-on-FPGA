//
// Created by saleh on 8/23/18.
//

#pragma once

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"
#include <ocl_imp/xilinx/xcl.h>
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"

class OclTensorI: public TensorI {
public:
    OclTensorI(int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorI(cl_context context, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorI(std::vector<unsigned> shape, cl_mem clBuff, int bank=-1);
    void Init(cl_context context, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    void Init(std::vector<unsigned> shape, cl_mem clBuff, int bank=-1);
    void InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned> shape, int *hostBuff, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    int getDramBank();
    void ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank=-1);
    TensorI* CloneToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank);
    TensorI* CloneIfNeededToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank);
    TensorI* TransferToHost(cl_command_queue queue);
    static int* PadHostBuffer(std::vector<unsigned> actualShape, int *hostSrcBuff, int vectorWords);
    static int* UnPadHostBuffer(std::vector<unsigned> actualShape, int *hostSrcBuff, int vectorWords);
    unsigned getPaddedLastDim();
    virtual ~OclTensorI();
    cl_mem ocl_buff;

private:
    int TranslateBankIndex(int bankIndex);

    //If bank arg were not specified, tensor would be allocated
    //on default bank which is default value of 'dramBank'
    int dramBank = 1;
    int vectorWords = -1;
};
