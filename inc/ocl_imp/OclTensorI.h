//
// Created by saleh on 8/23/18.
//

#pragma once

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"
#include <ocl_imp/xilinx/xcl2.hpp>
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"
#include <string>

class OclTensorI: public TensorI {
public:
    OclTensorI(int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorI(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    OclTensorI(std::vector<unsigned> shape, cl::Buffer clBuff, int bank=-1);
    void Init(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH, bool initToZero=false);
    void Init(std::vector<unsigned> shape, cl::Buffer clBuff, int bank=-1);
    void InitWithHostData(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, int *hostBuff, int bank=-1, int vectorWords = CONFIG_M_AXI_WIDTH);
    int getDramBank();
    void ChangeDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank=-1);
    TensorI* CloneToDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank);
    TensorI* CloneIfNeededToDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank);
    TensorI* TransferToHost(cl::CommandQueue *queue);
    static int* PadHostBuffer(std::vector<unsigned> actualShape, int *hostSrcBuff, int vectorWords);
    static int* UnPadHostBuffer(std::vector<unsigned> actualShape, int *hostSrcBuff, int vectorWords);
    unsigned getPaddedLastDim();
    std::string GetTensorTag();
    void SetTensorTag(std::string tag);
    virtual ~OclTensorI();
    cl::Buffer ocl_buff;

private:
    int TranslateBankIndex(int bankIndex);
    void ValidateBankIndex(int bankIndex);
    cl_mem_ext_ptr_t CreateExtendedPointer(void *hostPtr, cl_mem_flags memoryBank);

    //If bank arg were not specified, tensor would be allocated
    //on default bank which is default value of 'dramBank'
#ifdef USEMEMORYBANK0
    int dramBank = 0;
#else
    #ifdef USEMEMORYBANK1
        int dramBank = 1;
    #else
        #ifdef USEMEMORYBANK2
            int dramBank = 2;
        #else
            #ifdef USEMEMORYBANK3
                int dramBank = 3;
            #else
                assert(false);
            #endif  
        #endif  
    #endif  
#endif 
    int vectorWords = -1;
    std::string tensorTag = "unknownTag";
};
