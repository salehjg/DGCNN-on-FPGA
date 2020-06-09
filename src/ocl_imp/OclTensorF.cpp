//
// Created by saleh on 8/23/18.
//

#include "ocl_imp/OclTensorF.h"
#include "ocl_imp/DataMover.h"

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

OclTensorF::OclTensorF(int vectorWords){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
    this->vectorWords = vectorWords;
}

OclTensorF::OclTensorF(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, int bank, int vectorWords){
    ValidateBankIndex(bank);
    Init(context, queue, shape, bank, vectorWords);
}

OclTensorF::OclTensorF(std::vector<unsigned> shape, cl::Buffer clBuff, int bank){
    ValidateBankIndex(bank);
    Init(shape, clBuff, bank);
}

void OclTensorF::Init(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, int bank, int vectorWords, bool initToZero) {
    cl_int ocl_stat;
    if(initialized){
        SPDLOG_LOGGER_DEBUG(logger,"OclTensorF: buffer deleted");
        //delete(ocl_buff);
    }
    this->vectorWords = vectorWords;
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned lenPadded = getLengthBytesPadded(this->vectorWords);
    platform = PLATFORMS::GPU_OCL;

    ValidateBankIndex(bank);
    dramBank = bank==-1 ? dramBank : bank;

    // https://software.intel.com/en-us/forums/opencl/topic/731519
    //
    //      If blocking_write is CL_TRUE, the OpenCL implementation copies
    //      the data referred to by ptr and enqueues the write operation
    //      in the command-queue.The memory pointed to by ptr can be reused by
    //      the application after the clEnqueueWriteBuffer call returns.
    //

    cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(dramBank));
    cl_mem_flags  flags = CL_MEM_READ_WRITE;
    //flags |= CL_MEM_USE_HOST_PTR;
    flags |= CL_MEM_EXT_PTR_XILINX;

    OCL_CHECK(ocl_stat, ocl_buff = cl::Buffer(*context, flags, lenPadded, &extPtr, &ocl_stat));
    if(initToZero){
        const unsigned zeroInitBuffLen = getLengthPadded(this->vectorWords);
        float *zeroInitBuff = new float[zeroInitBuffLen];
        for(unsigned i=0; i<zeroInitBuffLen; i++){zeroInitBuff[i]=0;}
        //cl::Event event;
        OCL_CHECK(ocl_stat, ocl_stat = queue->enqueueWriteBuffer(ocl_buff, CL_TRUE, 0, lenPadded, zeroInitBuff, nullptr, nullptr));
        //event.wait();
    }
}

void OclTensorF::Init(std::vector<unsigned> shape, cl::Buffer clBuff, int bank){
    cl_int ocl_stat;
    SPDLOG_LOGGER_DEBUG(logger,"OclTensorF: Warning: No padding");
    if(initialized){
        SPDLOG_LOGGER_DEBUG(logger,"OclTensorF: buffer deleted");      
        //delete(ocl_buff);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    ocl_buff = clBuff;
}

void OclTensorF::InitWithHostData(cl::Context *context, cl::CommandQueue *queue, std::vector<unsigned> shape, float *hostBuff, int bank, int vectorWords) {
    cl_int ocl_stat;
    if(initialized){
        SPDLOG_LOGGER_DEBUG(logger,"OclTensorF: buffer deleted");
        //delete(ocl_buff);
    }
    this->vectorWords = vectorWords;
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned lenPadded = getLengthBytesPadded(this->vectorWords);
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    // https://software.intel.com/en-us/forums/opencl/topic/731519
    //
    //      If blocking_write is CL_TRUE, the OpenCL implementation copies
    //      the data referred to by ptr and enqueues the write operation
    //      in the command-queue.The memory pointed to by ptr can be reused by
    //      the application after the clEnqueueWriteBuffer call returns.
    //

    float *paddedBuff = PadHostBuffer(shape, hostBuff, vectorWords);

    cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(dramBank));
    cl_mem_flags  flags = CL_MEM_READ_WRITE;
    //flags |= CL_MEM_USE_HOST_PTR;
    flags |= CL_MEM_EXT_PTR_XILINX;

    OCL_CHECK(ocl_stat, ocl_buff = cl::Buffer(*context, flags, lenPadded, &extPtr, &ocl_stat));
    //cl::Event event;
    OCL_CHECK(ocl_stat, ocl_stat = queue->enqueueWriteBuffer(ocl_buff, CL_TRUE, 0, lenPadded, paddedBuff, nullptr, nullptr));
    //event.wait();
    delete(paddedBuff);
}

int OclTensorF::getDramBank(){
    return dramBank;
}

// The idea is to hide FPGA specific memory bank related stuff from top ModelArch.
// The only planned access to this method should be through 'XilinxImplementation' class.
// Because of this, XilinxImpUnitTests wont be able to access cl_program directely.
// It should be accessed through platformSelector.openclPlatformClass(They are public)
void OclTensorF::ChangeDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank){
    if(initialized){
        //The tensor has been initialized and DOES contain a clBuffer within a different bank.
        //We will run a kernel to read data from old bank and simelteneously write it to the new bank.

    	if(bank == dramBank){
            SPDLOG_LOGGER_ERROR(logger,"Trying to change to the same bank (OclTensorF)");
            std::exit(3);
        }
        assert(this->vectorWords>0);

        cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(bank));
        cl_mem_flags  flags = CL_MEM_READ_WRITE;
        //flags |= CL_MEM_USE_HOST_PTR;
        flags |= CL_MEM_EXT_PTR_XILINX;

        unsigned lenBytesPadded = getLengthBytesPadded(this->vectorWords);
        unsigned lenWordsPadded = getLengthPadded(this->vectorWords);

        //Creating new buffer within requested memory bank.
        cl_int ocl_stat;
        OCL_CHECK(ocl_stat, cl::Buffer newBuff = cl::Buffer(*context, flags, lenBytesPadded, &extPtr, &ocl_stat));

        //Launching data mover kernel to burst read data chunks and burst write them on destination memory bank.
        //Unsupported memory banks will be checked within 'LaunchDataMover' method.
        LaunchDataMover(
            program, 
            queue,
            context,
            ocl_buff, 
            newBuff, 
            dramBank, 
            bank, 
            lenWordsPadded,
            (unsigned)this->vectorWords);

        //Now we have to release the old buffer and replace it with the new one.
        // OCL C++ interface: will be released automatically

        //Replacing old released buffer with new one.
        ocl_buff = newBuff;

        dramBank = bank;

    }else{
        //The tensor has not yet been initialized, meaning that it does not contain clBuffer object yet to change its bank.
        dramBank = bank;
    }
}

//Creates a new tensor within bank index 'arg:bank' and copies the content there, then returns the new tensor.
//The content and the bank of the current tensor will be remained untouched. 
TensorF* OclTensorF::CloneToDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank){
    if(initialized){
        if(bank == dramBank){
            throw SameBankException();
        }

        //Creating new blank tensor within the required bank 
        OclTensorF* clonedTensor = new OclTensorF(context, queue, shape, bank);

        unsigned lenWordsPadded = getLengthPadded(this->vectorWords);

        //Launching data mover kernel to burst read data chunks and burst write them on destination memory bank.
        //Unsupported memory banks will be checked within 'LaunchDataMover' method.
        LaunchDataMover(
            program, 
            queue,
            context,
            ocl_buff, 
            clonedTensor->ocl_buff, 
            dramBank, 
            bank, 
            lenWordsPadded,
            (unsigned)this->vectorWords);

        return clonedTensor;
        
    }else{
        //The tensor has not yet been initialized!
        SPDLOG_LOGGER_ERROR(logger,"Trying to clone an uninitialized tensor(OclTensorF)");
        assert(false);
    }
}

TensorF* OclTensorF::CloneIfNeededToDDRBank(cl::Program *program, cl::Context *context, cl::CommandQueue *queue, int bank){
    try{
        return CloneToDDRBank(program, context, queue, bank);
    } catch(SameBankException& e) {
        SPDLOG_LOGGER_DEBUG(logger,"CloneIfNeededToDDRBank: same banks detected, returning the original tensor");
        return this;
    }
}

TensorF* OclTensorF::TransferToHost(cl::CommandQueue *queue) {
    TensorF* rsltTn;
    cl_int ocl_stat;
    float* hostBuffPadded = new float[getLengthPadded(vectorWords)];

    OCL_CHECK(ocl_stat,ocl_stat = queue->enqueueReadBuffer(
            ocl_buff,
            CL_TRUE,
            0,
            getLengthBytesPadded(vectorWords),
            hostBuffPadded,
            nullptr,
            nullptr));

    float* hostBuff = UnPadHostBuffer(shape, hostBuffPadded, vectorWords);
    rsltTn = new TensorF(getShape(),hostBuff);
    delete(hostBuffPadded);
    return rsltTn;
}

int OclTensorF::TranslateBankIndex(int bankIndex){
    switch(bankIndex){
        case 0:{
            return XCL_MEM_DDR_BANK0;
        }break;
        case 1:{
            return XCL_MEM_DDR_BANK1;
        }break;
        case 2:{
            return XCL_MEM_DDR_BANK2;
        }break;
        case 3:{
            return XCL_MEM_DDR_BANK3;
        }break;
    };
}

void OclTensorF::ValidateBankIndex(int bankIndex){
    if(bankIndex!=-1){
#ifndef USEMEMORYBANK0
        assert(bankIndex!=0);
#endif
#ifndef USEMEMORYBANK1
        assert(bankIndex!=1);
#endif
#ifndef USEMEMORYBANK2
        assert(bankIndex!=2);
#endif
#ifndef USEMEMORYBANK3
        assert(bankIndex!=3);
#endif
    }
}

float* OclTensorF::PadHostBuffer(std::vector<unsigned> actualShape, float *hostSrcBuff, int vectorWords){
    std::vector<unsigned> paddedShape = PadShape(actualShape, vectorWords);
    unsigned paddedLen = 1;
    for(int i=0; i<paddedShape.size(); i++){
        paddedLen = paddedLen * paddedShape[i];
    }

    const unsigned sliceCount = paddedLen / paddedShape[paddedShape.size()-1];
    const int actualSliceLen = actualShape[actualShape.size()-1];
    const int paddedSliceLen = paddedShape[actualShape.size()-1];
    float *paddedBuff = new float[paddedLen];

    for(unsigned slice=0; slice<sliceCount; slice++){
        for(int i=0; i<paddedSliceLen; i++){
            paddedBuff[slice*paddedSliceLen + i] = (i<actualSliceLen)? hostSrcBuff[slice*actualSliceLen + i] : 0;
        }
    }

    return paddedBuff;
}

float* OclTensorF::UnPadHostBuffer(std::vector<unsigned> actualShape, float *hostSrcBuff, int vectorWords){
    std::vector<unsigned> paddedShape = PadShape(actualShape, vectorWords);
    unsigned paddedLen = 1;
    for(int i=0; i<paddedShape.size(); i++){
        paddedLen = paddedLen * paddedShape[i];
    }

    const unsigned sliceCount = paddedLen / paddedShape[paddedShape.size()-1];
    const int actualSliceLen = actualShape[actualShape.size()-1];
    const int paddedSliceLen = paddedShape[actualShape.size()-1];
    float *unpaddedBuff = new float[paddedLen];

    for(unsigned slice=0; slice<sliceCount; slice++){
        for(int i=0; i<actualSliceLen; i++){
            unpaddedBuff[slice*actualSliceLen + i] = hostSrcBuff[slice*paddedSliceLen + i];
        }
    }

    return unpaddedBuff;
}

unsigned OclTensorF::getPaddedLastDim(){
    if(initialized){
        std::vector<unsigned> paddedShape = PadShape(shape, vectorWords);
        return paddedShape[paddedShape.size()-1];
    }else{
        return 0;
    }
}

OclTensorF::~OclTensorF() {
    /* https://stackoverflow.com/questions/17923370/override-identifier-after-destructor-in-c11
     * Even though destructors are not inherited, a destructor in a derived class
     * overrides a base class destructor declared virtual; see 12.4 and 12.5. */

    if(initialized){
        //std::cout<<"--- OclTensorF: buffer deleted.\n";
    }
}

cl_mem_ext_ptr_t OclTensorF::CreateExtendedPointer(void *hostPtr, cl_mem_flags memoryBank){
    cl_mem_ext_ptr_t extendedPointer;
    extendedPointer.flags = memoryBank;
    extendedPointer.obj = hostPtr;
    extendedPointer.param = 0;
    return extendedPointer;
}

std::string OclTensorF::GetTensorTag(){
    return tensorTag;
}

void OclTensorF::SetTensorTag(std::string tag){
    tensorTag = tag;
}