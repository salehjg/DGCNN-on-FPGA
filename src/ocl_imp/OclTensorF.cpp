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

using namespace std;

OclTensorF::OclTensorF(int vectorWords){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
    this->vectorWords = vectorWords;
}

OclTensorF::OclTensorF(cl_context context, std::vector<unsigned> shape, int bank, int vectorWords){
    Init(context, shape, bank, vectorWords);
}

OclTensorF::OclTensorF(std::vector<unsigned> shape, cl_mem clBuff, int bank){
    Init(shape, clBuff, bank);
}

void OclTensorF::Init(cl_context context, std::vector<unsigned> shape, int bank, int vectorWords) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        //delete(ocl_buff);
    }
    this->vectorWords = vectorWords;
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned lenPadded = getLengthBytesPadded(this->vectorWords);
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    cl_mem_ext_ptr_t memExt;
    memExt.flags = TranslateBankIndex(dramBank);
    memExt.obj = NULL;
    memExt.param = 0;

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, lenPadded, &memExt, &ocl_stat);
    assert(ocl_stat==CL_SUCCESS);
}

void OclTensorF::Init(std::vector<unsigned> shape, cl_mem clBuff, int bank){
    cl_int ocl_stat;
    std::cout<<"--- OclTensorF: Warning: No padding\n";
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
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

void OclTensorF::InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned> shape, float *hostBuff, int bank, int vectorWords) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        //delete(ocl_buff);
    }
    this->vectorWords = vectorWords;
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned lenPadded = getLengthBytesPadded(this->vectorWords);
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    cl_mem_ext_ptr_t memExt;
    memExt.flags = TranslateBankIndex(dramBank);
    memExt.obj = NULL;
    memExt.param = 0;

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, lenPadded, &memExt, &ocl_stat);
    assert(ocl_stat==CL_SUCCESS);

    // https://software.intel.com/en-us/forums/opencl/topic/731519
    //
    //      If blocking_write is CL_TRUE, the OpenCL implementation copies
    //      the data referred to by ptr and enqueues the write operation
    //      in the command-queue.The memory pointed to by ptr can be reused by
    //      the application after the clEnqueueWriteBuffer call returns.
    //

    float *paddedBuff = PadHostBuffer(shape, hostBuff, vectorWords);

    ocl_stat = clEnqueueWriteBuffer(queue, ocl_buff, CL_TRUE, 0, lenPadded, paddedBuff, 0, NULL, NULL);
    assert(ocl_stat==CL_SUCCESS);

    delete(paddedBuff);
}

int OclTensorF::getDramBank(){
    return dramBank;
}

// The idea is to hide FPGA specific memory bank related stuff from top ModelArch.
// The only planned access to this method should be through 'XilinxImplementation' class.
// Because of this, XilinxImpUnitTests wont be able to access cl_program directely.
// It should be accessed through platformSelector.openclPlatformClass(They are public)
void OclTensorF::ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank){
    if(initialized){
        //The tensor has been initialized and DOES contain a clBuffer within a different bank.
        //We will run a kernel to read data from old bank and simelteneously write it to the new bank.

    	if(bank == dramBank){cout<<"Trying to change to the same bank (OclTensorF)."<<endl; std::exit(3);}
        assert(this->vectorWords>0);

        //Forcing memory bank requirements using xilinx external memory extension to opencl.
        cl_mem_ext_ptr_t memExt;
        memExt.flags = TranslateBankIndex(bank);
        memExt.obj = NULL;
        memExt.param = 0;

        unsigned lenBytesPadded = getLengthBytesPadded(this->vectorWords);
        unsigned lenWordsPadded = getLengthPadded(this->vectorWords);

        //Creating new buffer within requested memory bank.
        cl_int ocl_stat;
        cl_mem newBuff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, lenBytesPadded, &memExt, &ocl_stat);
        assert(ocl_stat==CL_SUCCESS);

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
        cl_int error = clReleaseMemObject(ocl_buff);
        if(error != CL_SUCCESS){
            cout<<"Failed to release old buffer(opencl), Err: " << error << endl;
            assert(error==CL_SUCCESS);
        }

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
TensorF* OclTensorF::CloneToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank){
    if(initialized){
        if(bank == dramBank){
            throw SameBankException();
        }

        //Creating new blank tensor within the required bank 
        OclTensorF* clonedTensor = new OclTensorF(context, shape, bank);

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
        cout<<"Trying to clone an uninitialized tensor(OclTensorF)" << endl;
        assert(false);
    }
}

TensorF* OclTensorF::CloneIfNeededToDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank){
    try{
        return CloneToDDRBank(program, context, queue, bank);
    } catch(SameBankException& e) {
        std::cout<<"CloneIfNeededToDDRBank: same banks detected, returning the original tensor."<< std::endl;
        return this;
    }
}

TensorF* OclTensorF::TransferToHost(cl_command_queue queue) {
    TensorF* rsltTn;
    cl_int ocl_stat;
    float* hostBuffPadded = new float[getLengthPadded(vectorWords)];
    ocl_stat = clEnqueueReadBuffer(queue, ocl_buff, CL_TRUE, 0, getLengthBytesPadded(vectorWords), hostBuffPadded, 0, NULL, NULL);
    assert(ocl_stat==CL_SUCCESS);
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
        assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
    }
}
