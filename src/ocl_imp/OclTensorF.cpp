//
// Created by saleh on 8/23/18.
//

#include "ocl_imp/OclTensorF.h"

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

OclTensorF::OclTensorF(){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
}

OclTensorF::OclTensorF(cl_context context, std::vector<unsigned int> shape, int bank){
    Init(context, shape, bank);
}

OclTensorF::OclTensorF(std::vector<unsigned int> shape, cl_mem clBuff, int bank){
    Init(shape, clBuff, bank);
}

void OclTensorF::Init(cl_context context, std::vector<unsigned int> shape, int bank) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
        ///TODO: WFT is going on here? why delete(...) has been used?
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    cl_mem_ext_ptr_t memExt;
    memExt.flags = TranslateBankIndex(dramBank);
    memExt.obj = NULL;
    memExt.param = 0;

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, len, &memExt, &ocl_stat);
    assert(ocl_stat==CL_SUCCESS);
}

void OclTensorF::Init(std::vector<unsigned int> shape, cl_mem clBuff, int bank){
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
        ///TODO: WFT is going on here? why delete(...) has been used?
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    dramBank = bank==-1 ? dramBank : bank;

    ocl_buff = clBuff;
}

void OclTensorF::InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, float *hostBuff, int bank) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
        ///TODO: WFT is going on here? why delete(...) has been used?
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    if(len==0){
        std::cout<<"--- OclTensorF: Warning, Zero Buffer Length.\n";
        len = 16;
    }

    dramBank = bank==-1 ? dramBank : bank;

    cl_mem_ext_ptr_t memExt;
    memExt.flags = TranslateBankIndex(dramBank);
    memExt.obj = NULL;
    memExt.param = 0;

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, len, &memExt, &ocl_stat);
    assert(ocl_stat==CL_SUCCESS);

    // https://software.intel.com/en-us/forums/opencl/topic/731519
    //
    //      If blocking_write is CL_TRUE, the OpenCL implementation copies
    //      the data referred to by ptr and enqueues the write operation
    //      in the command-queue.The memory pointed to by ptr can be reused by
    //      the application after the clEnqueueWriteBuffer call returns.
    //

    ocl_stat = clEnqueueWriteBuffer(queue, ocl_buff, CL_TRUE, 0, len, hostBuff, 0, NULL, NULL);
    assert(ocl_stat==CL_SUCCESS);
}

int OclTensorF::getDramBank(){
    return dramBank;
}

int OclTensorF::LaunchDataMover( 
    cl_program program,
    cl_command_queue queue, 
    int srcBank, 
    int dstBank, 
    cl_mem srcBuff, 
    cl_mem dstBuff, 
    unsigned long len){

    cl_int error;

    if(!(srcBank>=DATAMOVER_KERNEL_BANK_A_INDEX && srcBank<=DATAMOVER_KERNEL_BANK_B_INDEX)){cout<< "Invalid or unsupported srcBank." <<endl; std::exit(3);}
    if(!(dstBank>=DATAMOVER_KERNEL_BANK_A_INDEX && dstBank<=DATAMOVER_KERNEL_BANK_B_INDEX)){cout<< "Invalid or unsupported dstBank." <<endl; std::exit(3);}

    cl_kernel kernel_datamover = clCreateKernel(program, "task_datamover_mod1_float", &error);
    if (error != CL_SUCCESS) {
        cout<<  "Failed to create internal data-mover task kernel, Err: " << error << endl;
        std::exit(1);
    }

    //Current datamover kernel only supports srcBuff within bank0 and dstBuff within bank1.
    //reverseSwitch=0 : Copy srcBuff(bank0) to dstBuff(bank1).
    //reverseSwitch=1 : Copy dstBuff(bank1) to srcBuff(bank0).
    int reverseSwitch = (srcBank==DATAMOVER_KERNEL_BANK_A_INDEX) ? 0 : 1;

    int argcnt=0;
    if(reverseSwitch==0){
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff); //Arg0 should always be on bank0
        error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff); 
    }else{
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff); //Arg0 should always be on bank0
        error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff); 
    }
    error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_int), (void*)&reverseSwitch); 
    error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_ulong), (void*)&len);
    
    if(error != CL_SUCCESS) cout<<"Failed to set internal data-mover kernel args, Err: "<< error <<endl;
    assert(error==CL_SUCCESS);

    cl_event exeEvt;
    error = clEnqueueTask( queue,
                           kernel_datamover,
                           0,
                           NULL,
                           &exeEvt);
    if(error != CL_SUCCESS) cout<<"Failed to launch internal data-mover kernel, Err: "<< error <<endl;
    assert(error==CL_SUCCESS);
    clWaitForEvents(1, &exeEvt);

    cout<< "_-_-_-_-_-_-_-_- Internal data-mover kernel executed successfully -_-_-_-_-_-_-_-_"<<endl;

    error = clReleaseKernel(kernel_datamover);
    if(error != CL_SUCCESS) cout<<"Failed to release internal data-mover kernel, Err: "<< error <<endl;
    assert(error==CL_SUCCESS);
}

// The idea is to hide FPGA specific memory bank related stuff from the top ModelArch.
// The only planned access to this method should be through 'XilinxImplementation' class.
// Because of this, XilinxImpUnitTests wont be able to access cl_program directely.
// It should be accessed through platformSelector.openclPlatformClass(They are public)
void OclTensorF::ChangeDDRBank(cl_program program, cl_context context, cl_command_queue queue, int bank){
    if(initialized){
        //The tensor has been initialized and DOES contain a clBuffer within a different bank.
        //We will run a kernel to read data from old bank and simelteneously write it to the new bank.

        //Forcing memory bank requirements using xilinx external memory extension to opencl.
        cl_mem_ext_ptr_t memExt;
        memExt.flags = TranslateBankIndex(bank);
        memExt.obj = NULL;
        memExt.param = 0;

        unsigned long lenBytes = getLengthBytes();
        unsigned long lenWords = getLength();

        //Creating new buffer within requested memory bank.
        cl_int ocl_stat;
        cl_mem newBuff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, lenBytes, &memExt, &ocl_stat);
        assert(ocl_stat==CL_SUCCESS);

        //Launching data mover kernel to burst read data chunks and burst write them on destination memory bank.
        //Unsupported memory banks will be checked within 'LaunchDataMover' method.
        LaunchDataMover(
            program, 
            queue, 
            dramBank, 
            bank, 
            ocl_buff, 
            newBuff, 
            lenWords);

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
        //Creating new blank tensor within the required bank 
        OclTensorF* clonedTensor = new OclTensorF(context, shape, bank);

        unsigned long lenWords = getLength();

        //Launching data mover kernel to burst read data chunks and burst write them on destination memory bank.
        //Unsupported memory banks will be checked within 'LaunchDataMover' method.
        LaunchDataMover(
            program, 
            queue, 
            dramBank, 
            bank, 
            ocl_buff, 
            clonedTensor->ocl_buff, 
            lenWords);

        return clonedTensor;
        
    }else{
        //The tensor has not yet been initialized!
        cout<<"Trying to clone an uninitialized tensor(OclTensorF)" << endl;
        assert(false);
    }
}

TensorF* OclTensorF::TransferToHost(cl_command_queue queue) {
    TensorF* rsltTn;
    cl_int ocl_stat;
    float* hostBuff = new float[getLength()];
    ocl_stat = clEnqueueReadBuffer(queue, ocl_buff, CL_TRUE, 0, getLengthBytes(), hostBuff, 0, NULL, NULL);
    assert(ocl_stat==CL_SUCCESS);
    rsltTn = new TensorF(getShape(),hostBuff);
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

OclTensorF::~OclTensorF() {
    /* https://stackoverflow.com/questions/17923370/override-identifier-after-destructor-in-c11
     * Even though destructors are not inherited, a destructor in a derived class
     * overrides a base class destructor declared virtual; see 12.4 and 12.5. */

    if(initialized){
        //std::cout<<"--- OclTensorF: buffer deleted.\n";
        assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
    }
}
