//
// Created by saleh on 8/23/18.
//

#include "ocl_imp/OclTensorF.h"

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <vector>


OclTensorF::OclTensorF(){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
}

OclTensorF::OclTensorF(cl_context context, std::vector<unsigned int> shape){
    Init(context, shape);
}

OclTensorF::OclTensorF(std::vector<unsigned int> shape, cl_mem clBuff){
    Init(shape,clBuff);
}

void OclTensorF::Init(cl_context context, std::vector<unsigned int> shape) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, len, NULL, &ocl_stat);
    assert(ocl_stat==CL_SUCCESS);
}

void OclTensorF::Init(std::vector<unsigned int> shape, cl_mem clBuff){
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_OCL;

    ocl_buff = clBuff;
}

void OclTensorF::InitWithHostData(cl_context context, cl_command_queue queue, std::vector<unsigned int> shape, float *hostBuff) {
    cl_int ocl_stat;
    if(initialized){
        std::cout<<"--- OclTensorF: buffer deleted.\n";
        //assert(clReleaseMemObject(ocl_buff) == CL_SUCCESS);
        delete(ocl_buff);
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

    ocl_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, len, NULL, &ocl_stat);
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

TensorF* OclTensorF::TransferToHost(cl_command_queue queue) {
    TensorF* rsltTn;
    cl_int ocl_stat;
    float* hostBuff = new float[getLength()];
    ocl_stat = clEnqueueReadBuffer(queue, ocl_buff, CL_TRUE, 0, getLengthBytes(), hostBuff, 0, NULL, NULL);
    assert(ocl_stat==CL_SUCCESS);
    rsltTn = new TensorF(getShape(),hostBuff);
    return rsltTn;
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
