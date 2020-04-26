#include "ocl_imp/DataMover.h"
#include "TensorF.h"
#include "ocl_imp/OclTensorF.h"
#include "ocl_imp/OclTensorI.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"
#include "ocl_imp/xilinx/xcl.h"
#include <exception>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>

int LaunchDataMover( 
    cl_program program,
    cl_command_queue queue,
    cl_context context,
    cl_mem srcBuff, 
    cl_mem dstBuff, 
    const unsigned srcBank, 
    const unsigned dstBank, 
    const unsigned len,
    const unsigned vectorWords){

    cl_int error;

    //OclTensorF *tnDummyBank0 = new OclTensorF(context, {1}, 0);
    OclTensorF *tnDummyBank1 = new OclTensorF(context, {1}, 1);
    OclTensorF *tnDummyBank2 = new OclTensorF(context, {1}, 2);
    OclTensorF *tnDummyBank3 = new OclTensorF(context, {1}, 3);

    //if(!(srcBank>=0 && srcBank<=3)){cout<< "Invalid or unsupported srcBank." <<endl; std::exit(3);}
    //if(!(dstBank>=0 && dstBank<=3)){cout<< "Invalid or unsupported dstBank." <<endl; std::exit(3);}
    if(!(srcBank>=1 && srcBank<=3)){cout<< "Invalid or unsupported srcBank." <<endl; std::exit(3);}
    if(!(dstBank>=1 && dstBank<=3)){cout<< "Invalid or unsupported dstBank." <<endl; std::exit(3);}
    assert(vectorWords>0);

    cl_kernel kernel_datamover = clCreateKernel(program, "task_datamover", &error);
    if (error != CL_SUCCESS) {
        cout<<  "Failed to create internal data-mover task kernel, Err: " << error << endl;
        std::exit(1);
    }

    unsigned lenVec = len / (vectorWords);

    int argcnt=0;
    // arguments should be like: bank0 only, bank1 only, bank2 only, and bank3 only.

    //Bank0
#ifdef USEMEMORYBANK0
    if(srcBank==0 || dstBank==0){
        if(srcBank==0){
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff);
        }else{
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff);
        }
    }else{
        cl_mem null_mem_object = NULL;
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), & tnDummyBank0->ocl_buff);
    }
#endif

    //Bank1
#ifdef USEMEMORYBANK1
    if(srcBank==1 || dstBank==1){
        if(srcBank==1){
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff);
        }else{
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff);
        }
    }else{
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), & tnDummyBank1->ocl_buff);
    }
#endif

    //Bank2
#ifdef USEMEMORYBANK2
    if(srcBank==2 || dstBank==2){
        if(srcBank==2){
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff);
        }else{
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff);
        }
    }else{
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), & tnDummyBank2->ocl_buff);
    }
#endif

    //Bank3
#ifdef USEMEMORYBANK3
    if(srcBank==3 || dstBank==3){
        if(srcBank==3){
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& srcBuff);
        }else{
            error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), (void*)& dstBuff);
        }
    }else{
        error  = clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_mem), & tnDummyBank3->ocl_buff);
    }
#endif

    error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_uint), (void*)&srcBank); 
    error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_uint), (void*)&dstBank); 
    error |= clSetKernelArg(kernel_datamover, argcnt++, sizeof(cl_uint), (void*)&lenVec);
    
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

    //delete(tnDummyBank0);
    delete(tnDummyBank1);
    delete(tnDummyBank2);
    delete(tnDummyBank3);
}