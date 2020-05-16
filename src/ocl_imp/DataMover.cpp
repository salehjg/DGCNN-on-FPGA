#include "ocl_imp/DataMover.h"
#include "TensorF.h"
#include "ocl_imp/OclTensorF.h"
#include "ocl_imp/OclTensorI.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"
#include "ocl_imp/xilinx/xcl2.hpp"
#include <exception>
#include <iostream>
#include <cassert>
#include <iostream>
#include <vector>

int LaunchDataMover( 
    cl::Program *program,
    cl::CommandQueue *queue,
    cl::Context *context,
    cl::Buffer &srcBuff,
    cl::Buffer &dstBuff,
    const unsigned srcBank, 
    const unsigned dstBank, 
    const unsigned len,
    const unsigned vectorWords){

    cl_int error;
#ifdef USEMEMORYBANK0
    OclTensorF *tnDummyBank0 = new OclTensorF(context, queue, {len}, 0, true);
#endif
#ifdef USEMEMORYBANK1
    OclTensorF *tnDummyBank1 = new OclTensorF(context, queue, {len}, 1, true);
#endif
#ifdef USEMEMORYBANK2
    OclTensorF *tnDummyBank2 = new OclTensorF(context, queue, {len}, 2, true);
#endif
#ifdef USEMEMORYBANK3
    OclTensorF *tnDummyBank3 = new OclTensorF(context, queue, {len}, 3, true);
#endif

    if(!(srcBank>=0 && srcBank<=3)){cout<< "Invalid or unsupported srcBank." <<endl; std::exit(3);}
    if(!(dstBank>=0 && dstBank<=3)){cout<< "Invalid or unsupported dstBank." <<endl; std::exit(3);}
    assert(vectorWords>0);

    OCL_CHECK(error,cl::Kernel kernel_datamover(*program, "task_datamover", &error));

    unsigned lenVec = len / (vectorWords);

    int argcnt=0;
    // arguments should be like: bank0 only, bank1 only, bank2 only, and bank3 only.

    //Bank0
#ifdef USEMEMORYBANK0
    if(srcBank==0 || dstBank==0){
        if(srcBank==0){
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, srcBuff));
        }else{
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, dstBuff));
        }
    }else{
        OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, tnDummyBank0->ocl_buff));
    }
#endif

    //Bank1
#ifdef USEMEMORYBANK1
    if(srcBank==1 || dstBank==1){
        if(srcBank==1){
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, srcBuff));
        }else{
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, dstBuff));
        }
    }else{
        OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, tnDummyBank1->ocl_buff));
    }
#endif

    //Bank2
#ifdef USEMEMORYBANK2
    if(srcBank==2 || dstBank==2){
        if(srcBank==2){
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, srcBuff));
        }else{
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, dstBuff));
        }
    }else{
        OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, tnDummyBank2->ocl_buff));
    }
#endif

    //Bank3
#ifdef USEMEMORYBANK3
    if(srcBank==3 || dstBank==3){
        if(srcBank==3){
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, srcBuff));
        }else{
            OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, dstBuff));
        }
    }else{
        OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, tnDummyBank3->ocl_buff));
    }
#endif

    OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, srcBank));
    OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, dstBank));
    OCL_CHECK(error, error = kernel_datamover.setArg(argcnt++, lenVec));

    cl::Event exeEvt;
    OCL_CHECK(error,error = queue->enqueueTask(kernel_datamover, nullptr, &exeEvt));
    exeEvt.wait();
    //queue->finish();

    cout<< "_-_-_-_-_-_-_-_- Internal data-mover kernel executed successfully -_-_-_-_-_-_-_-_"<<endl;

#ifdef USEMEMORYBANK0
    delete(tnDummyBank0);
#endif
#ifdef USEMEMORYBANK1
    delete(tnDummyBank1);
#endif
#ifdef USEMEMORYBANK2
    delete(tnDummyBank2);
#endif
#ifdef USEMEMORYBANK3
    delete(tnDummyBank3);
#endif
}