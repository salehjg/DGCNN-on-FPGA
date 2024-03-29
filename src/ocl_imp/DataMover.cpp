#include "build_config.h"
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

static cl_ulong get_duration_ns (const cl::Event &event) {
    cl_int err;
    uint64_t nstimestart, nstimeend;
    OCL_CHECK(err,err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart));
    OCL_CHECK(err,err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend));
    return(nstimeend-nstimestart);
}

static void ReportDuration(const std::string &name, const bool &isNDRange, const cl::Event &event){
    uint64_t ns = get_duration_ns(event);
        SPDLOG_LOGGER_INFO(reporter, 
            "** {}{}(us): {}, (ms): {}, (s): {}", 
            name, 
            (isNDRange?"(ndrange):: ":"(task):: "),
            ns/1000.0f,
            ns/1000000.0f,
            ns/1000000000.0f);
}

inline void PrintInfo(
        string opName,
        const string &setting1, int val1,
        const string &setting2, int val2,
        const string &setting3, float val3,
        vector<unsigned int> shape1,
        vector<unsigned int> shape2,
        vector<bool> comb){

    string finalStr ;
    if(!setting1.empty() && !setting2.empty()){
        finalStr = "## " + opName + ": " + setting1+ "=" + to_string(val1)+ ", " + setting2+ "=" + to_string(val2);
    }else if(!setting1.empty() && setting2.empty()){
        finalStr = "## " + opName + ": " + setting1+ "=" + to_string(val1);
    }else if(setting1.empty() && !setting2.empty()){
        finalStr = "## " + opName + ": " + setting2+ "=" + to_string(val2);
    }else if(setting1.empty() && setting2.empty()){
        finalStr = "## " + opName + ": " ;
    }

    if(!setting3.empty()){
        finalStr += ", " + setting3 + ": " + to_string(val3);
    }

    if(!shape1.empty()){
        finalStr += ", Shape1=";
        for(unsigned int i : shape1){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!shape2.empty()){
        finalStr += ", Shape2=";
        for(unsigned int i : shape2){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!comb.empty()){
        finalStr += ", Combination=";
        for(bool i : comb){ finalStr += to_string(i) + "-"; }
        finalStr += ", ";
    }
    //finalStr+="\n";
    SPDLOG_LOGGER_INFO(reporter, "{}", finalStr);
}

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

    SPDLOG_LOGGER_DEBUG(reporter,"Started");

    cl_int error;
#ifdef USEMEMORYBANK0
    OclTensorF *tnDummyBank0;
    if(!(srcBank==0 || dstBank==0)){
        tnDummyBank0 = new OclTensorF(context, queue, {len}, 0, true);
    }
#endif
#ifdef USEMEMORYBANK1
    OclTensorF *tnDummyBank1;
    if(!(srcBank==1 || dstBank==1)){
        tnDummyBank1 = new OclTensorF(context, queue, {len}, 1, true);
    }
#endif
#ifdef USEMEMORYBANK2
    OclTensorF *tnDummyBank2;
    if(!(srcBank==2 || dstBank==2)){
        tnDummyBank2 = new OclTensorF(context, queue, {len}, 2, true);
    }
#endif
#ifdef USEMEMORYBANK3
    OclTensorF *tnDummyBank3;
    if(!(srcBank==3 || dstBank==3)){
        tnDummyBank3 = new OclTensorF(context, queue, {len}, 3, true);
    }
#endif

    if(!(srcBank>=0 && srcBank<=3)){
        SPDLOG_LOGGER_ERROR(logger,"Invalid or unsupported srcBank");
        std::exit(1);
    }
    if(!(dstBank>=0 && dstBank<=3)){
        SPDLOG_LOGGER_ERROR(logger,"Invalid or unsupported dstBank");
        std::exit(1);
    }
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
    OCL_CHECK(error, error = queue->enqueueTask(kernel_datamover, nullptr, &exeEvt));
    OCL_CHECK(error, error = queue->flush());
    exeEvt.wait();
    //queue->finish();

    PrintInfo("LaunchDataMover","",0,"",0,"",0,{len},{},{});
    ReportDuration(__func__, false, exeEvt);
    SPDLOG_LOGGER_DEBUG(logger,"Internal data-mover kernel executed successfully");

#ifdef USEMEMORYBANK0
    if(!(srcBank==0 || dstBank==0)){
        delete(tnDummyBank0);
    }
#endif
#ifdef USEMEMORYBANK1
    if(!(srcBank==1 || dstBank==1)){
        delete(tnDummyBank1);
    }
#endif
#ifdef USEMEMORYBANK2
    if(!(srcBank==2 || dstBank==2)){
        delete(tnDummyBank2);
    }
#endif
#ifdef USEMEMORYBANK3
    if(!(srcBank==3 || dstBank==3)){
        delete(tnDummyBank3);
    }
#endif
    SPDLOG_LOGGER_DEBUG(reporter,"Finished");
}