//
// Created by saleh on 8/22/18.
//

#include <iostream>
#include <cassert>
#include <ocl_imp/xilinx/XilinxImplementation.h>
#include "xilinx/config.h"
#include <algorithm>
#include <vector>

using namespace std;

#define DISABLED_KERNEL (true)

XilinxImplementation::XilinxImplementation(int aa){
    a = aa;
    //======================================================================================================================
    {
        const RUN_MODE mode = GetModeEnvVar();
        if(mode==RUN_MODE::Unknown){
            cout<<"WARNING: XCL_EMULATION_MODE is not set. System run(real FPGA) is considered.";
            //assert(SetModeEnvVar(RUN_MODE::SwEmu)==0);
        }else{
            cout << "Mode: " << (
                mode==RUN_MODE::SwEmu?"Sw-emulation":
                mode==RUN_MODE::HwEmu?"Hw-emulation":
                "Hardware(FPGA)" ) << endl;
        }
    }

    //======================================================================================================================
    {
        auto devices = xcl::get_xil_devices();
        cout<<"Xilinx Devices Found: "<< devices.size()<<endl;
        assert(devices.size()>0);

        cout<<"Using device index 0"<<endl;
        device = devices[0];

        OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err,queue = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        deviceName = device.getInfo<CL_DEVICE_NAME>();
        cout << "Found Device=" << deviceName.c_str() << endl;

        auto fileBuf = xcl::read_binary_file(globalArgXclBin);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        OCL_CHECK(err,program = new cl::Program(*context, {device}, bins, NULL, &err));
    }

    //======================================================================================================================
    oclKernels = {
        /*
         clKernelObject(
            dir,
            fname,
            containerName,
            kernelName_ndrange,
            kernelName_task,
            use_ndrange_kernel)
            */

        /* IDX 0 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/concat.cpp",
                "binary_container_1.xclbin",
                "",
                "task_concat",
                false),
        /* IDX 1 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/reduce.cpp",
                "binary_container_1.xclbin",
                "",
                "task_reduce",
                false),
        /* IDX 2 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/matops.cpp",
                "binary_container_1.xclbin",
                "",
                "task_matops",
                false),
        /* IDX 3 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/tile.cpp",
                "binary_container_1.xclbin",
                "",
                "task_tile",
                false),
        /* IDX 4 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/matmul.cpp",
                "binary_container_1.xclbin",
                "",
                "task_matmul",
                false),
        /* IDX 5 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/transpose.cpp",
                "binary_container_1.xclbin",
                "",
                "task_transpose",
                false),
        /* IDX 6 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/gather.cpp",
                "binary_container_1.xclbin",
                "",
                "task_gather",
                false),
        /* IDX 7 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/conv2_1x1_direct.cpp",
                "binary_container_1.xclbin",
                "",
                "task_conv2_1x1_direct",
                false),
        /* IDX 8 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/topk.cpp",
                "binary_container_1.xclbin",
                "",
                "task_topk",
                false),
        /* IDX 9 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/pad_unpad.cpp",
                "binary_container_1.xclbin",
                "",
                "task_pad_unpad",
                false),
        /* IDX 10 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/relu_sqrt_square.cpp",
                "binary_container_1.xclbin",
                "",
                "task_relu_sqrt_square",
                false)
    };
    
    //======================================================================================================================
    for(OclKernelObject *kernelObject : oclKernels){
        if(kernelObject->disabled) continue;
        if(kernelObject->use_ndrange_kernel){
            //NYI
        }else{
            OCL_CHECK(err,kernelObject->kernel_task = new cl::Kernel(*program, kernelObject->kernelName_task, &err));
        }
    }

    std::cout<<"- - - - - - - - - - -"<<std::endl;
    
    //======================================================================================================================
}

XilinxImplementation::~XilinxImplementation(){
    cout<<"~XilinxImplementation"<<endl;
    /*
    delete(queue);
    delete(program);
    delete(context);

    for(OclKernelObject *kernelObject : oclKernels){
        if(kernelObject->use_ndrange_kernel)
            delete(kernelObject->kernel_ndrange);
        else
            delete(kernelObject->kernel_task);
    }
    */
}

cl::Context* XilinxImplementation::getContext(){
    return context;
}

cl::Program* XilinxImplementation::getProgram() {
    return program;
}

cl::CommandQueue* XilinxImplementation::getQueue() {
    return queue;
}

void XilinxImplementation::PrintInfo(
        string opName,
        const string &setting1, int val1,
        const string &setting2, int val2,
        const string &setting3, float val3,
        vector<unsigned int> shape1,
        vector<unsigned int> shape2,
        vector<bool> comb){

    string finalStr ;
    if(!setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1)+ ", " + setting2+ "=" + to_string(val2);
    }else if(!setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1);
    }else if(setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting2+ "=" + to_string(val2);
    }else if(setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " ;
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
    finalStr+="\n";
    cout<<finalStr;
}

TensorF* XilinxImplementation::Transpose(WorkScheduler scheduler, TensorF *batchedMat){
    PrintInfo("Transpose","",0,"",0,"",0,batchedMat->getShape(),{});

    cl_int error;
    assert(batchedMat->getRank()==2 || batchedMat->getRank()==3);
    int rankDiff = 3 - batchedMat->getRank();
    if(rankDiff) batchedMat->ExpandDimZero();

    cl_uint dim0,dim1,dim2;
    dim0 = batchedMat->getShape()[0];
    dim1 = batchedMat->getShape()[1];
    dim2 = batchedMat->getShape()[2];

    TensorF* _batchedMat = ((OclTensorF*)batchedMat)->CloneIfNeededToDDRBank(program,context,queue, ConfigTaskTranspose::BankIndex_inputTn);
    OclTensorF *rsltTn = new OclTensorF(context, queue, {dim0,dim2,dim1}, ConfigTaskTranspose::BankIndex_outputTn);
    OclKernelObject *kernelObject = oclKernels[5];

    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }else{
        int argcnt = 0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_batchedMat)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__, kernelObject->use_ndrange_kernel, exeEvt);

        if(rankDiff) batchedMat->SqueezeDimZero();
        return rsltTn;
    }
}

TensorF* XilinxImplementation::MatMul(WorkScheduler scheduler,
                                   TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());

    assert(batchedMat1->getRank()==3 || batchedMat1->getRank()==2);
    assert(batchedMat2->getRank()==3 || batchedMat2->getRank()==2);
    assert(batchedMat2->getRank()==batchedMat2->getRank());

    unsigned dim0A,dim1A,dim2A,dim0B,dim1B,dim2B;
    int rankDiff;

    rankDiff = 3 - batchedMat1->getRank();
    for(int i=0;i<rankDiff;i++){
        batchedMat1->ExpandDimZero();
        batchedMat2->ExpandDimZero();
    }

    dim0A = batchedMat1->getShape()[0]; // batch size
    dim1A = batchedMat1->getShape()[1]; // height of matrix ,N
    dim2A = batchedMat1->getShape()[2]; // width of matrix  ,K

    dim0B = batchedMat2->getShape()[0]; // batch size
    dim1B = batchedMat2->getShape()[1]; // height of matrix ,K
    dim2B = batchedMat2->getShape()[2]; // width of matrix  ,M

    // Width of A should be equal to the Height of B. (dim2A = dim1B)
    assert(dim0A == dim0B);
    assert(dim2A == dim1B);

    TensorF* _batchedMat1 = ((OclTensorF*)batchedMat1)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskMatMul::BankIndex_inputTn1);
    TensorF* _batchedMat2 = ((OclTensorF*)batchedMat2)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskMatMul::BankIndex_inputTn2);

    OclTensorF* rsltTn = new OclTensorF(context, queue, {dim0A,dim1A,dim2B}, ConfigTaskMatMul::BankIndex_outputTn);
    OclKernelObject *kernelObject = oclKernels[4];

    if(kernelObject->use_ndrange_kernel){
        for(int i=0;i<rankDiff;i++){
            batchedMat1->SqueezeDimZero();
            batchedMat2->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }
        return nullptr;
    }else{
        //printf("KERNEL PARAMS: B N K M = %d %d %d %d\n",dim0A,dim1A,dim2A,dim2B);
        cl_int error; int argcnt=0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_batchedMat1)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_batchedMat2)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0A));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1A));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2A));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2B));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        for(int i=0;i<rankDiff;i++){
            batchedMat1->SqueezeDimZero();
            batchedMat2->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }

        return rsltTn;
    }
}

TensorF* XilinxImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});
    return _ReluSqrtSquare(scheduler, batchedMat, false, false, true);
}

TensorF* XilinxImplementation::ReduceSum(WorkScheduler scheduler,
                                      TensorF* inputTn,
                                      bool over_axis0,
                                      bool over_axis1,
                                      bool over_axis2){
    PrintInfo("ReduceSum","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2});
    return _Reduce_Task(inputTn, true, false, 0, over_axis0, over_axis1, over_axis2, false);
}

TensorF* XilinxImplementation::_Reduce_Task(
        TensorF* inputTn,
        bool reduceSum,
        bool reduceMax,
        unsigned pow_y,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3){

    assert( !(reduceSum && reduceMax) && (reduceSum || reduceMax) );
    const unsigned rank = inputTn->getRank();
    unsigned _dim0,_dim1,_dim2,_dim3;
    cl_int error;
    int argcnt=0;
    unsigned mode=0;
    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskReduce::BankIndex_inputTn);
    OclTensorF* rsltTn;

    if(reduceSum && rank==3){
        assert(!overaxis0&&!overaxis1&&overaxis2); // ReduceSum3D FFT
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        _dim2 = inputTn->getShape()[2];
        mode = 1;
        rsltTn = new OclTensorF(context, queue, {_dim0,_dim1}, ConfigTaskReduce::BankIndex_outputTn);
    }else if(reduceSum && rank==4){
        assert(overaxis0&&overaxis1&&overaxis2&&!overaxis3); // ReduceSum4D TTTF
        assert(pow_y<=ConfigTaskReduce::Sum4D::MaxPowY);
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        _dim2 = inputTn->getShape()[2];
        _dim3 = inputTn->getShape()[3];
        mode = 2;
        rsltTn = new OclTensorF(context, queue, {_dim3}, ConfigTaskReduce::BankIndex_outputTn);
    }else if(reduceMax && rank==4){

        // ReduceMax4D FTF (ReduceMax3D kernel will be used on FTF mode)
        const bool reduceMaxFTFF = (!overaxis0&& overaxis1&& !overaxis2&& !overaxis3);

        // ReduceMax4D FTF (ReduceMax3D kernel will be used on FTF mode)
        const bool reduceMaxFFTF = (!overaxis0&& !overaxis1&& overaxis2&& !overaxis3);

        assert(
            !(reduceMaxFTFF && reduceMaxFFTF) &&
            (reduceMaxFTFF || reduceMaxFFTF)  
        );

        if(reduceMaxFTFF && inputTn->getShape()[2]!=1){
            cout<<"ReduceMax: For reductionDim=1, Dim2 should be equals 1."<<endl;
            return nullptr;
        }

        if(reduceMaxFTFF){
            _dim0 = inputTn->getShape()[0];
            _dim1 = inputTn->getShape()[1];
            _dim2 = inputTn->getShape()[3];
            rsltTn = new OclTensorF(
                context, 
                queue, 
                {inputTn->getShape()[0], inputTn->getShape()[2], inputTn->getShape()[3]}, 
                ConfigTaskReduce::BankIndex_outputTn);

        }else if(reduceMaxFFTF){
            _dim0 = inputTn->getShape()[0]*inputTn->getShape()[1];
            _dim1 = inputTn->getShape()[2];
            _dim2 = inputTn->getShape()[3];
            rsltTn = new OclTensorF(
                context, 
                queue, 
                {inputTn->getShape()[0], inputTn->getShape()[1], inputTn->getShape()[3]}, 
                ConfigTaskReduce::BankIndex_outputTn);

        }

        mode = 3;

    }else{
        assert(false); //NYI
    }

    OclKernelObject *kernelObject = oclKernels[1];

    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, mode));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, pow_y));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim0));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim1));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim2));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim3));

    cl::Event exeEvt;
    OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
    exeEvt.wait();
    //queue->finish();

    ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

    return rsltTn;
}

//Task
TensorF* XilinxImplementation::_ReduceSum4D_Task(
        TensorF* inputTn,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3,
        int pow_y){
    return _Reduce_Task(inputTn, true, false, (unsigned)pow_y, overaxis0, overaxis1, overaxis2, overaxis3);
}

///[axis0,axis1,axis2,axis3] //Not a batch op, uses data as is
TensorF* XilinxImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){
    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
    return _ReduceSum4D(scheduler, inputTn, over_axis0, over_axis1, over_axis2, over_axis3, 1);
}

TensorF* XilinxImplementation::_ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3,
                                        int pow_y){
    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
    assert(over_axis0&&over_axis1&&over_axis2&&!over_axis3); // TTTF ONLY

    return _ReduceSum4D_Task(inputTn,over_axis0,over_axis1,over_axis2,over_axis3,pow_y);
}


/**
 * @brief      Pads the last dimension of the input tensor. Supports sub-vec and super-vec padding.
 *
 * @param[in]  scheduler      The scheduler
 * @param      inputTn        The input tn
 * @param[in]  lastDimPadded  The last dim padded
 *
 * @return     Returns the padded tensor
 */
TensorF* XilinxImplementation::PadLastDim(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        unsigned int lastDimPadded){
    PrintInfo("PadLastDim","lastDimPadded",lastDimPadded,"",0,"",0,inputTn->getShape(),{},{});
    return _PadUnpadLastDim(inputTn, true, false, lastDimPadded, 0);
}


/**
 * @brief      Unpads the padded tensor. Supports super-vec unpadding. 
 *             The input's shape[-1] should be divisable by m_axi width.
 *             lastDimUnpadded should be divisable by m_axi width.
 *
 * @param[in]  scheduler        The scheduler
 * @param      inputTn          The input tn
 * @param[in]  lastDimUnpadded  The last dim unpadded
 *
 * @return     Returns unpadded tensor
 */
TensorF* XilinxImplementation::UnpadLastDim(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        unsigned int lastDimUnpadded){
    PrintInfo("UnpadLastDim","lastDimUnpadded",lastDimUnpadded,"",0,"",0,inputTn->getShape(),{},{});
    return _PadUnpadLastDim(inputTn, false, true, 0, lastDimUnpadded);
}

TensorF* XilinxImplementation::_PadUnpadLastDim(
        TensorF* inputTn, 
        bool pad,
        bool unpad,
        unsigned lastDimPadded,
        unsigned lastDimUnpadded){

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskPadUnpad::BankIndex_inputTn);

    cl_int error; 
    int argcnt=0;
    unsigned dim0, dim1, lcm=0, _gcd;
    unsigned mode;
    OclTensorF* outputTn;
    std::vector<unsigned> shape = inputTn->getShape();
    const unsigned rank = inputTn->getRank();

    if(rank!=1){
        dim0=1;
        for(int i=0; i<rank-1; i++){
            dim0*=shape[i];
        }
        dim1=shape[rank-1];
    }else{
        dim0 = 1;
        dim1 = shape[0];
    }

    if(pad){ //PAD
        if(shape[rank-1]<CONFIG_M_AXI_WIDTH){
            //sub-vector padding
            _gcd = __gcd(dim1, CONFIG_M_AXI_WIDTH);
            lcm = (dim1*CONFIG_M_AXI_WIDTH)/(_gcd);
        }else{
            lcm=0;
        }
        shape[rank-1] = lastDimPadded;
        outputTn = new OclTensorF(context, queue, shape, ConfigTaskPadUnpad::BankIndex_outputTn);
        mode=1;

    }else if(unpad){ //UNPAD
        shape[rank-1] = lastDimUnpadded;
        outputTn = new OclTensorF(context, queue, shape, ConfigTaskPadUnpad::BankIndex_outputTn);
        //std::cout<<"dim0:"<<dim0<<", dim1:"<<dim1<<std::endl;
        mode=2;
    
    }else{
        assert(false);//NYI
    }

    OclKernelObject *kernelObject = oclKernels[9];
    std::cout<<"dim0:"<<dim0<<", dim1:"<<dim1<<
                ", lastDimPadded:"<<lastDimPadded<<
                ", lcm:"<<lcm<<
                ", lastDimUnpadded:"<<lastDimUnpadded<<
                std::endl;

    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)outputTn)->ocl_buff));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, mode));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, lastDimPadded));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, lcm));
    OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, lastDimUnpadded));

    cl::Event exeEvt;
    OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
    exeEvt.wait();
    //queue->finish();

    ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

    return outputTn;
}

TensorF* XilinxImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){
    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});

    assert(inputTn->getRank()==2 || inputTn->getRank()==4);
    assert(
            (mean_axis0 && mean_axis1 && mean_axis2 && !mean_axis3 && inputTn->getRank()==4) ||
            (mean_axis0 && !mean_axis1 && !mean_axis2 && !mean_axis3 && inputTn->getRank()==2)
    );
    bool _mean_axis0, _mean_axis1, _mean_axis2, _mean_axis3;
    if(inputTn->getRank()==4){
        _mean_axis0 = mean_axis0;
        _mean_axis1 = mean_axis1;
        _mean_axis2 = mean_axis2;
        _mean_axis3 = mean_axis3;
    }else if (inputTn->getRank()==2){
        _mean_axis0 = true;
        _mean_axis1 = true;
        _mean_axis2 = true;
        _mean_axis3 = false;
    }
    bool expanded=false;
    if (inputTn->getRank()==2){
        inputTn->ExpandDims(0);
        inputTn->ExpandDims(0);
        expanded=true;
    }

    TensorF* reducedTn = _ReduceSum4D(scheduler,inputTn,_mean_axis0,_mean_axis1,_mean_axis2,_mean_axis3,1);
    float coef = inputTn->getLength() / reducedTn->getLength(); // dim0xdim1xdim2 (for TTTF)
    TensorF* rsltTn = MatOps(scheduler,reducedTn,coef,MAT_OPS::DIV_ELEMENTWISE);

    if(expanded){
        inputTn->SqueezeDimZero();
        inputTn->SqueezeDimZero();
    }

    return rsltTn;
}

TensorF* XilinxImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){
    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});

    assert(inputTn->getRank()==2 || inputTn->getRank()==4);
    assert(
            (variance_axis0 && variance_axis1 && variance_axis2 && !variance_axis3 && inputTn->getRank()==4) ||
            (variance_axis0 && !variance_axis1 && !variance_axis2 && !variance_axis3 && inputTn->getRank()==2)
    );
    bool _variance_axis0, _variance_axis1, _variance_axis2, _variance_axis3;
    if(inputTn->getRank()==4){
        _variance_axis0 = variance_axis0;
        _variance_axis1 = variance_axis1;
        _variance_axis2 = variance_axis2;
        _variance_axis3 = variance_axis3;
    }else if (inputTn->getRank()==2){
        _variance_axis0 = true;
        _variance_axis1 = true;
        _variance_axis2 = true;
        _variance_axis3 = false;
    }
    bool expanded=false;
    if (inputTn->getRank()==2){
        inputTn->ExpandDims(0);
        inputTn->ExpandDims(0);
        expanded=true;
    }

    TensorF* tmpTn = _ReduceSum4D(scheduler,inputTn,_variance_axis0,_variance_axis1,_variance_axis2,_variance_axis3,1);
    TensorF* varianceXi2Tn = _ReduceSum4D(scheduler,inputTn,_variance_axis0,_variance_axis1,_variance_axis2,_variance_axis3,2);

    float coef = inputTn->getLength() / tmpTn->getLength(); // dim0xdim1xdim2 (for TTTF)
    TensorF* meanTn = MatOps(scheduler,tmpTn,coef,MAT_OPS::DIV_ELEMENTWISE);
    TensorF* tmp2Tn = MatOps(scheduler,varianceXi2Tn,coef,MAT_OPS::DIV_ELEMENTWISE);
    TensorF* tmp3Tn = MatOps(scheduler,meanTn,meanTn,MAT_OPS::MUL_ELEMENTWISE);
    TensorF* rsltTn = MatOps(scheduler,tmp2Tn,tmp3Tn,MAT_OPS::SUB);

    if(expanded){
        inputTn->SqueezeDimZero();
        inputTn->SqueezeDimZero();
    }

    delete(tmpTn);
    delete(tmp2Tn);
    delete(tmp3Tn);
    delete(varianceXi2Tn);
    delete(meanTn);
    return rsltTn;
}

/**
 * @brief      Performs Addition, Subtraction, Multiplication, and Division on two tensors.
 *             The first tensor could be of rank r1 where 1<=r1<=4
 *             The second tensor should be of rank r2 where 1<=r2<=r1
 *             This kernel complies with the padded last dim policy. 
 *
 * @param[in]  scheduler  The scheduler
 * @param      inputTn1   The input tn 1
 * @param      inputTn2   The input tn 2
 * @param[in]  mode       The mode
 *
 * @return     Returns the results which is a tensor with the same shape and rank of inputTn1.
 */
TensorF* XilinxImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});


    int rankDiff;
    int rank1, rank2;
    rank1 = inputTn1->getRank();
    rank2 = inputTn2->getRank();

    if(!(inputTn1->getRank()<=4 && inputTn1->getRank()>=1 && inputTn2->getRank()<=4 && inputTn2->getRank()>=1 )){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E1"<<endl;
        return nullptr;
    }

    if(inputTn1->getRank() < inputTn2->getRank()){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E2"<<endl;
        return nullptr;
    }

    //forcing inputTn1 to be of rank 4. (always)
    rankDiff = 4- inputTn1->getRank();
    while(inputTn1->getRank()<4){
        inputTn1->ExpandDimZero();
    }

    unsigned int dim0, dim1, dim2, dim3;
    unsigned int dim0B, dim1B, dim2B, dim3B;
    int dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;

    TensorF* _inputTn1 = ((OclTensorF*)inputTn1)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskMatOps::BankIndex_inputTn1);
    TensorF* _inputTn2 = ((OclTensorF*)inputTn2)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskMatOps::BankIndex_inputTn2);
    OclTensorF* rsltTn = new OclTensorF(context, queue, inputTn1->getShape(), ConfigTaskMatOps::BankIndex_outputTn);
    
    dim0 = inputTn1->getShape()[0];
    dim1 = inputTn1->getShape()[1];
    dim2 = inputTn1->getShape()[2];
    dim3 = inputTn1->getShape()[3];

    if(inputTn2->getRank()==4){
        dim0B=inputTn2->getShape()[0];
        dim1B=inputTn2->getShape()[1];
        dim2B=inputTn2->getShape()[2];
        dim3B=inputTn2->getShape()[3];
    }
    if(inputTn2->getRank()==3){
        dim0B=0                     ;
        dim1B=inputTn2->getShape()[0];
        dim2B=inputTn2->getShape()[1];
        dim3B=inputTn2->getShape()[2];
    }
    if(inputTn2->getRank()==2){
        dim0B=0;
        dim1B=0;
        dim2B=inputTn2->getShape()[0];
        dim3B=inputTn2->getShape()[1];
    }
    if(inputTn2->getRank()==1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=inputTn2->getShape()[0];
    }

    int operationMode = mode==MAT_OPS::ADD ? 0 :
                        mode==MAT_OPS::SUB ? 1 :
                        mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                        3;

    OclKernelObject *kernelObject = oclKernels[2];
    if(kernelObject->use_ndrange_kernel){

        // NOT IMPLEMENTED YET

        for(int i =0;i<rankDiff;i++){
            inputTn1->SqueezeDimZero();
        }
        return nullptr;
    }else{
        int argcnt=0;
        cl_int error;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn1)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn2)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim3));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim3B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, rank1));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, rank2));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, operationMode));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        for(int i =0;i<rankDiff;i++){
            inputTn1->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }

        return rsltTn;
    }
}

TensorF* XilinxImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),{},{});
    float* val = new float[1]; val[0] = scalar;
    OclTensorF* tmpTn = new OclTensorF();
    tmpTn->InitWithHostData(context, queue, {1}, val, ConfigTaskMatOps::BankIndex_outputTn);
    return MatOps(scheduler,inputTn1,tmpTn,mode);
}

TensorF* XilinxImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("Sqrt","",0,"",0,"",0,inputTn->getShape(),{});
    return _ReluSqrtSquare(scheduler, inputTn, false, true, false);
}

TensorF* XilinxImplementation::Concat2(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2,
        int concatDim){
    PrintInfo("Concat2","concatDim",concatDim,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

    TensorF* _inputTn1 = ((OclTensorF*)inputTn1)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskConcat::BankIndex_inputTn1);
    TensorF* _inputTn2 = ((OclTensorF*)inputTn2)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskConcat::BankIndex_inputTn2);

    unsigned int dim0,dim1,dim2,dimR3;
    unsigned int dimA3,dimB3;

    dim0 = inputTn1->getShape()[0]; 
    dim1 = inputTn1->getShape()[1]; 
    dim2 = inputTn1->getShape()[2];
    dimA3 = inputTn1->getShape()[3]; 
    dimB3 = inputTn2->getShape()[3];
    dimR3 = dimA3 + dimB3;

    OclTensorF* rsltTn = new OclTensorF(context, queue, {dim0,dim1,dim2,dimR3},ConfigTaskConcat::BankIndex_outputTn);

    OclKernelObject *kernelObject = oclKernels[0];

    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }
    else
    {
        cl_int error; int argcnt=0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn1)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn2)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim0));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim1));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dim2));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dimA3));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, dimB3));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, concatDim));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        return rsltTn;
    }
}

TensorF* XilinxImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});
    bool overaxis0=(reductionDim==0), 
        overaxis1=(reductionDim==1), 
        overaxis2=(reductionDim==2), 
        overaxis3=(reductionDim==3);
    return _Reduce_Task(inputTn, false, true, 0, overaxis0, overaxis1, overaxis2, overaxis3);
}


/**
 * @brief      Finds top k or least k elements of a tensor in the given axis and returns indices of those elements in the axis.
 *             Currently, only axis=2, tensors of rank=3, and least k mode are supported.
 *             There are three kernels for this layer: selection sort based, insertion sort based, and merge sort based.
 *             1- The kernel with selection sort uses less BRAM/URAM, instead it has higher latency.
 *             2- The kernel with insertion sort uses huge amount of resources and only applicable for tensors with small slices(less than 128 or so), 
 *                instead it has a very high throughput.
 *             3- The kernel with merge sort offers medium throughput while it uses large amounts of BRAM/URAM. 
 *                This is the preferred kernel for tensors with large slices(larger than 512 or so) 
 *             Kernels 1 and 3 have multiple PEs arranged in a systolic array alike structure.
 *             This layer is configured to comply with "last dim padded" policy.
 *             
 *
 * @param[in]  scheduler   The scheduler
 * @param      batchedMat  The input tensor
 * @param[in]  axis        
 * @param[in]  k           
 *
 * @return     The tensor with top k / least k elements' indices as its data.
 */
TensorI* XilinxImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});
    assert(batchedMat->getRank()==3);
    assert(batchedMat->getShape()[2]%CONFIG_M_AXI_WIDTH==0);
    assert(axis==2);

    auto outputShape = batchedMat->getShape();
    const auto sliceSize = outputShape[2];
    outputShape[2] = k;

    TensorF* _batchedMat = 
        ((OclTensorF*)batchedMat)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskTopK::BankIndex_inputTn);
    OclTensorI* rsltIndicesSplitedTn = new OclTensorI(context, queue, outputShape, ConfigTaskTopK::BankIndex_indicesSplitedTn);
    OclKernelObject *kernelObject = oclKernels[8];
    
    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }else{
        cl_int error; int argcnt=0; 
        const unsigned batchSize = outputShape[0]*outputShape[1];
        const unsigned vecsPerSlice = DivCeil<unsigned>(sliceSize, CONFIG_M_AXI_WIDTH);
        const unsigned vecsPerOutputSlice = DivCeil<unsigned>(k, CONFIG_M_AXI_WIDTH);
        const unsigned _dim2 = batchedMat->getShape()[2];

        cout<<"BatchSize: "<<batchSize<<endl;
        cout<<"vecsPerSlice: "<<vecsPerSlice<<endl;
        cout<<"vecsPerOutputSlice: "<<vecsPerOutputSlice<<endl;
        cout<<"_dim2: "<<_dim2<<endl;
        cout<<"k: "<<k<<endl;

        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_batchedMat)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorI*)rsltIndicesSplitedTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, batchSize));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim2));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, k));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, vecsPerSlice));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, vecsPerOutputSlice));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);
        return rsltIndicesSplitedTn;
    }
}

TensorF* XilinxImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});

    assert(inputTn->getRank()==3);
    assert(indices->getRank()==3);
    assert(inputTn->getShape()[0]==indices->getShape()[0]);
    assert(inputTn->getShape()[1]==indices->getShape()[1]);

    unsigned B,N,D,K,indicesAxis;
    B = inputTn->getShape()[0];
    N = inputTn->getShape()[1];
    D = inputTn->getShape()[2];
    K = indices->getShape()[2];
    indicesAxis = 1;

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskGather::BankIndex_inputTn);
    TensorI* _indices = ((OclTensorI*)indices)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskGather::BankIndex_indicesTn);

    OclTensorF* rsltTn = new OclTensorF(context, queue, {B,N,K,D}, ConfigTaskGather::BankIndex_outputTn);
    OclKernelObject *kernelObject = oclKernels[6];

    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }else{
        cl_int error; int argcnt=0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_indices)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, indicesAxis));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, N));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, D));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, B));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, N));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, K));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        return rsltTn;
    }
}

/**
 * @brief      Two-dimensional Convolution. 
 *             Supports only 1x1 kernels.
 *             Integrated bias addition.
 *             This layer is configured to comply with "last dim padded" policy.
 *
 * @param[in]  scheduler     The scheduler
 * @param      inputTn       The input tn
 * @param      weights       The weights
 * @param      biases        The biases
 * @param[in]  overrideDim2  ///TODO: Check this.
 *
 * @return     
 */
TensorF* XilinxImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});

    unsigned int B  = inputTn->getShape()[0];
    unsigned int N  = inputTn->getShape()[1];
    unsigned int K  = inputTn->getShape()[2];
    unsigned int D1 = inputTn->getShape()[3];
    unsigned int D2 = weights->getShape()[3];
    unsigned int D1Padded=-1;
    unsigned int D2Padded=-1;

    const unsigned int vecSizeTranspose = ConfigTaskConv2::kTransposeWidthBytes / CONFIG_DTYPE_SIZE;
    std::cout<<"vecSizeTranspose: "<< vecSizeTranspose << std::endl;

    //-----------------------------------------------------------------
    // Block 1. Padding inputTn
    // This block is disabled, as all the inputs are considered last dim padded already.(not in shape but in data layout)
    TensorF* _inputPadded = inputTn;
    /*TensorF* _inputPadded;
    if(D1<vecSizeTranspose){
        //Sub-vec Padding( 6->16)
        D1Padded = DivCeil<unsigned>(D1, vecSizeTranspose)*vecSizeTranspose;
        _inputPadded = PadLastDim(scheduler, inputTn, D1Padded);
    }else{
        assert(D1%vecSizeTranspose==0);
        D1Padded = D1;
        _inputPadded = inputTn;
    }*/

    //-----------------------------------------------------------------
    // Block 2. Padding weightTn
    TensorF* _weightPadded;
    if(D2%ConfigTaskConv2::kOuterTileSizeM!=0){
        //Super-vec Padding( 64->128 )
        D2Padded = DivCeil<unsigned>(D2, ConfigTaskConv2::kOuterTileSizeM)*ConfigTaskConv2::kOuterTileSizeM;
        _weightPadded = PadLastDim(scheduler, weights, D2Padded);

        // The kernel is modified to not require the weight tensor to be 
        // padded in dimension-zero.
        std::cout<< "Padding weightTn(super-vec padding):" << std::endl;
        std::cout<<"\tD2: "<< D2 << std::endl;
        std::cout<<"\tD2Padded: "<< D2Padded << std::endl;
    }else{
        std::cout<< "Bypassing super-vec padding for weightTn" << std::endl;
        _weightPadded = weights;
        D2Padded = D2;
    }

    //-----------------------------------------------------------------
    // Block 3. Padding biasTn (?) NYI
    ///TODO: is this needed?

    //-----------------------------------------------------------------
    // Block 4. Crossing DDR Banks
    TensorF* _inputTn = ((OclTensorF*)_inputPadded)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskConv2::BankIndex_inputTn);
    TensorF* _weights = ((OclTensorF*)_weightPadded)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskConv2::BankIndex_weightTn);
    TensorF* _biases = ((OclTensorF*)biases)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskConv2::BankIndex_biasTn);

    //-----------------------------------------------------------------
    OclTensorF* rsltTnPadded = new OclTensorF(context, queue,
                                            {B,
                                             N,
                                             K,
                                             D2Padded},
                                             ConfigTaskConv2::BankIndex_outputTn);

    OclKernelObject *kernelObject = oclKernels[7];

    if(kernelObject->use_ndrange_kernel){
        //NYI
        return nullptr;
    }else{
        const unsigned size_n = B*N*K;
        const unsigned size_k = D1; //Should be the original shape, not the padded one.
        const unsigned size_m = D2Padded;
        cl_int error; int argcnt=0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_weights)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_biases)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTnPadded)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, size_n));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, size_k));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, size_m));

        std::cout << "Launching Conv2 gemmHLS\n";
        std::cout << "[N K M] = ["<< size_n << " " << size_k << " " << size_m <<"]\n";

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);
        std::cout << "Launching Conv2 gemmHLS...DONE\n";

        //-----------------------------------------------------------------
        // Block 5. Unpadding rsltTnPadded
        TensorF* rsltTn;
        if(D2%ConfigTaskConv2::kOuterTileSizeM!=0){ //same as Block2
            //Super-vec Unpadding( 128->64 )
            rsltTn = UnpadLastDim(scheduler, rsltTnPadded, D2);
            std::cout<< "Unpadding results(super-vec unpadding)." << std::endl;
        }else{
            rsltTn = rsltTnPadded;
            std::cout<< "Bypassing super-vec unpadding of results." << std::endl;
        }        

        //-----------------------------------------------------------------
        return rsltTn;
    }
}

TensorF* XilinxImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});
    return _ReluSqrtSquare(scheduler, inputTn, true, false, false);
}

TensorF* XilinxImplementation::_ReluSqrtSquare(WorkScheduler scheduler, TensorF* inputTn, bool runRelu, bool runSqrt, bool runSquare){
    PrintInfo("ReluSqrtSquare","runRelu",runRelu,"runSqrt",runSqrt,"runSquare",runSquare,inputTn->getShape(),{},{});
    assert(
        (runRelu&& !runSqrt&& !runSquare)||
        (!runRelu&& runSqrt&& !runSquare)||
        (!runRelu&& !runSqrt&& runSquare));
    assert(inputTn->getLength()!=0);

    const unsigned mode = runRelu?ConfigTaskReluSqrtSquare::ModeRelu:
                          runSqrt?ConfigTaskReluSqrtSquare::ModeSqrt:
                          runSquare?ConfigTaskReluSqrtSquare::ModeSquare:
                          100;

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskReluSqrtSquare::BankIndex_inputTn);
    OclTensorF *rsltTn = new OclTensorF(context, queue, inputTn->getShape(),ConfigTaskReluSqrtSquare::BankIndex_outputTn);
    
    OclKernelObject *kernelObject = oclKernels[10];

    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }else{
        cl_int error; int argcnt=0;
        cl_uint len = (cl_uint)inputTn->getVectorCountPadded(CONFIG_M_AXI_WIDTH);
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, len));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, mode));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        return rsltTn;
    }
}


/**
 * @brief      Performs tiling operation on the input tensor.
 *             Currently, only the combinations below are supported(T=tileCount):
 *                  1) Input(Expanded, Rank=2, Shape=BxN),   tileAxis=1, Output(Shape=BxTxN)
 *                  2) Input(Expanded, Rank=2, Shape=BxN),   tileAxis=2, Output(Shape=BxNxT)
 *                  3) Input(Expanded, Rank=3, Shape=BxNxD), tileAxis=2, Output(Shape=BxNxTxD)
 *
 * @param[in]  scheduler  The scheduler
 * @param      inputTn    The input tn
 * @param[in]  tileAxis   The tile axis
 * @param[in]  tileCount  The tile count
 *
 * @return     
 */
TensorF* XilinxImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    PrintInfo("Tile","tileAxis",tileAxis,"tileCount",tileCount,"",0,inputTn->getShape(),{},{});

    unsigned rank = inputTn->getRank();
   
    assert(
        (rank==2 && tileAxis==1) ||
        (rank==2 && tileAxis==2) ||
        (rank==3 && tileAxis==2)
        );

    if(rank==2){
        assert(
            (rank==2 && tileAxis==1) ||
            (rank==2 && tileAxis==2)
            );
    }
    if(rank==3){
        assert(rank==3 && tileAxis==2);
    }

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneIfNeededToDDRBank(program,context,queue,ConfigTaskTile::BankIndex_inputTn);

    // Eliminating expaded dimensions manually.
    unsigned _dim0=0,_dim1=0,_dim2=0;
    OclTensorF* rsltTn = nullptr;

    if(rank==2){
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        if(tileAxis==1){
            rsltTn= new OclTensorF(context, queue, {_dim0,(unsigned int)tileCount,_dim1},ConfigTaskTile::BankIndex_outputTn);
        }else if(tileAxis==2){
            rsltTn= new OclTensorF(context, queue, {_dim0,_dim1,(unsigned int)tileCount},ConfigTaskTile::BankIndex_outputTn);
        }else{
            // Something is not right.
            assert(false);
        }
        
    }else if(rank==3 && tileAxis==2){
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        _dim2 = inputTn->getShape()[2];
        rsltTn= new OclTensorF(context, queue, {_dim0,_dim1,(unsigned)tileCount,_dim2},ConfigTaskTile::BankIndex_outputTn);
    }else{
        // Something is not right.
        assert(false);
    }

    OclKernelObject *kernelObject = oclKernels[3];

    if(kernelObject->use_ndrange_kernel){
        return nullptr;
    }else{
        cl_int error; int argcnt=0;
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)_inputTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, ((OclTensorF*)rsltTn)->ocl_buff));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim0));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim1));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, _dim2));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, rank));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, tileAxis));
        OCL_CHECK(error, error = kernelObject->kernel_task->setArg(argcnt++, tileCount));

        cl::Event exeEvt;
        OCL_CHECK(error,error = queue->enqueueTask(*kernelObject->kernel_task, nullptr, &exeEvt));
        exeEvt.wait();
        //queue->finish();

        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        return rsltTn;
    }
}

void XilinxImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorF* inputTn,
        string npy_dir){
    TensorF *_inputTn = ((OclTensorF*)inputTn)->TransferToHost(queue);
#ifdef DUMP_ENABLED
        vector<unsigned int> shape = _inputTn->getShape();
        vector<unsigned long > shape_size_t(shape.begin(), shape.end());
        cnpy::npy_save<float>(npy_dir+npy_fname,_inputTn->_buff ,shape_size_t,"w");
#endif
}

void XilinxImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorI* inputTn,
        string npy_dir){
    TensorI *_inputTn = ((OclTensorI*)inputTn)->TransferToHost(queue);
#ifdef DUMP_ENABLED
        vector<unsigned int> shape = _inputTn->getShape();
        vector<unsigned long  > shape_size_t(shape.begin(), shape.end());
        cnpy::npy_save<int>(npy_dir+npy_fname,_inputTn->_buff ,shape_size_t,"w");
#endif
}

bool XilinxImplementation::CompareTensors(WorkScheduler scheduler,TensorF *inputTn1, TensorF *inputTn2) {
    return false;
}

bool XilinxImplementation::CompareTensorsInteger(WorkScheduler scheduler,TensorI *inputTn1, TensorI *inputTn2) {
    return false;
}

const char * XilinxImplementation::getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

cl_ulong XilinxImplementation::get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    OCL_CHECK(err,err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart));
    OCL_CHECK(err,err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend));
    return(nstimeend-nstimestart);
}

void XilinxImplementation::ReportDuration(const std::string &name, const bool &isNDRange, const cl::Event &event){
    uint64_t ns = get_duration_ns(event);
#ifdef REPORT_EXECUTION_DURATION
        std::cout<< "\t** "<< name << (isNDRange?"(ndrange)":"(task)")<<":: "<<
                "\t(us): " << ns/1000.0f <<
                "\t(ms): " << ns/1000000.0f <<
                "\t( s): " << ns/1000000000.0f <<
                std::endl;
#endif
}

int XilinxImplementation::SetModeEnvVar(const RUN_MODE mode){
    int result = 0;
    if(mode==RUN_MODE::Unknown) return -2;
    const char* strMode = mode==RUN_MODE::SwEmu? "sw_emu":
                          mode==RUN_MODE::HwEmu? "hw_emu":
                          "system";
    result = setenv("XCL_EMULATION_MODE", strMode, 1); // Env var override is enabled.

    if(result<0){
        cerr<<"SetModeEnvVar: Error setting XCL_EMULATION_MODE env. var."<<endl;
    }
    return result;
}

RUN_MODE XilinxImplementation::GetModeEnvVar(){
    if(const char *_xcl_mode = getenv("XCL_EMULATION_MODE")){
        const string xcl_mode = string(_xcl_mode);
        RUN_MODE mode =  xcl_mode=="sw_emu" ? RUN_MODE::SwEmu:
                    xcl_mode=="hw_emu" ? RUN_MODE::HwEmu:
                    xcl_mode=="system" ? RUN_MODE::Hw: 
                    RUN_MODE::Unknown ;
        return mode;
    }else{
        return RUN_MODE::Unknown;
    }
}