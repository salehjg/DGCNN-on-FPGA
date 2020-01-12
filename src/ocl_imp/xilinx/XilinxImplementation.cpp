//
// Created by saleh on 8/22/18.
//

#include <iostream>
#include <assert.h>
#include <ocl_imp/xilinx/XilinxImplementation.h>

using namespace std;

#define DISABLED_KERNEL (true)

XilinxImplementation::XilinxImplementation(int aa) {
    a = aa;
    //======================================================================================================================
    {
        err = clGetPlatformIDs(1, &cpPlatform, NULL);
        assert(err == CL_SUCCESS);
        err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
        assert(err == CL_SUCCESS);
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        queue = clCreateCommandQueue(
                context,
                device_id,
#ifdef REPORT_EXECUTION_DURATION
                CL_QUEUE_PROFILING_ENABLE,
#else
                0,
#endif
                &err);
        assert(err == CL_SUCCESS);
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
                "/xilinx/sqrt.cpp",
                "binary_container_1.xclbin",
                "",
                "task_sqrt",
                false),
        /* IDX 2 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/reducemax.cpp",
                "binary_container_1.xclbin",
                "",
                "task_reducemax",
                false),
        /* IDX 3 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/reducesum4d.cpp",
                "binary_container_1.xclbin",
                "",
                "task_reducesum4d",
                false),
        /* IDX 4 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/reducesum.cpp",
                "binary_container_1.xclbin",
                "",
                "task_reducesum",
                false),
        /* IDX 5 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/matops.cpp",
                "binary_container_1.xclbin",
                "",
                "task_matops",
                false),
        /* IDX 6 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/relu.cpp",
                "binary_container_1.xclbin",
                "",
                "task_relu",
                false),
        /* IDX 7 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/square.cpp",
                "binary_container_1.xclbin",
                "",
                "task_square",
                false),
        /* IDX 8 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/tile.cpp",
                "binary_container_1.xclbin",
                "",
                "task_tile",
                false),
        /* IDX 9 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/matmul.cpp",
                "binary_container_1.xclbin",
                "",
                "task_matmul",
                false),
        /* IDX 10 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/transpose.cpp",
                "binary_container_1.xclbin",
                "",
                "task_transpose",
                false),
        /* IDX 11 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/gather.cpp",
                "binary_container_1.xclbin",
                "",
                "task_gather",
                false),

        /* IDX 12 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/conv2_1x1_direct.cpp",
                "binary_container_1.xclbin",
                "",
                "task_conv2_1x1_direct",
                false,
                DISABLED_KERNEL),
        /* IDX 13 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/topk.cpp",
                "binary_container_1.xclbin",
                "",
                "task_topk",
                false,
                DISABLED_KERNEL),
//      /* IDX 13 :*/
//      new OclKernelObject(
//              KERNEL_DIR,
//              "/xilinx/splitinteger.cl",
//              "binary_container_1.xclbin",
//              "",
//              "task_split_integer",
//              false),
//      /* IDX 14 :*/
//      new OclKernelObject(
//              KERNEL_DIR,
//              "/xilinx/splitfloat.cl",
//              "binary_container_1.xclbin",
//              "",
//              "task_split_float",
//              false),

    };
    
    //======================================================================================================================
    //Using signle binary container for all of the kernels for now!
    char *_xcl_mode = getenv("XCL_EMULATION_MODE");
    string xcl_mode = string(_xcl_mode);
    xcl_mode =  xcl_mode=="sw_emu" ? "Emulation-SW/" :
                xcl_mode=="hw_emu" ? "Emulation-HW/" :
                xcl_mode=="system" ? "System/" : "UNDEF" ;

    //cout<<xcl_mode<<endl;

    cout<<"*Using first kernel's container as default container.\n*Multiple container scenario is not supported yet."<<endl;
    size_t binary_content_length = load_file_to_memory(globalArgXclBin, &binary_content);

    program = clCreateProgramWithBinary(
                            context, 
                            1,
                            &device_id,
                            &binary_content_length,
                            (const unsigned char**) &binary_content,
                            NULL,
                            &err);

    if (err != CL_SUCCESS) {
        cout<<"Failed to create OpenCL program from binary."<<endl;
        std::exit(1);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        cout<<"Failed on clBuildProgram."<<endl;
        std::exit(1);
    }
 
    for(OclKernelObject *kernelObject : oclKernels){
        if(kernelObject->disabled) continue;
        if(kernelObject->use_ndrange_kernel){
            //OCL_CHECK(err, kernelObject->kernel_ndrange = new cl::Kernel(program, kernelObject->kernelName_ndrange, &err));
            kernelObject->kernel_ndrange = clCreateKernel(program, kernelObject->kernelName_ndrange, &err);
            if (err != CL_SUCCESS) {
                printf("Failed to create ndrange kernel %d\n", (int) err);
                std::exit(1);
            }

        }else{
            // OCL_CHECK(err, kernelObject->kernel_task = new cl::Kernel(program, kernelObject->kernelName_task, &err));
            kernelObject->kernel_task = clCreateKernel(program, kernelObject->kernelName_task, &err);
            if (err != CL_SUCCESS) {
                printf("Failed to create task kernel %d\n", (int) err);
                std::exit(1);
            }
        }
    }

    //Cannot release these two just yet, because OclTensorF and OclTensorI classes will be needing them for datamover kernel-
    //creation.
    //clReleaseProgram(program);
    //free(binary_content);

    std::cout<<"- - - - - - - - - - -"<<std::endl;
    
    //======================================================================================================================
}

XilinxImplementation::~XilinxImplementation(){
    cout<<"~XilinxImplementation"<<endl;
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    for(OclKernelObject *kernelObject : oclKernels){
        if(kernelObject->use_ndrange_kernel)
            clReleaseKernel(kernelObject->kernel_ndrange);
        else
            clReleaseKernel(kernelObject->kernel_task);
    }
}

cl_context XilinxImplementation::getContext(){
    return context;
}

cl_program XilinxImplementation::getProgram() {
    return program;
}

cl_command_queue XilinxImplementation::getQueue() {
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

void XilinxImplementation::GetPaddedWorkSize(
        int dims,
        size_t * inBlockSize,
        size_t * inWorkSize,
        size_t * outPaddedWorkSize){
    for(int i = 0; i < dims; i++){
        outPaddedWorkSize[i] = (inWorkSize[i] + inBlockSize[i] - 1 ) / (inBlockSize[i]);
        outPaddedWorkSize[i] *= inBlockSize[i];
    }
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

    OclTensorF *rsltTn = new OclTensorF(context,{dim0,dim2,dim1});
    OclKernelObject *kernelObject = oclKernels[10];

    if(kernelObject->use_ndrange_kernel){

    }else{
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_uint), (void*)&dim0);
        error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void*)&dim1);
        error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void*)&dim2);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue, kernelObject->kernel_task, 0, NULL, &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        if(rankDiff) batchedMat->SqueezeDimZero();
        return rsltTn;
    }
    return nullptr;
}

TensorF* XilinxImplementation::MatMul(WorkScheduler scheduler,
                                   TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());
    assert(batchedMat1->getRank()==3 || batchedMat1->getRank()==2);
    assert(batchedMat2->getRank()==3 || batchedMat2->getRank()==2);
    assert(batchedMat2->getRank()==batchedMat2->getRank());

    unsigned int dim0A,dim1A,dim2A,dim0B,dim1B,dim2B;
    int rankDiff;

    rankDiff = 3 - batchedMat1->getRank();
    for(int i=0;i<rankDiff;i++){
        batchedMat1->ExpandDimZero();
        batchedMat2->ExpandDimZero();
    }

    dim0A = batchedMat1->getShape()[0]; // batch size
    dim1A = batchedMat1->getShape()[1]; // height of matrix
    dim2A = batchedMat1->getShape()[2]; // width of matrix

    dim0B = batchedMat2->getShape()[0]; // batch size
    dim1B = batchedMat2->getShape()[1]; // height of matrix
    dim2B = batchedMat2->getShape()[2]; // width of matrix

    //Width of A should be equal to the Height of B. (dim2A = dim1B)
    assert(dim0A == dim0B);
    assert(dim2A == dim1B);

    TensorF* _batchedMat1 = ((OclTensorF*)batchedMat1)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorF* _batchedMat2 = ((OclTensorF*)batchedMat2)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);

    OclTensorF*rsltTn = new OclTensorF(context,{dim0A,dim1A, dim2B}, DATAMOVER_KERNEL_BANK_B_INDEX);
    OclKernelObject *kernelObject = oclKernels[9];

    if(kernelObject->use_ndrange_kernel){

        //NOT IMPLEMENTED YET.

        for(int i=0;i<rankDiff;i++){
            batchedMat1->SqueezeDimZero();
            batchedMat2->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }

        return rsltTn;
    }else{
        cl_int error; int argcnt=0;
        error  = clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&((OclTensorF*)_batchedMat1)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&((OclTensorF*)_batchedMat2)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&(rsltTn->ocl_buff));
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim0A);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim1A);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim2A);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim0B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim1B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim2B);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;

        error = clEnqueueTask( queue,
                               kernelObject->kernel_task,
                               0,
                               NULL,
                               &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        for(int i=0;i<rankDiff;i++){
            batchedMat1->SqueezeDimZero();
            batchedMat2->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }

        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }

}

TensorF* XilinxImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});

    assert(batchedMat->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,batchedMat->getShape());
    OclKernelObject *kernelObject = oclKernels[7];

    if(kernelObject->use_ndrange_kernel){

    }else{
        cl_int error;
        cl_ulong len = batchedMat->getVectorCountPadded(CONFIG_M_AXI_WIDTH);
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_ulong), (void*)&len);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue, kernelObject->kernel_task, 0, NULL, &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
          printf("Kernel execution failure!\n");
          exit(-22);
        }

        return rsltTn;
    }

    return nullptr;
}

TensorF* XilinxImplementation::ReduceSum(WorkScheduler scheduler,
                                      TensorF* inputTn,
                                      bool over_axis0,
                                      bool over_axis1,
                                      bool over_axis2){
    PrintInfo("ReduceSum","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2});
    unsigned int _dim0,_dim1,_dim2;
    int _overAxis0, _overAxis1, _overAxis2;

    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];

    _overAxis0 = over_axis0;
    _overAxis1 = over_axis1;
    _overAxis2 = over_axis2;

    OclTensorF* rsltTn ;
    if(inputTn->getRank()==3 &&  !over_axis0 && !over_axis1 && over_axis2)rsltTn= new OclTensorF(context, {_dim0,_dim1});
    if(inputTn->getRank()==3 &&  !over_axis0 && over_axis1 && !over_axis2)rsltTn= new OclTensorF(context, {_dim0,_dim2});
    if(inputTn->getRank()==3 &&  over_axis0 && !over_axis1 && !over_axis2)rsltTn= new OclTensorF(context, {_dim1,_dim2});


    if(inputTn->getRank()==2 &&  !over_axis0 && over_axis1 )rsltTn= new OclTensorF(context, {_dim0});

    /*
    reduce_sum_3d_try03(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            over_axis0,
            over_axis1,
            over_axis2);
    */

    OclKernelObject *kernelObject = oclKernels[4];

    if(kernelObject->use_ndrange_kernel){

    }else{
        cl_int error;
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_uint), (void*)&_dim0);
        error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void*)&_dim1);
        error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void*)&_dim2);
        error |= clSetKernelArg(kernelObject->kernel_task, 5, sizeof(cl_int), (void*)&_overAxis0);
        error |= clSetKernelArg(kernelObject->kernel_task, 6, sizeof(cl_int), (void*)&_overAxis1);
        error |= clSetKernelArg(kernelObject->kernel_task, 7, sizeof(cl_int), (void*)&_overAxis2);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue, kernelObject->kernel_task, 0, NULL, &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        return rsltTn;
    }
    return nullptr;
}

int XilinxImplementation::_ReduceSum4D_Try05_NDRange_Find_Kernel_Launches_Needed(int sliceCount, int SPT, int TGPB){
    int i=0, sliceLeft=sliceCount,p=sliceCount,q=SPT*TGPB;
    int LIMIT=50;
    for(i=0;i<LIMIT;i++){
        if(i==0){
            sliceLeft = ( p + (q-1) ) / q;
        }else{
            sliceLeft = ( sliceLeft + (q-1) ) / q;
        }
        if(sliceLeft==1){
            return i;
        }
    }
    return -1;
}


//NDRange
void XilinxImplementation::_ReduceSum4D_Try05_NDRange(
        TensorF* inputTn,
        TensorF* outputTn,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3,
        int pow_y){
    //assert(over_axis0&&over_axis1&&over_axis2&&!over_axis3); // TTTF ONLY

    //OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF
    OclKernelObject *kernelObject = oclKernels[3];

    unsigned long SPT,TGC,TGO,TGPB,TPG;
    unsigned int BLOCK_SIZE = XILINX_BOTTLENCK_BLOCKSIZE;

    //Dim3 slice per thread
    SPT = 2048; //cte

    //thread group offset
    TGO = dim3 * SPT;

    //thread group count
    TGC = (unsigned long)((dim0*dim1*dim2+(SPT-1))/SPT);

    //thread group per block
    TGPB = (unsigned long)((BLOCK_SIZE)/ dim3);
    if(TGPB%2 && TGPB > 1) TGPB--;

    TPG = (unsigned long)dim3; //threads per group

    unsigned long grid = ( TGC+(TGPB-1) ) / TGPB;
    size_t global_work_size[] = {grid*(BLOCK_SIZE)};
    size_t global_padded_work_size[1];
    size_t local_block_size[] = {BLOCK_SIZE};
    unsigned long shared_mem_size = (TGPB*TPG)*sizeof(cl_float);
    GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);


    cout<< "LOCAL:      " << local_block_size[0] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << "\n";


    OclTensorF* g_buffer1, *g_buffer2;
    g_buffer1 = new OclTensorF(context, {grid * dim3});
    g_buffer2 = new OclTensorF(context, {grid * dim3});

    cl_int error;
    cl_int _overAxis0,_overAxis1,_overAxis2,_overAxis3;
    _overAxis0 = overaxis0;
    _overAxis1 = overaxis1;
    _overAxis2 = overaxis2;
    _overAxis3 = overaxis3;

    long iLast = _ReduceSum4D_Try05_NDRange_Find_Kernel_Launches_Needed(dim0*dim1*dim2,SPT,TGPB) ;
    int grid_old=0,_pow_y=pow_y;
    cl_event exeEvt;
    long _limit = dim0*dim1*dim2;

    for(long i=0;i<=(iLast);i++){
        printf("i=%d of %d\n",i,iLast);
        printf("launching kernel_reduce_sum_4d_try05...\n");
        if(i>0){
            _pow_y=1;
            _limit=grid_old;
        }
        error  = clSetKernelArg(kernelObject->kernel_ndrange, 0, sizeof(cl_mem), (void *) &((i==0)    ? (OclTensorF *)inputTn : (i%2)?g_buffer1:g_buffer2)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 1, sizeof(cl_mem), (void *) &((i==iLast)? (OclTensorF *)outputTn: (i%2)?g_buffer2:g_buffer1)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 2, shared_mem_size, NULL);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 3, sizeof(cl_int), (void *) &_pow_y);

        error |= clSetKernelArg(kernelObject->kernel_ndrange, 4, sizeof(cl_ulong), (void *) &_limit);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 5, sizeof(cl_ulong), (void *) &(dim3));

        error |= clSetKernelArg(kernelObject->kernel_ndrange, 6, sizeof(cl_int), (void *) &_overAxis0);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 7, sizeof(cl_int), (void *) &_overAxis1);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 8, sizeof(cl_int), (void *) &_overAxis2);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 9, sizeof(cl_int), (void *) &_overAxis3);

        error |= clSetKernelArg(kernelObject->kernel_ndrange, 10, sizeof(cl_ulong), (void *) &TGC);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 11, sizeof(cl_ulong), (void *) &TGPB);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 12, sizeof(cl_ulong), (void *) &SPT);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 13, sizeof(cl_ulong), (void *) &TGO);


        if (error != CL_SUCCESS) cout << getErrorString(error) << endl;
        assert(error == 0);

        error = clEnqueueNDRangeKernel(queue,
                                       kernelObject->kernel_ndrange,
                                       1,
                                       NULL,
                                       global_padded_work_size,
                                       local_block_size,
                                       0,
                                       NULL,
                                       &exeEvt);

        if (error != CL_SUCCESS) cout << getErrorString(error) << endl;

        TGC = (unsigned long)((TGC+(SPT-1))/SPT);
        grid_old = grid;
        grid = ( TGC+(TGPB-1) ) / TGPB;
        global_work_size[0] = grid*(BLOCK_SIZE);
        GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);
        printf("========================\n");
        printf("KERNEL_TGC_NEXT   :   %ld\n", TGC);
        printf("KERNEL_GRID_NEXT  :   %ld\n", grid);
    }


    clWaitForEvents(1, &exeEvt);
    ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }
    delete(g_buffer1);
    delete(g_buffer2);

}

//Task
void XilinxImplementation::_ReduceSum4D_Task(
        TensorF* inputTn,
        TensorF* outputTn,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3,
        int pow_y){
    OclKernelObject *kernelObject = oclKernels[3];

    cl_event exeEvt;
    cl_int error;
    cl_int _overAxis0,_overAxis1,_overAxis2,_overAxis3;

    _overAxis0 = overaxis0;
    _overAxis1 = overaxis1;
    _overAxis2 = overaxis2;
    _overAxis3 = overaxis3;

    error  = clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void *) &((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void *) &((OclTensorF*)outputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_int), (void *) &pow_y);

    error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void *) &dim0);
    error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void *) &dim1);
    error |= clSetKernelArg(kernelObject->kernel_task, 5, sizeof(cl_uint), (void *) &dim2);
    error |= clSetKernelArg(kernelObject->kernel_task, 6, sizeof(cl_uint), (void *) &dim3);

    error |= clSetKernelArg(kernelObject->kernel_task, 7, sizeof(cl_int), (void *) &_overAxis0);
    error |= clSetKernelArg(kernelObject->kernel_task, 8, sizeof(cl_int), (void *) &_overAxis1);
    error |= clSetKernelArg(kernelObject->kernel_task, 9, sizeof(cl_int), (void *) &_overAxis2);
    error |= clSetKernelArg(kernelObject->kernel_task, 10, sizeof(cl_int), (void *) &_overAxis3);



    if (error != CL_SUCCESS) cout << getErrorString(error) << endl;
    assert(error == 0);

    error = clEnqueueTask(queue,kernelObject->kernel_task,0,NULL,&exeEvt);
    if (error != CL_SUCCESS) cout << getErrorString(error) << endl;

    clWaitForEvents(1, &exeEvt);
    ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    } 
}

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is
TensorF* XilinxImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){
    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
    assert(over_axis0&&over_axis1&&over_axis2&&!over_axis3); // TTTF ONLY

    OclKernelObject *kernelObject = oclKernels[3];

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF

    if(kernelObject->use_ndrange_kernel){
        _ReduceSum4D_Try05_NDRange(inputTn,rsltTn,_dim0,_dim1,_dim2,_dim3,over_axis0,over_axis1,over_axis2,over_axis3,1);
        return rsltTn;
    }else{
        _ReduceSum4D_Task(inputTn,rsltTn,_dim0,_dim1,_dim2,_dim3,over_axis0,over_axis1,over_axis2,over_axis3,1);
        return rsltTn;
    }
    return nullptr;
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

    OclKernelObject *kernelObject = oclKernels[3];

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF

    if(kernelObject->use_ndrange_kernel){
        _ReduceSum4D_Try05_NDRange(inputTn,rsltTn,_dim0,_dim1,_dim2,_dim3,over_axis0,over_axis1,over_axis2,over_axis3,pow_y);
        return rsltTn;
    }else{
        _ReduceSum4D_Task(inputTn,rsltTn,_dim0,_dim1,_dim2,_dim3,over_axis0,over_axis1,over_axis2,over_axis3,pow_y);
        return rsltTn;
    }
    return nullptr;
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

TensorF* XilinxImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});


    int rankDiff;

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

    unsigned long indxS1;
    unsigned long indxS2;
    unsigned int dim0, dim1, dim2, dim3;
    unsigned int dim0B, dim1B, dim2B, dim3B;
    int dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;

    TensorF* _inputTn1 = ((OclTensorF*)inputTn1)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorF* _inputTn2 = ((OclTensorF*)inputTn2)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    OclTensorF* rsltTn = new OclTensorF(context, inputTn1->getShape(), DATAMOVER_KERNEL_BANK_B_INDEX);
    
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
    if(inputTn2->getRank()==1 && inputTn2->getShape()[0]!=1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=inputTn2->getShape()[0];
    }
    if(inputTn2->getShape()[0]==1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=1; //and rank should be 1 which already is
    }


    int tmp =15>>(4-inputTn2->getRank());
    dim0B_IsNotZero = (tmp >> 3) & 1;
    dim1B_IsNotZero = (tmp >> 2) & 1;
    dim2B_IsNotZero = (tmp >> 1) & 1;
    dim3B_IsNotZero = (tmp >> 0) & 1;

    if(inputTn2->getRank()==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dim3B_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }
    int operationMode = mode==MAT_OPS::ADD ? 0 :
                        mode==MAT_OPS::SUB ? 1 :
                        mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                        3;

    OclKernelObject *kernelObject = oclKernels[5];
    if(kernelObject->use_ndrange_kernel){

        // NOT IMPLEMENTED YET

        for(int i =0;i<rankDiff;i++){
            inputTn1->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }
        return rsltTn;
    }else{
        int argcnt=0;
        cl_int error;
        error =  clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&((OclTensorF*)_inputTn1)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&((OclTensorF*)_inputTn2)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem), (void*)&(rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim0);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim1);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim2);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim3);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim0B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim1B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim2B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_uint), (void*)&dim3B);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_int), (void*)&dim0B_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_int), (void*)&dim1B_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_int), (void*)&dim2B_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_int), (void*)&dim3B_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_int), (void*)&operationMode);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue, kernelObject->kernel_task, 0, NULL, &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        for(int i =0;i<rankDiff;i++){
            inputTn1->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }

        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }





    return nullptr;
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
    tmpTn->InitWithHostData(context, queue, {1}, val);
    return MatOps(scheduler,inputTn1,tmpTn,mode);
}

TensorF* XilinxImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("Sqrt","",0,"",0,"",0,inputTn->getShape(),{});
    assert(inputTn->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());
    OclKernelObject *kernelObject = oclKernels[1];

    if(kernelObject->use_ndrange_kernel){
        
    }
    else
    {
        cl_int error;
        cl_ulong len = inputTn->getVectorCountPadded(CONFIG_M_AXI_WIDTH);

        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_ulong), (void*)&len);

        cl_event exeEvt;

        //Launch the kernel
        error = clEnqueueTask( queue,
                               kernelObject->kernel_task,
                               0, 
                               NULL,
                               &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;

        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);
        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }
        return rsltTn;
    }
    return nullptr;
}

///concat 2 matrices
/// [matA, matB]
TensorF* XilinxImplementation::Concat2(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2,
        int concatDim){
    PrintInfo("Concat2","concatDim",concatDim,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

    TensorF* _inputTn1 = ((OclTensorF*)inputTn1)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorF* _inputTn2 = ((OclTensorF*)inputTn2)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);


    unsigned int dimA0,dimA1,dimA2,dimA3;
    unsigned int dimB0,dimB1,dimB2,dimB3;
    dimA0 = inputTn1->getShape()[0]; dimB0 = inputTn2->getShape()[0];
    dimA1 = inputTn1->getShape()[1]; dimB1 = inputTn2->getShape()[1];
    dimA2 = inputTn1->getShape()[2]; dimB2 = inputTn2->getShape()[2];
    dimA3 = inputTn1->getShape()[3]; dimB3 = inputTn2->getShape()[3];
    OclTensorF* rsltTn = new OclTensorF(context, {dimA0,dimA1,dimA2,dimA3+dimB3},DATAMOVER_KERNEL_BANK_B_INDEX);

    OclKernelObject *kernelObject = oclKernels[0];

    if(kernelObject->use_ndrange_kernel){

    }
    else
    {
        cl_int error;
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem) , (void*)&((OclTensorF*)_inputTn1)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem) , (void*)&((OclTensorF*)_inputTn2)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);

        error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void*)&dimA0);
        error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void*)&dimA1);
        error |= clSetKernelArg(kernelObject->kernel_task, 5, sizeof(cl_uint), (void*)&dimA2);
        error |= clSetKernelArg(kernelObject->kernel_task, 6, sizeof(cl_uint), (void*)&dimA3);

        error |= clSetKernelArg(kernelObject->kernel_task, 7, sizeof(cl_uint), (void*)&dimB0);
        error |= clSetKernelArg(kernelObject->kernel_task, 8, sizeof(cl_uint), (void*)&dimB1);
        error |= clSetKernelArg(kernelObject->kernel_task, 9, sizeof(cl_uint), (void*)&dimB2);
        error |= clSetKernelArg(kernelObject->kernel_task, 10,sizeof(cl_uint), (void*)&dimB3);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;

        error = clEnqueueTask( queue,
                               kernelObject->kernel_task,
                               NULL,
                               NULL,
                               &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }


        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }

    return nullptr;
}

TensorF* XilinxImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});
    assert(inputTn->getRank()==4);

    size_t kGrid;
    int kDim0,kDim1,kDim2;
    int overAxis0, overAxis1, overAxis2;
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  reductionDim==1)rsltTn= new OclTensorF(context, {_dim0,_dim2,_dim3});
    if(inputTn->getRank()==4 &&  reductionDim==2)rsltTn= new OclTensorF(context, {_dim0,_dim1,_dim3});

    if(reductionDim==2){
        kDim0 = _dim0*_dim1;
        kDim1 = _dim2;
        kDim2 = _dim3;
        kGrid = kDim0*kDim2;

        overAxis0 = 0;
        overAxis1 = 1;
        overAxis2 = 0;
    }

    if(reductionDim==1){
        if(_dim2!=1){
            cout<<"ReduceMax: For reductionDim=1, Dim2 should be equals 1."<<endl;
            return nullptr;
        }
        kDim0 = _dim0;
        kDim1 = _dim1;
        kDim2 = _dim3;
        kGrid = kDim0*kDim2;

        overAxis0 = 0;
        overAxis1 = 1;
        overAxis2 = 0;
    }

    OclKernelObject *kernelObject = oclKernels[2];
    
    if(kernelObject->use_ndrange_kernel){
        cl_int error;
        
        error =  clSetKernelArg(kernelObject->kernel_ndrange, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 1 , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 2 , sizeof(cl_uint) , (void*)&kDim2);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 3 , sizeof(cl_uint) , (void*)&kDim1);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 4 , sizeof(cl_uint) , (void*)&kDim0);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 5 , sizeof(cl_int), (void*)&overAxis0);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 6 , sizeof(cl_int), (void*)&overAxis1);
        error |= clSetKernelArg(kernelObject->kernel_ndrange, 7 , sizeof(cl_int), (void*)&overAxis2);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        //unsigned long localThreads[]  = {16, 16};
        size_t globalThreads[] = {kGrid};

        error = clEnqueueNDRangeKernel(queue,
                                       kernelObject->kernel_ndrange,
                                       1,
                                       NULL,
                                       globalThreads,
                                       NULL, //localThreads,
                                       0,
                                       NULL,
                                       &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        return rsltTn;
    }else{
        cl_int error;

        error =  clSetKernelArg(kernelObject->kernel_task, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1 , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2 , sizeof(cl_uint) , (void*)&kDim0);
        error |= clSetKernelArg(kernelObject->kernel_task, 3 , sizeof(cl_uint) , (void*)&kDim1);
        error |= clSetKernelArg(kernelObject->kernel_task, 4 , sizeof(cl_uint) , (void*)&kDim2);
        error |= clSetKernelArg(kernelObject->kernel_task, 5 , sizeof(cl_int), (void*)&overAxis0);
        error |= clSetKernelArg(kernelObject->kernel_task, 6 , sizeof(cl_int), (void*)&overAxis1);
        error |= clSetKernelArg(kernelObject->kernel_task, 7 , sizeof(cl_int), (void*)&overAxis2);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;

        error = clEnqueueTask( queue,
                               kernelObject->kernel_task,
                               0,
                               NULL,
                               &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        return rsltTn;
    }
    return nullptr;
}

TensorI* XilinxImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});

    assert(batchedMat->getRank()==3);
    unsigned int b,m,n;
    b = batchedMat->getShape()[0];
    m = batchedMat->getShape()[1];
    n = batchedMat->getShape()[2];

    assert(m==n); //current kernel supports square distance matrices as input

    OclTensorI *rsltIndicesSlicedTn = new OclTensorI(context,{
           b,
           m,
           (unsigned int)k
    },-1);

    //==================================================================================================================
    {//1.topk.cl.cc
 
        OclKernelObject *kernelObject = oclKernels[13];

        if(kernelObject->use_ndrange_kernel){

        }else{
            cl_int error;

            error =  clSetKernelArg(kernelObject->kernel_task, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)batchedMat)->ocl_buff);
            error |= clSetKernelArg(kernelObject->kernel_task, 1 , sizeof(cl_mem) , (void*)&((OclTensorI*)rsltIndicesSlicedTn)->ocl_buff);
            error |= clSetKernelArg(kernelObject->kernel_task, 2 , sizeof(cl_int) , (void*)&b);
            error |= clSetKernelArg(kernelObject->kernel_task, 3 , sizeof(cl_int) , (void*)&m);
            error |= clSetKernelArg(kernelObject->kernel_task, 4 , sizeof(cl_int) , (void*)&n);
            error |= clSetKernelArg(kernelObject->kernel_task, 5 , sizeof(cl_int) , (void*)&k);

            if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
            assert(error==0);

            cl_event exeEvt;
            error = clEnqueueTask(queue,kernelObject->kernel_task,0,NULL,&exeEvt);
            if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
            clWaitForEvents(1, &exeEvt);
            ReportDuration(std::string() +__func__ + "_topk",kernelObject->use_ndrange_kernel,exeEvt);

            if(error != CL_SUCCESS) {
                printf("Kernel execution failure!\n");
                exit(-22);
            }
        }

    }

    //==================================================================================================================

    return rsltIndicesSlicedTn;
}

TensorF* XilinxImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});
    assert(inputTn->getRank()==3);
    assert(indices->getRank()==3);
    assert(inputTn->getShape()[0]==indices->getShape()[0]);
    assert(inputTn->getShape()[1]==indices->getShape()[1]);

    unsigned int B,N,D,K,indicesAxis;
    B = inputTn->getShape()[0];
    N = inputTn->getShape()[1];
    D = inputTn->getShape()[2];
    K = indices->getShape()[2];
    indicesAxis = 1;

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorI* _indices = ((OclTensorI*)indices)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);

    OclTensorF* rsltTn = new OclTensorF(context,{B,N,K,D}, DATAMOVER_KERNEL_BANK_B_INDEX);
    OclKernelObject *kernelObject = oclKernels[11];

    if(kernelObject->use_ndrange_kernel){

    }else{
        cl_int error;
        error =  clSetKernelArg(kernelObject->kernel_task, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)_inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1 , sizeof(cl_mem) , (void*)&((OclTensorI*)_indices)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2 , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 3 , sizeof(cl_uint) , (void*)&indicesAxis);
        error |= clSetKernelArg(kernelObject->kernel_task, 4 , sizeof(cl_uint) , (void*)&B);
        error |= clSetKernelArg(kernelObject->kernel_task, 5 , sizeof(cl_uint) , (void*)&N);
        error |= clSetKernelArg(kernelObject->kernel_task, 6 , sizeof(cl_uint) , (void*)&D);
        error |= clSetKernelArg(kernelObject->kernel_task, 7 , sizeof(cl_uint) , (void*)&B);
        error |= clSetKernelArg(kernelObject->kernel_task, 8 , sizeof(cl_uint) , (void*)&N);
        error |= clSetKernelArg(kernelObject->kernel_task, 9 , sizeof(cl_uint) , (void*)&K);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;

        error = clEnqueueTask(queue,kernelObject->kernel_task,0,NULL,&exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }
}

TensorF* XilinxImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorF* _weights = ((OclTensorF*)weights)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);
    TensorF* _biases = ((OclTensorF*)biases)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);

    OclTensorF* rsltTn = new OclTensorF(context,
                                            {inputTn->getShape()[0],
                                             inputTn->getShape()[1],
                                             inputTn->getShape()[2],
                                             weights->getShape()[3]},
                                             DATAMOVER_KERNEL_BANK_B_INDEX);

    unsigned int B      = inputTn->getShape()[0];
    unsigned int N      = inputTn->getShape()[1];
    unsigned int K      = inputTn->getShape()[2];
    unsigned int D      = inputTn->getShape()[3];
    unsigned int chOut  = weights->getShape()[3];

    assert(D<=XILINX_BOTTLENCK_BLOCKSIZE); // this kernel cannot accept dim3>OCL_BOTTLENCK_BLOCKSIZE

    OclKernelObject *kernelObject = oclKernels[12];

    if(kernelObject->use_ndrange_kernel){

    }else{
        unsigned int dim0D, dim1D, dim2D, dim3D, dim0W,  dim1W,  dim2W,  dim3W,  dim0B;
        dim0D = B;
        dim1D = N;
        dim2D = K;
        dim3D = D;
        dim0W = 1; // 1x1 conv2d kernel
        dim1W = 1; // 1x1 conv2d kernel
        dim2W = D;
        dim3W = chOut;
        dim0B = chOut;

        cl_int error; int argcnt=0;
        error =  clSetKernelArg(kernelObject->kernel_task, argcnt++, sizeof(cl_mem) , (void*)&((OclTensorF*)_inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_mem) , (void*)&((OclTensorF*)_weights)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_mem) , (void*)&((OclTensorF*)_biases)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim0D);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim1D);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim2D);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim3D);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim0W);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim1W);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim2W);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim3W);
        error |= clSetKernelArg(kernelObject->kernel_task, argcnt++ , sizeof(cl_uint) , (void*)&dim0B);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue,kernelObject->kernel_task,0,NULL,&exeEvt);

        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);

        if(error != CL_SUCCESS) {
            printf("Kernel execution failure!\n");
            exit(-22);
        }

        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }




}

TensorF* XilinxImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});

    assert(inputTn->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());
    OclKernelObject *kernelObject = oclKernels[6];

    if(kernelObject->use_ndrange_kernel){

    }else{
        cl_int error;
        cl_ulong len = inputTn->getVectorCountPadded(CONFIG_M_AXI_WIDTH);
        std::cout<<len<<std::endl;
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_ulong), (void*)&len);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        error = clEnqueueTask(queue, kernelObject->kernel_task, 0, NULL, &exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
          printf("Kernel execution failure!\n");
          exit(-22);
        }

        return rsltTn;
    }
    return nullptr;
}

TensorF* XilinxImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    //Makes new tensor with same rank as inputTn's with tileAxis, tileCount times multiplied
    //tileAxis is in respect to the input tensor's axes.
    //----------------------------------------------------------------------------------------
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNx1xD ----> BxNxKxD        2               4
    // BxNx1   ----> BxNxK          2               3
    // Bx1xN   ----> BxKxN          1               3
    // 1xD     ----> KxD            0               2

    PrintInfo("Tile","tileAxis",tileAxis,"tileCount",tileCount,"",0,inputTn->getShape(),{},{});

    TensorF* _inputTn = ((OclTensorF*)inputTn)->CloneToDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_B_INDEX);

    int _tileAxis=tileAxis;
    int rank = inputTn->getRank();
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  tileAxis==2){
      rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount,_dim3},DATAMOVER_KERNEL_BANK_B_INDEX);
    }
    if(inputTn->getRank()==3 &&  tileAxis==1){
      rsltTn= new OclTensorF(context, {_dim0,(unsigned int)tileCount,_dim2},DATAMOVER_KERNEL_BANK_B_INDEX);
    }
    if(inputTn->getRank()==3 &&  tileAxis==2){
      rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount},DATAMOVER_KERNEL_BANK_B_INDEX);
    }

    if(rank==4 && tileAxis==2){
        // Force to use rank3 axis1 kernel to save up resources
        rank=3;
        _tileAxis=1;
        _dim0 = _dim0 * _dim1;
        _dim1 = 1;
        _dim2 = _dim3;
    }

    OclKernelObject *kernelObject = oclKernels[8];

    if(kernelObject->use_ndrange_kernel){

    }else{
        cl_int error;
        cl_ulong len = inputTn->getLength();
        error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem),  (void*)&((OclTensorF*)_inputTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem),  (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_uint), (void*)&_dim0);
        error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void*)&_dim1);
        error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void*)&_dim2);
        error |= clSetKernelArg(kernelObject->kernel_task, 5, sizeof(cl_uint), (void*)&_dim3);
        error |= clSetKernelArg(kernelObject->kernel_task, 6, sizeof(cl_int),  (void*)&rank);
        error |= clSetKernelArg(kernelObject->kernel_task, 7, sizeof(cl_int),  (void*)&_tileAxis);
        error |= clSetKernelArg(kernelObject->kernel_task, 8, sizeof(cl_int),  (void*)&tileCount);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        //unsigned long localThreads[]  = {16, 16};
        size_t globalThreads[] = {len};

        error = clEnqueueTask(queue,kernelObject->kernel_task,0,  NULL,&exeEvt);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        clWaitForEvents(1, &exeEvt);
        ReportDuration(__func__,kernelObject->use_ndrange_kernel,exeEvt);

        if(error != CL_SUCCESS) {
          printf("Kernel execution failure!\n");
          exit(-22);
        }

        rsltTn->ChangeDDRBank(program,context,queue,DATAMOVER_KERNEL_BANK_A_INDEX);
        return rsltTn;
    }
    return nullptr;
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

cl_ulong XilinxImplementation::get_duration_ns (const cl_event &event) {
    cl_ulong nstimestart, nstimeend;
    //event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    //event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(nstimestart), &nstimestart, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(nstimeend), &nstimeend, NULL);

    return(nstimeend-nstimestart);
}

void XilinxImplementation::ReportDuration(const std::string &name, const bool &isNDRange, const cl_event &event){
    uint64_t ns = get_duration_ns(event);
#ifdef REPORT_EXECUTION_DURATION
        std::cout<< "\t** "<< name << (isNDRange?"(ndrange)":"(task)")<<":: "<<
                "\t(us): " << ns/1000.0f <<
                "\t(ms): " << ns/1000000.0f <<
                "\t( s): " << ns/1000000000.0f <<
                std::endl;
#endif
}
