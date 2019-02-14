//
// Created by saleh on 8/22/18.
//

#include <iostream>
#include <assert.h>
#include <ocl_imp/xilinx/XilinxImplementation.h>

using namespace std;

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
				"/xilinx/concat.cl",
				"binary_container_1.xclbin",
				"",
				"task_concat",
				false),
		/* IDX 1 :*/
		new OclKernelObject(
				KERNEL_DIR,
				"/xilinx/sqrt.cl",
				"binary_container_1.xclbin",
				"ndrange_sqrt",
				"task_sqrt",
				true),
		/* IDX 2 :*/
		new OclKernelObject(
				KERNEL_DIR,
				"/xilinx/reducemax.cl",
				"binary_container_1.xclbin",
				"ndrange_reducemax",
				"task_reducemax",
				false),
        /* IDX 3 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/reducesum4d.cl",
                "binary_container_1.xclbin",
                "ndrange_reducesum4d",
                "task_reducesum4d",
                false), 
		/* IDX 4 :*/
		new OclKernelObject(
				KERNEL_DIR,
				"/xilinx/reducesum.cl",
				"binary_container_1.xclbin",
				"",
				"task_reducesum",
				false),
		/* IDX 5 :*/
		new OclKernelObject(
				KERNEL_DIR,
				"/xilinx/tile.cl",
				"binary_container_1.xclbin",
				"",
				"task_tile",
				false),
        /* IDX 6 :*/
        new OclKernelObject(
                KERNEL_DIR,
                "/xilinx/transpose.cl",
                "binary_container_1.xclbin",
                "",
                "task_transpose",
                false),
		/* IDX 7 :*/
		new OclKernelObject(
				KERNEL_DIR,
				"/xilinx/relu.cl",
				"binary_container_1.xclbin",
				"",
				"task_relu",
				false),
    };
    
    //======================================================================================================================
    //Using signle binary container for all of the kernels for now!
    char *binary_content;
    char *_xcl_mode = getenv("XCL_EMULATION_MODE");
    string xcl_mode = string(_xcl_mode);
    xcl_mode = 	xcl_mode=="sw_emu" ? "Emulation-SW/" :
    			xcl_mode=="hw_emu" ? "Emulation-HW/" :
    			xcl_mode=="system" ? "System/" : "UNDEF" ;

    //cout<<xcl_mode<<endl;

    cout<<"*Using first kernel's container as default container.\n*Multiple container scenario is not supported yet."<<endl;
    size_t binary_content_length = load_file_to_memory( (REPO_DIR+ xcl_mode + oclKernels[0]->containerName   ).c_str(), &binary_content);

    cl_program program = clCreateProgramWithBinary(
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

    clReleaseProgram(program);
    free(binary_content);
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
    //cout<<finalStr;
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
	OclKernelObject *kernelObject = oclKernels[6];

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
    return nullptr;
}

TensorF* XilinxImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});
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

TensorF* XilinxImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){

    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});
    return nullptr;
}

TensorF* XilinxImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){
    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});
    return nullptr;
}

TensorF* XilinxImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    return nullptr;
}

TensorF* XilinxImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),{},{});
    return nullptr;
}

TensorF* XilinxImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("Sqrt","",0,"",0,"",0,inputTn->getShape(),{});
    assert(inputTn->getLength()!=0);
	OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());
	OclKernelObject *kernelObject = oclKernels[1];

	if(kernelObject->use_ndrange_kernel){
		cl_int error;
		cl_ulong len = inputTn->getLength();
		error =  clSetKernelArg(kernelObject->kernel_ndrange, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_ndrange, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_ndrange, 2, sizeof(cl_ulong), (void*)&len);
		if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
		assert(error==0);

		cl_event exeEvt;
		//unsigned long localThreads[]  = {16, 16};
		size_t globalThreads[] = {len};

		error = clEnqueueNDRangeKernel(queue,
									   kernelObject->kernel_ndrange,
									   1, //two-dim
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
	}
	else
	{
		cl_int error;
		cl_ulong len = inputTn->getLength();

		error =  clSetKernelArg(kernelObject->kernel_ndrange, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_ndrange, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_ndrange, 2, sizeof(cl_ulong), (void*)&len);

		cl_event exeEvt;

		//Launch the kernel
		error = clEnqueueTask( queue,
							   kernelObject->kernel_ndrange,
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

    unsigned int dimA0,dimA1,dimA2,dimA3;
    unsigned int dimB0,dimB1,dimB2,dimB3;
    dimA0 = inputTn1->getShape()[0]; dimB0 = inputTn2->getShape()[0];
    dimA1 = inputTn1->getShape()[1]; dimB1 = inputTn2->getShape()[1];
    dimA2 = inputTn1->getShape()[2]; dimB2 = inputTn2->getShape()[2];
    dimA3 = inputTn1->getShape()[3]; dimB3 = inputTn2->getShape()[3];
    OclTensorF* rsltTn = new OclTensorF(context, {dimA0,dimA1,dimA2,dimA3+dimB3});

    OclKernelObject *kernelObject = oclKernels[0];

    if(kernelObject->use_ndrange_kernel){

    }
    else
    {
    	cl_int error;
		error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn1)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn2)->ocl_buff);
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
    return nullptr;

}

TensorF* XilinxImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});
    return nullptr;
    
}

TensorF* XilinxImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});
    return nullptr;

}

TensorF* XilinxImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});

    assert(inputTn->getLength()!=0);
	OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());
	OclKernelObject *kernelObject = oclKernels[7];

	if(kernelObject->use_ndrange_kernel){

	}else{
		cl_int error;
		cl_ulong len = inputTn->getLength();
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

	int rank = inputTn->getRank();
	unsigned int _dim0,_dim1,_dim2,_dim3;
	_dim0 = inputTn->getShape()[0];
	_dim1 = inputTn->getShape()[1];
	_dim2 = inputTn->getShape()[2];
	_dim3 = inputTn->getShape()[3];

	OclTensorF* rsltTn = nullptr;
	if(inputTn->getRank()==4 &&  tileAxis==2){
	  rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount,_dim3});
	}
	if(inputTn->getRank()==3 &&  tileAxis==1){
	  rsltTn= new OclTensorF(context, {_dim0,(unsigned int)tileCount,_dim2});
	}
	if(inputTn->getRank()==3 &&  tileAxis==2){
	  rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount});
	}

	OclKernelObject *kernelObject = oclKernels[5];

	if(kernelObject->use_ndrange_kernel){

	}else{
		cl_int error;
		cl_ulong len = inputTn->getLength();
		error =  clSetKernelArg(kernelObject->kernel_task, 0, sizeof(cl_mem),  (void*)&((OclTensorF*)inputTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_task, 1, sizeof(cl_mem),  (void*)&((OclTensorF*)rsltTn)->ocl_buff);
		error |= clSetKernelArg(kernelObject->kernel_task, 2, sizeof(cl_uint), (void*)&_dim0);
		error |= clSetKernelArg(kernelObject->kernel_task, 3, sizeof(cl_uint), (void*)&_dim1);
		error |= clSetKernelArg(kernelObject->kernel_task, 4, sizeof(cl_uint), (void*)&_dim2);
		error |= clSetKernelArg(kernelObject->kernel_task, 5, sizeof(cl_uint), (void*)&_dim3);
		error |= clSetKernelArg(kernelObject->kernel_task, 6, sizeof(cl_int),  (void*)&rank);
		error |= clSetKernelArg(kernelObject->kernel_task, 7, sizeof(cl_int),  (void*)&tileAxis);
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

		return rsltTn;
	}
    return nullptr;
}

void XilinxImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorF* inputTn,
        string npy_dir){

}

void XilinxImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorI* inputTn,
        string npy_dir){

}

bool XilinxImplementation::CompareTensors(WorkScheduler scheduler,TensorF *inputTn1, TensorF *inputTn2) {
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
