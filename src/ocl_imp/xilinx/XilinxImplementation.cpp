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
		// The get_xil_devices will return vector of Xilinx Devices 
    	std::cout<<"- - - - - - - - - - -"<<std::endl;
        std::vector<cl::Device> devices = xcl::get_xil_devices();
        if(devices.size()>1){
            cout<<"WARNING:  More than 1 xilinx device has been found. Selecting index zero as default device."<<endl;
        }
        device = devices[0];
    }

    //======================================================================================================================
    {
        //Creating Context and Command Queue for selected Device 
        OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, queue = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, device_name = device.getInfo<CL_DEVICE_NAME>(&err));
        std::cout << "Found Device=" << device_name.c_str() << std::endl;
        std::cout<<"- - - - - - - - - - -"<<std::endl;
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
    };
    
    //======================================================================================================================
    //std::string emulation_mode = "Emulation-SW/";
	//std::string binaryFile = "";
	//binaryFile.append(REPO_DIR );
	//binaryFile.append("Emulation-SW/" );
	//binaryFile.append((kernelObject->containerName) );
    /*
    for(OclKernelObject *kernelObject : oclKernels){
        std::string binaryFile = xcl::find_binary_file(device_name, kernelObject->containerName);
        cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
        std::cout << "\tProcessing Kernel(filename): " << kernelObject->fileName << std::endl;
        std::cout << "\tKernel Name(function name): " << kernelObject->kernelName << std::endl;
        std::cout << "\tBinary Container(filename):"<< binaryFile << std::endl;
        OCL_CHECK(err, cl::Program program(*context, {device}, bins, NULL, &err));
        OCL_CHECK(err, kernelObject->kernel = new cl::Kernel(program, kernelObject->kernelName, &err));
    }*/

    //======================================================================================================================
    //Using signle binary container for all of the kernels for now!
    std::string binaryFile = xcl::find_binary_file(device_name, "binary_container_1.xclbin");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    OCL_CHECK(err, cl::Program program(*context, {device}, bins, NULL, &err));
    for(OclKernelObject *kernelObject : oclKernels){
         //std::cout << "\tProcessing Kernel(filename): " << kernelObject->fileName << std::endl;
         //std::cout << "\tKernel Name(function name): " << kernelObject->kernelName << std::endl;
         //std::cout << "\tBinary Container(filename):"<< binaryFile << std::endl;
         if(kernelObject->use_ndrange_kernel){
        	 OCL_CHECK(err, kernelObject->kernel_ndrange = new cl::Kernel(program, kernelObject->kernelName_ndrange, &err));
         }else{
        	 OCL_CHECK(err, kernelObject->kernel_task = new cl::Kernel(program, kernelObject->kernelName_task, &err));
         }
     }
    std::cout<<"- - - - - - - - - - -"<<std::endl;

    //======================================================================================================================
}

cl::Context* XilinxImplementation::getContext(){
    return context;
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
    //cout<<finalStr;
}

TensorF* XilinxImplementation::Transpose(WorkScheduler scheduler, TensorF *batchedMat){
    PrintInfo("Transpose","",0,"",0,"",0,batchedMat->getShape(),{});
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
    return nullptr;
}

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is
TensorF* XilinxImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){

    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
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
		size_t globalThreads[] = {len};
		int narg = 0;

		kernelObject->kernel_ndrange->setArg(narg++, *((OclTensorF*)inputTn)->ocl_buff);
		kernelObject->kernel_ndrange->setArg(narg++, *(rsltTn->ocl_buff));
		kernelObject->kernel_ndrange->setArg(narg++, len);

		cl::Event exeEvt;

		//Launch the kernel
		getQueue()->enqueueNDRangeKernel(
				*kernelObject->kernel_ndrange,
				cl::NullRange,
				cl::NDRange(globalThreads[0]),
				cl::NullRange,
				NULL,
				&exeEvt);

		error = cl::WaitForEvents({exeEvt});
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
		size_t globalThreads[] = {len};
		int narg = 0;

		kernelObject->kernel_task->setArg(narg++, *((OclTensorF*)inputTn)->ocl_buff);
		kernelObject->kernel_task->setArg(narg++, *(rsltTn->ocl_buff));
		kernelObject->kernel_task->setArg(narg++, len);

		cl::Event exeEvt;

		//Launch the kernel
		getQueue()->enqueueTask(*kernelObject->kernel_task, NULL, &exeEvt);

		error = cl::WaitForEvents({exeEvt});
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
    	int narg = 0;
		kernelObject->kernel_task->setArg(narg++, *((OclTensorF*)inputTn1)->ocl_buff);
		kernelObject->kernel_task->setArg(narg++, *((OclTensorF*)inputTn2)->ocl_buff);
		kernelObject->kernel_task->setArg(narg++, *(rsltTn->ocl_buff));

		kernelObject->kernel_task->setArg(narg++, dimA0);
		kernelObject->kernel_task->setArg(narg++, dimA1);
		kernelObject->kernel_task->setArg(narg++, dimA2);
		kernelObject->kernel_task->setArg(narg++, dimA3);

		kernelObject->kernel_task->setArg(narg++, dimB0);
		kernelObject->kernel_task->setArg(narg++, dimB1);
		kernelObject->kernel_task->setArg(narg++, dimB2);
		kernelObject->kernel_task->setArg(narg++, dimB3);

		cl::Event exeEvt;

		//Launch the kernel
		getQueue()->enqueueTask(*kernelObject->kernel_task, NULL, &exeEvt);

		cl_int error = cl::WaitForEvents({exeEvt});
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

uint64_t XilinxImplementation::get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
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
