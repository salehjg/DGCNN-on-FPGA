#include "ocl_imp/xilinx/xcl2.hpp"

using namespace std;

cl_mem_ext_ptr_t CreateExtendedPointer(void *hostPtr, cl_mem_flags memoryBank){
    cl_mem_ext_ptr_t extendedPointer;
    extendedPointer.flags = memoryBank;
    extendedPointer.obj = hostPtr;
    extendedPointer.param = 0;
    return extendedPointer;
}

int TranslateBankIndex(int bankIndex){
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

void CreateBufferAndTest(cl::Context *context, cl::CommandQueue *queue, int len, int bankIndex){
    cl_int ocl_stat;
    cl::Buffer ocl_buff;
    float* testArrayHost = new float[len];
    float* testArrayUDT = new float[len];
    for(int i=0; i<len; i++){
        testArrayHost[i] =  (float)i;
        testArrayUDT[i] = 0;
    }
    cout<<"\n===================================\nBANK "<<bankIndex<<" TEST:"<<endl;
    cout<<"\nPHASE 1: SEND DATA TO THE DEVICE...\n"<<endl;

    //-------------------------------------------------------------------------------------
    cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(bankIndex));
    cl_mem_flags  flags = CL_MEM_READ_WRITE;
    //flags |= CL_MEM_USE_HOST_PTR;
    flags |= CL_MEM_EXT_PTR_XILINX;
    cout<<"Created xilinx extended ptr."<<endl;

    //-------------------------------------------------------------------------------------
    OCL_CHECK(ocl_stat, ocl_buff = cl::Buffer(*context, flags, len*sizeof(float), &extPtr, &ocl_stat));
    cout<<"Created cl::Buffer."<<endl;
    OCL_CHECK(ocl_stat, ocl_stat = queue->enqueueWriteBuffer(ocl_buff, CL_TRUE, 0, len*sizeof(float), testArrayHost, nullptr, nullptr));
    cout<<"Added enqueueWriteBuffer to transfer data to the device memory."<<endl;

    //-------------------------------------------------------------------------------------
    cout<<"\nPHASE 2: READ BACK THE DATA FROM DEVICE...\n"<<endl;

    OCL_CHECK(ocl_stat,ocl_stat = queue->enqueueReadBuffer(
            ocl_buff,
            CL_TRUE,
            0,
            len*sizeof(float),
            testArrayUDT,
            nullptr,
            nullptr));
    cout<<"Added enqueueReadBuffer to transfer data back from the device to the host."<<endl;

    //-------------------------------------------------------------------------------------
    cout<<"\nPHASE 3: COMPARE RESULTS...\n"<<endl;
    for(int i=0; i<len; i++){
        cout<<"HostGold["<<i<<"]= "<< testArrayHost[i]<<endl;
        cout<<"DeviceUDT["<<i<<"]= "<< testArrayUDT[i]<<endl;
    }
}

int main(int argc, char **argv) {
    cl_int err;
    cl::Context *context;
    cl::Program *program;
    cl::CommandQueue *queue;


    cout<< "This program checks out some host side operations on the real device."<<endl;
    if(argc!=2){
        cerr<<"Please enter the abs path for xclbin or awsxclbin file as the first and only argument.\nAborting...\n";
        exit(1);
    }
    std::string BinaryFile(argv[1]);
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, context  = new cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err,queue = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    printf("INFO: loading the fpga image...\n");
    auto fileBuf = xcl::read_binary_file(BinaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err,program = new cl::Program(*context, {device}, bins, NULL, &err));

    CreateBufferAndTest(context, queue, 128, 1); //Test Bank1
    CreateBufferAndTest(context, queue, 128, 2); //Test Bank2
    //CreateBufferAndTest(context, queue, 128, 0); //Test Bank0
    //CreateBufferAndTest(context, queue, 128, 3); //Test Bank3
}