#include "build_config.h"
#include "ocl_imp/xilinx/xcl2.hpp"

using namespace std;
spdlog::logger *logger;

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

    SPDLOG_LOGGER_INFO(logger,"===================================");
    SPDLOG_LOGGER_INFO(logger,"BANK {} Test...", bankIndex);
    SPDLOG_LOGGER_INFO(logger,"PHASE 1: TRANSFERING DATA TO THE DEVICE...");

    //-------------------------------------------------------------------------------------
    cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(bankIndex));
    cl_mem_flags  flags = CL_MEM_READ_WRITE;
    //flags |= CL_MEM_USE_HOST_PTR;
    flags |= CL_MEM_EXT_PTR_XILINX;
    SPDLOG_LOGGER_DEBUG(logger,"Created xilinx extended ptr");

    //-------------------------------------------------------------------------------------
    OCL_CHECK(ocl_stat, ocl_buff = cl::Buffer(*context, flags, len*sizeof(float), &extPtr, &ocl_stat));
    SPDLOG_LOGGER_DEBUG(logger,"Created cl::Buffer");
    OCL_CHECK(ocl_stat, ocl_stat = queue->enqueueWriteBuffer(ocl_buff, CL_TRUE, 0, len*sizeof(float), testArrayHost, nullptr, nullptr));
    SPDLOG_LOGGER_DEBUG(logger,"Added enqueueWriteBuffer to transfer data to the device memory");

    //-------------------------------------------------------------------------------------
    SPDLOG_LOGGER_INFO(logger,"PHASE 2: READING BACK THE DATA FROM THE DEVICE...");

    OCL_CHECK(ocl_stat,ocl_stat = queue->enqueueReadBuffer(
            ocl_buff,
            CL_TRUE,
            0,
            len*sizeof(float),
            testArrayUDT,
            nullptr,
            nullptr));
    SPDLOG_LOGGER_DEBUG(logger,"Added enqueueReadBuffer to transfer data back from the device to the host");

    //-------------------------------------------------------------------------------------
    SPDLOG_LOGGER_INFO(logger,"PHASE 3: COMPARE RESULTS...");
    for(int i=0; i<len; i++){
        SPDLOG_LOGGER_INFO(logger,"HostGold[{}]= {}", i, testArrayHost[i]);
        SPDLOG_LOGGER_INFO(logger,"DeviceUDT[{}]= {}", i, testArrayUDT[i]);
    }
}

int main(int argc, char **argv) {
    cl_int err;
    cl::Context *context;
    cl::Program *program;
    cl::CommandQueue *queue;

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%H:%M:%S.%e][%^%l%$] %v");
    logger = new spdlog::logger("HostTest-logger", {console_sink});
    logger->set_level(spdlog::level::trace); 

    SPDLOG_LOGGER_INFO(logger,"This program checks out some host side operations on the real device");

    if(argc!=2){
        cerr<<"Please enter the abs path for the *.xclbin or *.awsxclbin file as the first and only argument.\nAborting...\n";
        exit(1);
    }
    std::string BinaryFile(argv[1]);
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, context  = new cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err,queue = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    SPDLOG_LOGGER_INFO(logger, "Found Device: {}", device_name.c_str());
    
    SPDLOG_LOGGER_INFO(logger, "Loading the fpga image...");
    auto fileBuf = xcl::read_binary_file(BinaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err,program = new cl::Program(*context, {device}, bins, NULL, &err));

#ifdef USEMEMORYBANK0
    CreateBufferAndTest(context, queue, 128, 0); //Test Bank0
#endif
#ifdef USEMEMORYBANK1
    CreateBufferAndTest(context, queue, 128, 1); //Test Bank1
#endif
#ifdef USEMEMORYBANK2
    CreateBufferAndTest(context, queue, 128, 2); //Test Bank2
#endif
#ifdef USEMEMORYBANK3
    CreateBufferAndTest(context, queue, 128, 3); //Test Bank3
#endif
}