#include "argparse.h"
#include "ClassifierMultiPlatform.h"
#include <ocl_imp/xilinx/XilinxImpUnitTests.h>
#include <iostream>
#include <execinfo.h>
#include <unistd.h>
#include <string>
#include <csignal>

using namespace std;
using namespace argparse;

#define RUN_KERNELTESTS false    
#define RUN_MODELARCH   true

string globalArgXclBin;
string globalArgDataPath;
unsigned globalBatchsize;

void handler(int sig) {
    void *array[40];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 40);

    // print out all the frames to stderr
    cerr<<"The host program has crashed, printing call stack:\n";
    cerr<<"Error: signal "<< sig<<"\n";
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

int main(int argc, const char* argv[]){
    signal(SIGSEGV, handler);
    signal(SIGABRT, handler);

    ArgumentParser parser("ArgParse");
    parser.add_argument()
        .names({"-i", "--image"})
        .description("FPGA image(*.xclbin or *.awsxclbin)")
        .required(true);

    parser.add_argument()
        .names({"-d", "--data"})
        .description("Data directory")
        .required(true);  

    parser.add_argument()
        .names({"-b", "--batchsize"})
        .description("Batch-size")
        .required(false);     
          
    parser.add_argument()
        .names({"-e", "--emumode"})
        .description("Forced emulation mode(sw_emu or hw_emu)")
        .required(false);  

    parser.enable_help();
    auto err = parser.parse(argc, argv);
    if(err){
        std::cerr << err << std::endl;
        parser.print_help();
        return -1;
    }

    if(parser.exists("help")){
        parser.print_help();
        return 0;
    }

    if(parser.exists("b")) {
        globalBatchsize = parser.get<unsigned>("b");
    }else{
        globalBatchsize = 5;
    }
    std::cout<<"Batch-size: "<<globalBatchsize<<std::endl;

    if(parser.exists("i")) {
        globalArgXclBin = parser.get<string>("i").c_str();
        std::cout<<"FPGA Image: "<<globalArgXclBin<<std::endl;
    }

    if(parser.exists("d")) {
        globalArgDataPath = parser.get<string>("d");
        std::cout<<"Data Directory: "<<globalArgDataPath<<std::endl;
    }

    if(parser.exists("e")) {
        const char *forcedMode = parser.get<string>("e").c_str();
        std::cout<<"Forced Emulation Mode: "<<forcedMode<<std::endl;
        if (setenv("XCL_EMULATION_MODE", forcedMode, 1) < 0) {
            std::cerr <<"Error setting XCL_MODE env. var."<<std::endl;
        }
    }

    if(RUN_KERNELTESTS){
        cout<< "======================================================" <<endl;
        cout<< "Running Kernel Unit Tests ...\n" <<endl;
        XilinxImpUnitTests xilinxImpUnitTests;
        xilinxImpUnitTests.RunAll();
        //xilinxImpUnitTests.~XilinxImpUnitTests();
    }
    //---------------------
    if(RUN_MODELARCH){
        cout<< "======================================================" <<endl;
        cout<< "Running Selected ModelArch ...\n" <<endl;
        ClassifierMultiplatform();

    }
}
