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

string globalArgXclBin;
string globalArgDataPath;
unsigned globalBatchsize;
bool globalRunTests=false;
bool globalRunClassifier=true;
bool globalDumpTensors=false;

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

    parser.add_argument()
        .names({"-t", "--testsonly"})
        .description("Only run OCl tests(no value is needed for this argument)")
        .required(false);

    parser.add_argument()
        .names({"-c", "--classifieronly"})
        .description("Only run OCl classifier(no value is needed for this argument)")
        .required(false);   

    parser.add_argument()
        .names({"-k", "--dumptensors"})
        .description("Dump tensors into *.npy files in the data directory(no value is needed for this argument)")
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

    if(parser.exists("testsonly")) {
        globalRunClassifier = false;
        globalRunTests = true;
        std::cout<<"Only OCl tests are going to be run."<<std::endl;
    }

    if(parser.exists("classifieronly")) {
        globalRunClassifier = true;
        globalRunTests = false;
        std::cout<<"Only OCl classifier is going to be run."<<std::endl;
    }

    if(parser.exists("dumptensors")) {
        globalDumpTensors = true;
        std::cout<<"Tensors will be dumped into separate numpy files in the data directory."<<std::endl;
    }

    if(globalRunTests){
        cout<< "======================================================" <<endl;
        cout<< "Running Kernel Unit Tests ...\n" <<endl;
        XilinxImpUnitTests xilinxImpUnitTests;
        xilinxImpUnitTests.RunAll();
    }

    if(globalRunClassifier){
        cout<< "======================================================" <<endl;
        cout<< "Running Selected ModelArch ...\n" <<endl;
        ClassifierMultiplatform();

    }
}
