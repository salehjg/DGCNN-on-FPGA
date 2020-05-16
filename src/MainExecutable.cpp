#include "ClassifierMultiPlatform.h"
#include <ocl_imp/xilinx/XilinxImpUnitTests.h>
#include <iostream>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;


#define RUN_KERNELTESTS false    
#define RUN_MODELARCH   true 

char* globalArgXclBin;
char* globalArgDataPath;

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

int main(int argc, char* argv[]){
    signal(SIGSEGV, handler);
    signal(SIGABRT, handler);

    // Check the number of parameters
    if (argc < 3) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " < Path to *.xclbin >" << " < Path to data dir with ...data >"<< " [< Emulation Mode(sw_emu, hw_emu) >]" << std::endl;
        return 1;
    }
    // Print the user's name:
    std::cout << "Executable Path: " << argv[0] << std::endl;
    std::cout << "FPGA BIN FNAME: " << argv[1] << std::endl;
    std::cout << "DATA DIR PATH: " << argv[2] << std::endl;
    if(argc>3) {
        // for debugging...
        std::cout << "SELECTED EMULATION MODE: " << argv[3] << std::endl;
        if (setenv("XCL_EMULATION_MODE", argv[3], 1) < 0) {
            fprintf(stderr, "Error setting XCL_MODE env. var.\n");
        }
    }

    globalArgXclBin = argv[1];
    globalArgDataPath = argv[2];


    cout<< "======================================================" <<endl;
    cout<< "DeepPoint-V1-FPGA" <<endl;
    cout<< "Details:\n" <<endl;

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
