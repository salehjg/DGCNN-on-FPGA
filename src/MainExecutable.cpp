#include "ClassifierMultiPlatform.h"
#include <ocl_imp/xilinx/XilinxImpUnitTests.h>
#include <iostream>
using namespace std;


#define RUN_KERNELTESTS true
#define RUN_MODELARCH   false 

char* globalArgXclBin;
char* globalArgDataPath;

int main(int argc, char* argv[]){
    // Check the number of parameters
    if (argc < 3) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " < Path to *.xclbin >" << " < Path to data dir with ...data >"<< "[< Emulation Mode(sw_emu, hw_emu) >]" << std::endl;
        return 1;
    }
    // Print the user's name:
    std::cout << "Executable Path: " << argv[0] << std::endl;
    std::cout << "FPGA BIN FNAME: " << argv[1] << std::endl;
    std::cout << "DATA DIR PATH: " << argv[2] << std::endl;
    if(argc>3) {
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
