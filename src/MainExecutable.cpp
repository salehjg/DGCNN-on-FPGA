#include "ClassifierMultiPlatform.h"
#include <ocl_imp/xilinx/XilinxImpUnitTests.h>
#include <iostream>
using namespace std;


#define RUN_KERNELTESTS false
#define RUN_MODELARCH	true

int main(){
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
