#ifndef XILINXIMPUNITTESTS_
#define XILINXIMPUNITTESTS_

#include "WorkScheduler.h"
#include "TensorF.h"
#include "TensorI.h"
#include <PlatformSelector.h>

using namespace std;

struct ReportObject{
	std::string unitTestName;
	bool passed;

	ReportObject(string name, bool isPassed){
		unitTestName = name;
		passed = isPassed;
	}
};

class XilinxImpUnitTests{
public:
	XilinxImpUnitTests();

	ReportObject* TensorFloat();
	ReportObject* KernelConcat2();
	ReportObject* KernelSqrt();
	ReportObject* KernelReduceMax();
	ReportObject* KernelReduceSum4D();
	ReportObject* KernelReduceSum();
	ReportObject* KernelTile();
	ReportObject* KernelTranspose();
	ReportObject* KernelRelu();
	ReportObject* KernelSquare();
	ReportObject* KernelMatops();
	ReportObject* KernelMean();
	ReportObject* KernelVariance();
	ReportObject* KernelMatmul();
	ReportObject* KernelConv2Mlp();
	ReportObject* KernelTopK();
	ReportObject* KernelGather();

	~XilinxImpUnitTests();
	void RunAll();
private:
	TensorF* GenerateTensor(int pattern, vector<unsigned int> shape);
	float float_rand(float min, float max);
	TensorI* GenerateTensor(int intMin, int intMax, vector<unsigned int> shape);
	void PrintReport(ReportObject *reportObj);


    PlatformSelector* platformSelector;
    WorkScheduler scheduler;
};

#endif
