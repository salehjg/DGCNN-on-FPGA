#include <ocl_imp/xilinx/XilinxImpUnitTests.h>

XilinxImpUnitTests::XilinxImpUnitTests(){
	// GPU_OCL is FPGA now, because there is no reason to run code both on FPGA and GPU at same executable!
	// So ... 
	// GPU_OCL with attention to OCL part of the name means FPGA 
	platformSelector = new PlatformSelector(PLATFORMS::GPU_OCL, {PLATFORMS::CPU,PLATFORMS::GPU_OCL});
}

float XilinxImpUnitTests::float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

TensorF* XilinxImpUnitTests::GenerateTensor(int pattern, vector<unsigned int> shape){
    TensorF *testTn = new TensorF(shape);
    unsigned long _len = testTn->getLength();
    if(pattern==-1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 0;
        }
    }
    if(pattern==0){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = float_rand(0,2.50f);
        }
    }
    if(pattern==1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i %5 ;//+ float_rand(0,2.50f);
        }
    }
    if(pattern==2){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i %10 + float_rand(0,2.50f);
        }
    }
    if(pattern==3){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    if(pattern==4){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    if(pattern==5){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    return testTn;
}

TensorI* XilinxImpUnitTests::GenerateTensor(int intMin, int intMax, vector<unsigned int> shape){
    TensorI *testTn = new TensorI(shape);
    unsigned long _len = testTn->getLength();

    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = (int)float_rand((float)intMin,(float)intMax);
    }

    return testTn;
}

void XilinxImpUnitTests::PrintReport(ReportObject *reportObj){
	cout << "TEST: "<<reportObj->unitTestName << "\t\tRESULT: "<< (reportObj->passed? "PASS":"FAIL") <<endl;
}

ReportObject* XilinxImpUnitTests::TensorFloat(){
	TensorF* src = GenerateTensor(0, {5,2,2});
	TensorF* srcDevice = platformSelector->CrossThePlatform(src, PLATFORMS::GPU_OCL);
	bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, src, srcDevice);
	ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
	return obj;
}

ReportObject* XilinxImpUnitTests::KernelConcat2(){
	TensorF* tensorSrc1 = GenerateTensor(3,{5,2,50,20});
	TensorF* tensorSrc2 = GenerateTensor(3,{5,2,50,30});

	TensorF* tensorCpu = platformSelector->Concat2(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,3);
	TensorF* tensorGpu = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,3);

	bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
	ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
	return obj;
}

ReportObject* XilinxImpUnitTests::KernelSqrt(){
	TensorF* tensorSrc = GenerateTensor(3,{5,2,50,20});

	TensorF* tensorCpu = platformSelector->Sqrt(PLATFORMS::CPU,scheduler,tensorSrc);
	TensorF* tensorGpu = platformSelector->Sqrt(PLATFORMS::GPU_OCL,scheduler,tensorSrc);

	bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
	ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
	return obj;
}


ReportObject* XilinxImpUnitTests::KernelReduceMax(){
    TensorF* tensorSrc1 = GenerateTensor(0,{5,2,50,20});
    TensorF* tensorCpu1 = platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc1,2);
    TensorF* tensorGpu1 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,2);
    bool comparisonResult1 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu1,tensorGpu1);

    TensorF* tensorSrc2 = GenerateTensor(0,{5,5,1,20});
    TensorF* tensorCpu2 = platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc2,1);
    TensorF* tensorGpu2 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc2,1);
    bool comparisonResult2 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu2,tensorGpu2);

	ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult1 && comparisonResult2);
	return obj;
}

ReportObject* XilinxImpUnitTests::KernelReduceSum4D(){
    //Rank4_TTTF
    //TensorF* tensorSrc = GenerateTensor(1,{5,1024,20,256});
    TensorF* tensorSrc = GenerateTensor(1,{5,1024,20,256});
    TensorF* tensorCpu = platformSelector->ReduceSum4D(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
    TensorF* tensorGpu = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

void XilinxImpUnitTests::RunAll(){
	PrintReport(TensorFloat());
	PrintReport(KernelConcat2());
	PrintReport(KernelSqrt());
	PrintReport(KernelReduceMax());
	PrintReport(KernelReduceSum4D());
}

XilinxImpUnitTests::~XilinxImpUnitTests(){
	cout<<"~XilinxImpUnitTests"<<endl;
	delete(platformSelector);
}


