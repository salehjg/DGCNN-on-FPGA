#include "xilinx/config.h"
#include <ocl_imp/xilinx/XilinxImpUnitTests.h>
#include <ocl_imp/xilinx/AxiHelper.h>
#include <cnpy.h>


XilinxImpUnitTests::XilinxImpUnitTests(){
    platformSelector = new PlatformSelector(PLATFORMS::GPU_OCL, {PLATFORMS::CPU,PLATFORMS::GPU_OCL},false);
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
    if(pattern==6){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i;
        }
    }
    if(pattern==7){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = float_rand(-2.50f,2.50f);
        }
    }
    return testTn;
}

TensorI* XilinxImpUnitTests::GenerateTensorInteger(int pattern, vector<unsigned int> shape){
    TensorI *testTn = new TensorI(shape);
    unsigned long _len = testTn->getLength();
    if(pattern==-1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 0;
        }
    }
    if(pattern==1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i%5 + i ;
        }
    }

    return testTn;
}

TensorI* XilinxImpUnitTests::GenerateTensorInteger(int intMin, int intMax, vector<unsigned int> shape){
    TensorI *testTn = new TensorI(shape);
    unsigned long _len = testTn->getLength();

    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = (int)float_rand((float)intMin,(float)intMax);
    }

    return testTn;
}

void XilinxImpUnitTests::PrintReport(ReportObject *reportObj){
    if(reportObj){
        if(reportObj->passed)
            SPDLOG_LOGGER_INFO(logger,"\033[1;36mTEST: {}\t\tRESULT: {}\033[0m", reportObj->unitTestName, "PASS");
        else
            SPDLOG_LOGGER_ERROR(logger,"\033[1;31mTEST: {}\t\tRESULT: {}\033[0m", reportObj->unitTestName, "FAIL");
    }
}

ReportObject* XilinxImpUnitTests::TensorFloat(){
    TensorF* src = GenerateTensor(0, {5,2,2});
    TensorF* srcDevice = platformSelector->CrossThePlatform(src, PLATFORMS::GPU_OCL);
    bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, src, srcDevice);
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorBankFloat(){
    TensorF* tensorCpu = GenerateTensor(7,{5,5,2});
    OclTensorF* tensorSrc_defaultBank = (OclTensorF*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);

    bool rslt_before_changing_bank = platformSelector->CompareTensors(
        PLATFORMS::CPU,
        scheduler,
        tensorCpu,
        tensorSrc_defaultBank);
    
    tensorSrc_defaultBank->ChangeDDRBank(
        platformSelector->openclPlatformClass->getProgram(),
        platformSelector->openclPlatformClass->getContext(),
        platformSelector->openclPlatformClass->getQueue(),
        DATAMOVER_KERNEL_BANK_B_INDEX);
    
    bool rslt_after_changing_bank = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorSrc_defaultBank);
    

    tensorSrc_defaultBank->ChangeDDRBank(
        platformSelector->openclPlatformClass->getProgram(),
        platformSelector->openclPlatformClass->getContext(),
        platformSelector->openclPlatformClass->getQueue(),
        DATAMOVER_KERNEL_BANK_A_INDEX);

    bool rslt_after_changing_bank_reverse = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorSrc_defaultBank);

    
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt_before_changing_bank && rslt_after_changing_bank && rslt_after_changing_bank_reverse);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorBankInteger(){
    TensorI* tensorCpu = GenerateTensorInteger(7,{5,5,2});
    OclTensorI* tensorSrc_defaultBank = (OclTensorI*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);

    bool rslt_before_changing_bank = platformSelector->CompareTensorsInteger(
        PLATFORMS::CPU,
        scheduler,
        tensorCpu,
        tensorSrc_defaultBank);

    tensorSrc_defaultBank->ChangeDDRBank(
        platformSelector->openclPlatformClass->getProgram(),
        platformSelector->openclPlatformClass->getContext(),
        platformSelector->openclPlatformClass->getQueue(),
        DATAMOVER_KERNEL_BANK_B_INDEX);

    bool rslt_after_changing_bank = platformSelector->CompareTensorsInteger(PLATFORMS::CPU,scheduler,tensorCpu,tensorSrc_defaultBank);


    tensorSrc_defaultBank->ChangeDDRBank(
        platformSelector->openclPlatformClass->getProgram(),
        platformSelector->openclPlatformClass->getContext(),
        platformSelector->openclPlatformClass->getQueue(),
        DATAMOVER_KERNEL_BANK_A_INDEX);

    bool rslt_after_changing_bank_reverse = platformSelector->CompareTensorsInteger(PLATFORMS::CPU,scheduler,tensorCpu,tensorSrc_defaultBank);


    ReportObject* obj = new ReportObject(__FUNCTION__, rslt_before_changing_bank && rslt_after_changing_bank && rslt_after_changing_bank_reverse);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorCloneBankFloat(){

    unsigned err=0;

    TensorF* tensorCpu = GenerateTensor(7,{5,5,2});
    OclTensorF* tensorSrc_defaultBank = (OclTensorF*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);

    err += !platformSelector->CompareTensors(
            PLATFORMS::CPU,
            scheduler,
            tensorCpu,
            tensorSrc_defaultBank);

    vector<unsigned> Banks;
#ifdef USEMEMORYBANK0
    Banks.push_back(0);
#endif
#ifdef USEMEMORYBANK1
    Banks.push_back(1);
#endif
#ifdef USEMEMORYBANK2
    Banks.push_back(2);
#endif
#ifdef USEMEMORYBANK3
    Banks.push_back(3);
#endif

    for(unsigned bankSrc:Banks){
        for(unsigned bankDest:Banks){
            SPDLOG_LOGGER_INFO(logger,"From bank {} to {}", bankSrc, bankDest);
            if(err>0){
                SPDLOG_LOGGER_ERROR(logger,"Tensor data mismatch");
            }
            OclTensorF* tensorCloned_BankB = (OclTensorF*)
                tensorSrc_defaultBank->CloneIfNeededToDDRBank(
                    platformSelector->openclPlatformClass->getProgram(),
                    platformSelector->openclPlatformClass->getContext(),
                    platformSelector->openclPlatformClass->getQueue(),
                    bankSrc);

            err += !platformSelector->CompareTensors(
                PLATFORMS::CPU,
                scheduler,
                tensorCpu,
                tensorCloned_BankB);

            OclTensorF* tensorCloned_BankA = (OclTensorF*)
                tensorCloned_BankB->CloneIfNeededToDDRBank(
                    platformSelector->openclPlatformClass->getProgram(),
                    platformSelector->openclPlatformClass->getContext(),
                    platformSelector->openclPlatformClass->getQueue(),
                    bankDest);

            err += !platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorCloned_BankA);
        }       
    }

    ReportObject* obj = new ReportObject(__FUNCTION__, err==0);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorCloneBankInteger(){

    unsigned err=0;

    TensorI* tensorCpu = GenerateTensorInteger(7,{5,5,2});
    OclTensorI* tensorSrc_defaultBank = (OclTensorI*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);

    err += !platformSelector->CompareTensorsInteger(
            PLATFORMS::CPU,
            scheduler,
            tensorCpu,
            tensorSrc_defaultBank);
    
    vector<unsigned> Banks;
#ifdef USEMEMORYBANK0
    Banks.push_back(0);
#endif
#ifdef USEMEMORYBANK1
    Banks.push_back(1);
#endif
#ifdef USEMEMORYBANK2
    Banks.push_back(2);
#endif
#ifdef USEMEMORYBANK3
    Banks.push_back(3);
#endif

    for(unsigned bankSrc:Banks){
        for(unsigned bankDest:Banks){
            SPDLOG_LOGGER_INFO(logger,"From bank {} to {}", bankSrc, bankDest);
            if(err>0){
                SPDLOG_LOGGER_ERROR(logger,"Tensor data mismatch");
            }

            OclTensorI* tensorCloned_BankB = (OclTensorI*)
                    tensorSrc_defaultBank->CloneIfNeededToDDRBank(
                            platformSelector->openclPlatformClass->getProgram(),
                            platformSelector->openclPlatformClass->getContext(),
                            platformSelector->openclPlatformClass->getQueue(),
                            bankSrc);

            err += !platformSelector->CompareTensorsInteger(PLATFORMS::CPU,scheduler,tensorCpu,tensorCloned_BankB);

            OclTensorI* tensorCloned_BankA = (OclTensorI*)
                    tensorCloned_BankB->CloneIfNeededToDDRBank(
                            platformSelector->openclPlatformClass->getProgram(),
                            platformSelector->openclPlatformClass->getContext(),
                            platformSelector->openclPlatformClass->getQueue(),
                            bankDest);

            err += !platformSelector->CompareTensorsInteger(PLATFORMS::CPU,scheduler,tensorCpu,tensorCloned_BankA);
        }
    }

    ReportObject* obj = new ReportObject(__FUNCTION__, err==0);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorPadUnpadCpuFloat(){
    //Tests Cpu last dim padding/unpadding
    TensorF* tensorCpu = GenerateTensor(0,{5,5,2});
    OclTensorF* tensorPadded_defaultBank = (OclTensorF*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);
    TensorF* tensorCpu2 = platformSelector->CrossThePlatform(tensorPadded_defaultBank, PLATFORMS::CPU);
    bool rslt = platformSelector->CompareTensors(
        PLATFORMS::CPU,
        scheduler,
        tensorCpu,
        tensorCpu2);
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}

ReportObject* XilinxImpUnitTests::TensorPadUnpadCpuInteger(){
    //Tests Cpu last dim padding/unpadding
    TensorI* tensorCpu = GenerateTensorInteger(0,{5,5,2});
    OclTensorI* tensorPadded_defaultBank = (OclTensorI*) platformSelector->CrossThePlatform(tensorCpu, PLATFORMS::GPU_OCL);
    TensorI* tensorCpu2 = platformSelector->CrossThePlatform(tensorPadded_defaultBank, PLATFORMS::CPU);


    bool comparisonResult = true;
    if (tensorCpu->getShape() != tensorCpu2->getShape()){
        comparisonResult = false;
    }
    else{

        for(unsigned long i=0; i<tensorCpu->getLength(); i++){
            int rCpu = tensorCpu->_buff[i];
            int rGpu = tensorCpu2->_buff[i];
            if(rCpu != rGpu){
                comparisonResult=false;
            }

        }

    }


    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelPadLastDimFloat(){
    bool rslt = true;
    /*{ //Sub-vec Padding
        const unsigned int lastDimPadded = 16;
        TensorF* tensorCpu = GenerateTensor(0,{16,16,6});
        TensorF* tensorPaddedGold = platformSelector->PadLastDim(PLATFORMS::CPU,scheduler,tensorCpu,lastDimPadded);
        TensorF* tensorPaddedTest = platformSelector->PadLastDim(PLATFORMS::GPU_OCL,scheduler,tensorCpu,lastDimPadded);
        rslt &= platformSelector->CompareTensors(
            PLATFORMS::CPU,
            scheduler,
            tensorPaddedGold,
            tensorPaddedTest);
    }*/
    { //Super-vec Padding
        const unsigned int lastDimPadded = 32;
        TensorF* tensorCpu = GenerateTensor(0,{8,16});
        TensorF* tensorPaddedGold = platformSelector->PadLastDim(PLATFORMS::CPU,scheduler,tensorCpu,lastDimPadded);
        TensorF* tensorPaddedTest = platformSelector->PadLastDim(PLATFORMS::GPU_OCL,scheduler,tensorCpu,lastDimPadded);
        rslt &= platformSelector->CompareTensors(
            PLATFORMS::CPU,
            scheduler,
            tensorPaddedGold,
            tensorPaddedTest);
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelUnpadLastDimFloat(){
    bool rslt = true;
    { //Super-vec Unadding
        const unsigned int lastDimUnpadded = 32;
        TensorF* tensorCpu = GenerateTensor(0,{2,8,128});
        TensorF* tensorPaddedGold = platformSelector->UnpadLastDim(PLATFORMS::CPU,scheduler,tensorCpu,lastDimUnpadded);
        TensorF* tensorPaddedTest = platformSelector->UnpadLastDim(PLATFORMS::GPU_OCL,scheduler,tensorCpu,lastDimUnpadded);
        rslt &= platformSelector->CompareTensors(
            PLATFORMS::CPU,
            scheduler,
            tensorPaddedGold,
            tensorPaddedTest);
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelConcat2(){
    TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,3});
    TensorF* tensorSrc2 = GenerateTensor(0,{2,2,2,3});

    TensorF* tensorCpu = platformSelector->Concat2(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,3);
    TensorF* tensorGpu = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,3);

    bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelSqrt(){
    TensorF* tensorSrc = GenerateTensor(3,{2,2,2,2});

    TensorF* tensorCpu = platformSelector->Sqrt(PLATFORMS::CPU,scheduler,tensorSrc);
    TensorF* tensorGpu = platformSelector->Sqrt(PLATFORMS::GPU_OCL,scheduler,tensorSrc);

    bool rslt = platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, rslt);
    return obj;
}


ReportObject* XilinxImpUnitTests::KernelReduceMax(){
    TensorF* tensorSrc1 = GenerateTensor(0,{2,2,3,32});
    TensorF* tensorCpu1 = platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc1,2);
    TensorF* tensorGpu1 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,2);
    bool comparisonResult1 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu1,tensorGpu1);

    TensorF* tensorSrc2 = GenerateTensor(0,{2,3,1,16});
    TensorF* tensorCpu2 = platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc2,1);
    TensorF* tensorGpu2 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc2,1);
    bool comparisonResult2 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu2,tensorGpu2);

    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult1 && comparisonResult2);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelReduceSum4D(){
    bool comparisonResult=true;

    //Rank4_TTTF
    {
        TensorF* tensorSrc = GenerateTensor(3,{2,2,16,64});
        TensorF* tensorCpu = platformSelector->ReduceSum4D(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }

    //Rank4_TTTF
    {
        TensorF* tensorSrc = GenerateTensor(4,{2,2,2,16});
        TensorF* tensorCpu = platformSelector->ReduceSum4D(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }

    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelReduceSum(){
    bool comparisonResult1,comparisonResult2,comparisonResult3;

    {
        //TEST(Rank3_OverAxis2)
        TensorF* tensorSrc3 = GenerateTensor(0,{2,3,17});
        TensorF* tensorCpu3 = platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc3,false,false,true);
        TensorF* tensorGpu3 = platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc3,false,false,true);
        comparisonResult1 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu3,tensorGpu3);
    }
    {
        //TEST(Rank3_OverAxis2)
        TensorF* tensorSrc3 = GenerateTensor(0,{2,16,60});
        TensorF* tensorCpu3 = platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc3,false,false,true);
        TensorF* tensorGpu3 = platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc3,false,false,true);
        comparisonResult2 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu3,tensorGpu3);
    }
    {
        //TEST(Rank3_OverAxis2)
        TensorF* tensorSrc3 = GenerateTensor(0,{2,17,16});
        TensorF* tensorCpu3 = platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc3,false,false,true);
        TensorF* tensorGpu3 = platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc3,false,false,true);
        comparisonResult3 = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu3,tensorGpu3);
    }

    bool comparisonResult = comparisonResult1 && comparisonResult2 && comparisonResult3;
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelTile(){
    bool comparisonResult=true;
    //TEST(Rank4_Axis2)
    {
        int tileCount = 8;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,17});
        TensorF* tensorCpu = platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }

    //TEST(Rank3_Axis2)
    {
        int tileCount = 7;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = GenerateTensor(0,{2,5});
        TensorF* tensorCpu = platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    
    //TEST(Rank3_Axis1)
    {
        int tileCount = 3;
        int tileAxis  = 1;
        TensorF* tensorSrc1 = GenerateTensor(0,{2,7});
        TensorF* tensorCpu = platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    //TEST(Rank3_Axis1)
    {
        int tileCount = 2;
        int tileAxis  = 1;
        TensorF* tensorSrc1 = GenerateTensor(0,{2,18});
        TensorF* tensorCpu = platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelTranspose(){
    TensorF* tensorSrc = GenerateTensor(0,{5,32,8});
    TensorF* tensorCpu = platformSelector->Transpose(PLATFORMS::CPU,scheduler,tensorSrc);
    TensorF* tensorGpu = platformSelector->Transpose(PLATFORMS::GPU_OCL,scheduler,tensorSrc);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelRelu(){
    TensorF* tensorSrc = GenerateTensor(7,{2,2,2});
    TensorF* tensorCpu = platformSelector->ReLU(PLATFORMS::CPU,scheduler,tensorSrc);
    TensorF* tensorGpu = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,tensorSrc);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelSquare(){
    TensorF* tensorSrc = GenerateTensor(3,{2,2,2});
    TensorF* tensorCpu = platformSelector->Square(PLATFORMS::CPU,scheduler,tensorSrc);
    TensorF* tensorGpu = platformSelector->Square(PLATFORMS::GPU_OCL,scheduler,tensorSrc);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelMatops(){
    bool comparisonResult=true;
    bool printLog=true;

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_4_4)");
    {
        //Ranks: 4,4
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2,2,2,2});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }

    /*
    if(printLog) cout << "TEST(Rank_4_3)"<<endl;
    {
        //Ranks: 4,4
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,17});
        TensorF* tensorSrc2 = GenerateTensor(0,{2,2,17});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }

    if(printLog) cout << "TEST(Rank_4_3)"<<endl;
    {
        //Ranks: 4,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(3,{2,2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(5,{2,2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) cout << "TEST(Rank_4_2)"<<endl;
    {
        //Ranks: 4,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(3,{2,2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(3,{2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }
    */
    
    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_4_1)");
    {
        //Ranks: 4,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_4_1)");
    {
        //Ranks: 4,4
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,17});
        TensorF* tensorSrc2 = GenerateTensor(0,{17});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_4_0)");
    {
        //Ranks: 4,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_3_3)");
    {
        //Ranks: 3,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2,2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    /*
    if(printLog) cout << "TEST(Rank_3_2)"<<endl;
    {
        //Ranks: 3,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(3,{2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(3,{2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }
    */

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_3_1)");
    {
        //Ranks: 3,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_3_0)");
    {
        //Ranks: 3,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_2_2)");
    {
        //Ranks: 2,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    
    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_2_1)");
    {
        //Ranks: 2,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{5,9});
        TensorF* tensorSrc2 = GenerateTensor(0,{9});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }
    
    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_2_0)");
    {
        //Ranks: 2,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2,2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_1_1)");
    {
        //Ranks: 1,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2});
        TensorF* tensorSrc2 = GenerateTensor(0,{2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,tensorSrc2, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,tensorGpu);
        }
    }

    if(printLog) SPDLOG_LOGGER_INFO(logger,"TEST(Rank_1_0)");
    {
        //Ranks: 1,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{2});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f, op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f, op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
        }
    }

    /*
    if(printLog) cout << "TEST(Rank_1_0V2)"<<endl;
    {
        //Ranks: 1,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(3,{64});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,op);
            TensorF *tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu, tensorGpu);
        }
    }

    if(printLog) cout << "TEST(ETC_1)"<<endl;
    {
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{5,1024,1024});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,0.5f,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,0.5f,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }

    if(printLog) cout << "TEST(ETC_2)"<<endl;
    {
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{5,1024,1024});
        TensorF* tensorSrc2 = GenerateTensor(0,{5,1024,1024});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }

    if(printLog) cout << "TEST(ETC_3)"<<endl;
    {
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = GenerateTensor(0,{8,256,256});
        TensorF* tensorSrc2 = GenerateTensor(0,{8,256,256});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
        }
    }*/

    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelMean(){
    bool comparisonResult = true;
    {
        TensorF* tensorSrc = GenerateTensor(0,{2,2,2,5});
        TensorF* tensorCpu = platformSelector->Mean(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = platformSelector->Mean(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }

    {
        TensorF* tensorSrc = GenerateTensor(0,{2,2,2,17});
        TensorF* tensorCpu = platformSelector->Mean(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = platformSelector->Mean(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelVariance(){
    TensorF* tensorSrc = GenerateTensor(0,{2,2,2,5});
    TensorF* tensorCpu = platformSelector->Variance(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
    TensorF* tensorGpu = platformSelector->Variance(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelMatmul(){
    bool comparisonResult = true;
    {
        TensorF* tensorSrc1 = GenerateTensor(0,{1,4,16});
        TensorF* tensorSrc2 = GenerateTensor(0,{1,16,16});
        TensorF* tensorCpu = platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    {
        TensorF* tensorSrc1 = GenerateTensor(3,{3,4});
        TensorF* tensorSrc2 = GenerateTensor(3,{4,5});
        TensorF* tensorCpu = platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    {
        TensorF* tensorSrc1 = GenerateTensor(0,{5,3,64});
        TensorF* tensorSrc2 = GenerateTensor(0,{5,64,5});
        TensorF* tensorCpu = platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    {
        TensorF* tensorSrc1 = GenerateTensor(0,{5,17,31});
        TensorF* tensorSrc2 = GenerateTensor(0,{5,31,15});
        TensorF* tensorCpu = platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelConv2Mlp(){
    if(platformSelector->openclPlatformClass->GetModeEnvVar()!=RUN_MODE::HwEmu){
        ReportObject* obj = new ReportObject(__FUNCTION__, false);
        SPDLOG_LOGGER_WARN(logger,"Skipping KernelConv2Mlp, only HwEmu mode is supported");
        return obj;
    }
    bool comparisonResult = true;
    {
        TensorF* tensorSrc = GenerateTensor(0,{1,256,1,6});
        TensorF* tensorWeight = GenerateTensor(0,{1,1,6,16});
        TensorF* tensorBiases = GenerateTensor(0,{128});
        TensorF* tensorCpu = platformSelector->Conv2D(PLATFORMS::CPU,scheduler,tensorSrc,tensorWeight,tensorBiases);
        TensorF* tensorGpu = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,tensorWeight,tensorBiases);
        comparisonResult &= platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu); 
    }
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelTopK(){
    if(platformSelector->openclPlatformClass->GetModeEnvVar()!=RUN_MODE::HwEmu){
        ReportObject* obj = new ReportObject(__FUNCTION__, false);
        SPDLOG_LOGGER_WARN(logger,"Skipping KernelTopK, only HwEmu mode is supported");
        return obj;
    }
    const unsigned kVal=20 , N=ConfigTaskTopK::MaxSliceLen , B=ConfigTaskTopK::UnitCount+2;
    SPDLOG_LOGGER_INFO(logger,"Please confirm that TOPK kernel is configured for K={} and N={}, Press ESC to skip...",kVal,N);    
    if(cin.get()==27) return nullptr;
    assert(N==ConfigTaskTopK::MaxSliceLen);

    TensorF *tensorSrc = GenerateTensor(0, {B, N, N});
    TensorI *tensorCpu = platformSelector->TopK(PLATFORMS::CPU, scheduler, tensorSrc, 2, kVal);
    TensorI *tensorGpu = platformSelector->TopK(PLATFORMS::GPU_OCL, scheduler, tensorSrc, 2, kVal);
    TensorI *tensorGpuTransfered = platformSelector->CrossThePlatform(tensorGpu,PLATFORMS::CPU);

    bool comparisonResult = true;
    if (tensorCpu->getShape() != tensorGpu->getShape()){
        comparisonResult = false;
    }
    else{

        for(int b=0;b<B;b++){
            for(int n1=0;n1<N;n1++){
                for(int kk=0;kk<kVal;kk++){
                    unsigned int i = b*N*kVal + n1*kVal + kk;
                    int rCpu = tensorCpu->_buff[i];
                    int rUdt = tensorGpuTransfered->_buff[i];
                    SPDLOG_LOGGER_INFO(
                        logger,
                        "Index(B,N,K)= ({},{},{}), rCPU={}, rUDT={}, Value[rCPU]={}, Value[rUDT]={}",
                        b,n1,kk,
                        rCpu,rUdt,
                        tensorSrc->_buff[b*N*N+ n1*N+ rCpu], tensorSrc->_buff[b*N*N+ n1*N+ rUdt]);

                    /*cout <<
                            "Index(B,N,K)= ("<< b <<", "<<n1<<", "<<kk<<")   " <<
                            " ,iCPU= "<<rCpu<<" ,iGPU= " <<rGpu<< ",   "
                            "Value[iCPU]= " << tensorSrc->_buff[b*N*N+ n1*N+ rCpu] << ", "
                            "Value[iGPU]= " << tensorSrc->_buff[b*N*N+ n1*N+ rGpu] <<
                            endl;*/
                    if(rCpu != rUdt){
                        comparisonResult=false;
                    }
                }
            }
        }

    }
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

ReportObject* XilinxImpUnitTests::KernelGather(){
    TensorF* tensorSrc = GenerateTensor(7,{5,5,2});
    TensorI* tensorIndices = GenerateTensorInteger(0,5,{5,5,3});
    TensorF* tensorCpu = platformSelector->Gather(PLATFORMS::CPU,scheduler,tensorSrc,tensorIndices,1);
    TensorF* tensorGpu = platformSelector->Gather(PLATFORMS::GPU_OCL,scheduler,tensorSrc,tensorIndices,1);
    bool comparisonResult = platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
    ReportObject* obj = new ReportObject(__FUNCTION__, comparisonResult);
    return obj;
}

void XilinxImpUnitTests::RunAll(){
    
    PrintReport(TensorFloat());
    PrintReport(TensorBankFloat());
    PrintReport(TensorBankInteger());
    PrintReport(TensorCloneBankFloat());
    PrintReport(TensorCloneBankInteger());
    PrintReport(TensorPadUnpadCpuFloat());
    PrintReport(TensorPadUnpadCpuInteger());
    PrintReport(KernelPadLastDimFloat());
    PrintReport(KernelUnpadLastDimFloat());
    //PrintReport(KernelConv2Mlp());
    //PrintReport(KernelTopK());
    //PrintReport(KernelMatops());
    PrintReport(KernelReduceSum4D());
    PrintReport(KernelReduceMax());
    PrintReport(KernelReduceSum());
    PrintReport(KernelMean());
    PrintReport(KernelVariance());
    //PrintReport(KernelMatmul());
    PrintReport(KernelTile());
    PrintReport(KernelGather());
    PrintReport(KernelConcat2());
    PrintReport(KernelRelu());
    PrintReport(KernelSqrt());
    PrintReport(KernelSquare());
    //PrintReport(KernelTranspose());
    
}

XilinxImpUnitTests::~XilinxImpUnitTests(){
    SPDLOG_LOGGER_TRACE(logger,"~XilinxImpUnitTests");
    delete(platformSelector);
}


