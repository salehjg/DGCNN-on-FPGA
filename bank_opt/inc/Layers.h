#pragma once

#include <vector>
#include <string>

#include "FakeTensorF.h"
#include "FakeTensorI.h"
#include "Helper.h"

using namespace std;

class WorkScheduler{};

class Layers{
public:
    Layers(PLATFORMS defaultPlatform , vector<PLATFORMS> neededPlatforms, bool loadWeights);

    TensorF* Transpose(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    //TensorF* MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, float scalar);
    TensorF* Square(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, MAT_OPS mode);
    TensorF* MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode);
    TensorF* Sqrt(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Concat2(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, int reductionDim);

    TensorI* TopK(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2=-1);
    TensorF* ReLU(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    TensorF* PadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimPadded);
    TensorF* UnpadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimUnpadded);

    TensorF* CreateDummyTensorF(int bank, string tag);
    TensorI* CreateDummyTensorI(int bank, string tag);
    int dataMoverLaunches=0;
    vector<string> objective;

private:
    void ChangeBankIfNeeded(TensorF* tn, int dstBank);
    void ChangeBankIfNeeded(TensorI* tn, int dstBank);


};