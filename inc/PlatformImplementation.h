//
// Created by saleh on 8/22/18.
//

#pragma once

#include "../inc/WorkScheduler.h"
#include "../inc/TensorF.h"
#include "../inc/TensorI.h"
#include <string>
using namespace std;

enum class MAT_OPS{
    ADD,
    SUB,
    MUL_ELEMENTWISE,
    DIV_ELEMENTWISE
};

class PlatformImplementation {
public:
    virtual TensorF* Transpose(WorkScheduler scheduler, TensorF *batchedMat)=0;
    virtual TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2)=0;
    //virtual TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar)=0;
    virtual TensorF* Square(WorkScheduler scheduler, TensorF* batchedMat)=0;
    virtual TensorF* ReduceSum(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2)=0;
    virtual TensorF* ReduceSum4D(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3)=0;
    virtual TensorF* Mean(WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3)=0;
    virtual TensorF* Variance(WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3)=0;
    virtual TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode)=0;
    virtual TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode)=0;
    virtual TensorF* Sqrt(WorkScheduler scheduler, TensorF* inputTn)=0;
    virtual TensorF* Concat2(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim)=0;
    virtual TensorF* ReduceMax(WorkScheduler scheduler, TensorF* inputTn, int reductionDim)=0;
    virtual TensorI* TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k)=0;
    virtual TensorF* Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis)=0;
    virtual TensorF* Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2)=0;
    virtual TensorF* ReLU(WorkScheduler scheduler, TensorF* inputTn)=0;
    virtual TensorF* Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount)=0;
    virtual void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir)=0;
    virtual void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir)=0;
    virtual bool     CompareTensors(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2)=0;
    virtual bool     CompareTensorsInteger(WorkScheduler scheduler, TensorI* inputTn1, TensorI* inputTn2)=0;
    virtual TensorF* PadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimPadded)=0;
    virtual TensorF* UnpadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimUnpadded)=0;

private:


};
