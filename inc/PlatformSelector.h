//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_PLATFORMSELECTOR_H
#define DEEPPOINTV1_PLATFORMSELECTOR_H

#include <vector>
#include "../inc/TensorF.h"
#include "../inc/TensorI.h"
#include "../inc/WeightsLoader.h"
#include "../inc/PlatformImplementation.h"
#include <build_config.h>
#ifdef USE_OCL
#include "ocl_imp/xilinx/XilinxImplementation.h"
#include <CL/cl.h>
#endif
#include <build_config.h>
using namespace std;


class PlatformSelector {
public:
    PlatformSelector(PLATFORMS defaultPlatform , vector<PLATFORMS> neededPlatforms, bool loadWeights);
    TensorF* CrossThePlatform(TensorF* srcTn, PLATFORMS platform);
    TensorI* CrossThePlatform(TensorI* srcTn, PLATFORMS platform);
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

    void     DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir=REPO_DIR"/data/matrix_dumps/");
    void     DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir=REPO_DIR"/data/matrix_dumps/");
    bool     CompareTensors(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    bool     CompareTensorsInteger(PLATFORMS platform, WorkScheduler scheduler, TensorI* inputTn1, TensorI* inputTn2);
    ~PlatformSelector();

    WeightsLoader* weightsLoader;
    PLATFORMS defaultPlatform;
    PlatformImplementation *cpuPlatformClass;
    PlatformImplementation *cudaPlatformClass;
    #ifdef USE_OCL
    XilinxImplementation *openclPlatformClass;
    #endif
private:

};


#endif //DEEPPOINTV1_PLATFORMSELECTOR_H
