#pragma once

#include <iostream>
#include <ctime>
#include "Layers.h"
#include "FakeTensorF.h"
#include "FakeTensorI.h"

using namespace std;

struct ModelInfo{
    string Version="";
    string ModelType="";
    string DesignNotes="";
    string ExperimentNotes="";
    string ToDo="";
    string Date="";
};

class ModelArchTop01 {
public:
    ModelArchTop01(int dataset_offset, int batchsize, int pointcount, int knn_k);
    TensorF*    FullyConnected_Forward(WorkScheduler scheduler, TensorF* input_BxD, TensorF* weights, TensorF* biases);
    TensorF*    Batchnorm_Forward(WorkScheduler scheduler, TensorF* input, TensorF* gamma, TensorF* beta, TensorF* ema_ave, TensorF* ema_var, int rankOverride);
    TensorF*    GetEdgeFeatures(WorkScheduler scheduler, TensorF* input_BxNxD, TensorI* knn_output_BxNxK);
    TensorF*    PairwiseDistance(WorkScheduler scheduler, TensorF* input_BxNxD);
    TensorF*    TransformNet(WorkScheduler scheduler, TensorF* edgeFeatures);
    TensorF*    Execute(WorkScheduler scheduler);
    int GetDataMoverLaunches();
    Layers* platformSelector;

private:
    unsigned B=-1;
    unsigned N=-1;
    unsigned K=-1;
    TensorF* input_pcl_BxNxD;
    TensorI* input_labels_B;
    int DB_OFFSET=-1;

};
