//
// Created by saleh on 9/3/18.
//
#include <ModelArchTop01.h>
#include <iostream>
#include "build_config.h"

using namespace std;

void CalculateAccuracy(TensorF* scores, TensorI* labels, int B, int classCount){
    //find argmax(net) and compute bool array of corrects.
    bool *correct = new bool[B];
    float accu =0;

    SPDLOG_LOGGER_INFO(logger,"Computing Accuracy...");
    {
        float max_cte = -numeric_limits<float>::infinity();
        float max = 0;
        int max_indx=-1;
        int *a1 = new int[B];


        for(int b=0;b<B;b++){

            max = max_cte;
            for(int c=0;c<classCount;c++){
                if(max   <   scores->_buff[b*classCount+c]  ){
                    max = scores->_buff[b*classCount+c];
                    max_indx = c;
                }
            }

            //set maximum score for current batch index
            a1[b]=max_indx;
        }

        for(int b=0;b<B;b++){
            if(a1[b]==(int)labels->_buff[b]){
                correct[b]=true;
            }
            else{
                correct[b]=false;
            }
        }

        free(a1);
    }
    //----------------------------------------------------------------------------------------
    // compute accuracy using correct array.
    {
        float correct_cnt=0;
        for(int b=0;b<B;b++){
            if(correct[b]==true) correct_cnt++;
        }
        accu = correct_cnt / (float)B;

        SPDLOG_LOGGER_INFO(logger,"Correct Count: {}", correct_cnt);
        SPDLOG_LOGGER_INFO(logger,"Accuracy: {}", accu);
    }
    
}

void ClassifierMultiplatform(){
    WorkScheduler scheduler;
    ModelArchTop01 modelArchTop(0,globalBatchsize,1024,20);
    std::string pclPath = globalArgDataPath; pclPath.append("/dataset/dataset_B2048_pcl.npy");
    std::string labelPath = globalArgDataPath; labelPath.append("/dataset/dataset_B2048_labels_int32.npy");

    SPDLOG_LOGGER_INFO(logger,"PCL NPY PATH: {}", pclPath);
    SPDLOG_LOGGER_INFO(logger,"LBL NPY PATH: {}", labelPath);

    modelArchTop.SetModelInput_data(pclPath.c_str());
    modelArchTop.SetModelInput_labels(labelPath.c_str());

    double timerStart = seconds();
    TensorF* classScores = modelArchTop.Execute(scheduler);
    SPDLOG_LOGGER_INFO(logger,"Model execution time with batchsize({}): {} Seconds", globalBatchsize, (seconds() -timerStart));

    CalculateAccuracy(classScores,modelArchTop.GetLabels(),modelArchTop.GetBatchSize(),40);
}

