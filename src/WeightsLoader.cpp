//
// Created by saleh on 8/30/18.
//

#include <fstream>
#include <algorithm>
#include "../inc/WeightsLoader.h"
#ifdef USE_OCL
#include <CL/cl.h>
#endif

WeightsLoader::WeightsLoader(vector<PLATFORMS> neededPlatforms) {
    for(std::vector<PLATFORMS>::iterator it = neededPlatforms.begin(); it != neededPlatforms.end(); ++it) {
        switch(*it){
            case PLATFORMS::CPU : {
                _isUsedCPU = true;
                break;
            }
#ifdef USE_CUDA
            case PLATFORMS::GPU_CUDA : {
                _isUsedCUDA = true;
                break;
            }
#endif
#ifdef USE_OCL
            case PLATFORMS::GPU_OCL : {
                _isUsedOCL = true;
                break;
            }
#endif
        }
    }
}
#ifdef USE_OCL
void WeightsLoader::LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList, cl::Context* oclContex, cl::CommandQueue* oclQueue) {
#else
void WeightsLoader::LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList) {
#endif
    string line; int idx=-1;

    std::ifstream inFile(pathToTxtFnameList);
    long weightCount = std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');
    weightsCPU  = new TensorF*[weightCount];
    weightsCUDA = new TensorF*[weightCount];
    weightsOCL  = new TensorF*[weightCount];

    ifstream txtfile (pathToTxtFnameList);
    if (!txtfile.is_open()) {
        cout<<"Failed to open text file!";
        return;
    }


    while (std::getline(txtfile, line)) {
        idx++;
        string weight_npy_path = weightsBaseDir + line;
        strToIndexMap.insert( std::make_pair(line, idx) );
        _cnpyBuff.push_back(cnpy::npy_load(weight_npy_path));
        vector<unsigned int> __shape(_cnpyBuff.back().shape.begin(),_cnpyBuff.back().shape.end());

        if(_isUsedCPU){

            weightsCPU[idx] = new TensorF(
                    __shape,
                    _cnpyBuff.back().data<float>()
            );
        }

#ifdef USE_CUDA
        if(_isUsedCUDA){
            weightsCUDA[idx] = new CudaTensorF();
            ((CudaTensorF*)weightsCUDA[idx])->InitWithHostData(__shape, _cnpyBuff.back().data<float>());
        }
#endif

#ifdef USE_OCL
        if(_isUsedOCL){
            if(__shape.size()==1 && __shape[0]==0) continue;
            weightsOCL[idx] = new OclTensorF();
            ((OclTensorF*)weightsOCL[idx])->InitWithHostData(oclContex,oclQueue,__shape,_cnpyBuff.back().data<float>());

        }
#endif
    }

    txtfile.close();
}

/*
void WeightsLoader::LoadFromDiskhhh(string weightsBaseDir, string pathToTxtFnameList) {
    std::ifstream inFile(pathToTxtFnameList);
    long weightCount = std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');
    weightsCPU = new vector<TensorF*>(weightCount);
    weightsCUDA = new vector<TensorF*>(weightCount);
    weightsOCL = new vector<TensorF*>(weightCount);

    ifstream txtfile (pathToTxtFnameList);
    if (!txtfile.is_open())
    {
        cout<<"Failed to open text file!";
        return;
    }

    string line; int idx=-1;

    while (std::getline(txtfile, line)) {
        string weight_npy_path = weightsBaseDir + line;
        cnpy::NpyArray current = cnpy::npy_load(weight_npy_path);
        vector<unsigned int> __shape(current.shape.begin(),current.shape.end());
        idx++;
        strToIndexMap.insert( std::make_pair(line, idx) );

        if(_isUsedCPU){
            (*weightsCPU)[idx] = new TensorF(__shape, current.data<float>()) ;
            if(line == "transform_net1.tconv1.weights.npy") {
                for (int i = 0; i < 64; i++) {
                    //cout << current.data<float>()[i] << endl;
                    //cout << tmpTensor->_buff[i] << endl;
                    cout << (*weightsCPU)[idx]->_buff[i] << endl;
                }
            }
        }

        if(_isUsedCUDA){
            CudaTensorF cudaWeight;
            cudaWeight.InitWithHostData(__shape, current.data<float>());
            //weightsCUDA->insert(weightsCUDA->begin()+idx, cudaWeight);
            //(*weightsCUDA)[idx] = new TensorF(__shape, current.data<float>()) ;
        }

        if(_isUsedOCL){
            throw "Not Implemented.";
        }
    }

    txtfile.close();
}
*/

TensorF* WeightsLoader::AccessWeights(PLATFORMS platform, string name) {
    switch(platform){
        case PLATFORMS::CPU:{
            return weightsCPU[strToIndexMap[name]];
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA:{
            return weightsCUDA[strToIndexMap[name]];
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL:{
            return weightsOCL[strToIndexMap[name]];
            break;
        }
#endif
        default:{
            throw "Unknown Platform.";
            break;
        }
    }
}
