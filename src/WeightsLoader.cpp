//
// Created by saleh on 8/30/18.
//

#include <fstream>
#include <algorithm>
#include <string>
#include <cassert>
#include "xilinx/config.h"
#include "../inc/WeightsLoader.h"
#ifdef USE_OCL
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
void WeightsLoader::LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList, cl::Context *oclContex, cl::CommandQueue *oclQueue) {
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
        SPDLOG_LOGGER_ERROR(logger,"Failed to open text file (WeightsLoader::LoadFromDisk)");
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
            int bank = ResolveMemoryBank(PLATFORMS::GPU_OCL, line);
            ((OclTensorF*)weightsOCL[idx])->InitWithHostData(oclContex,oclQueue,__shape,_cnpyBuff.back().data<float>(),bank);

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
            assert(false);
            break;
        }
    }
}

int WeightsLoader::ResolveMemoryBank(PLATFORMS platform, string name){
    switch(platform){
        case PLATFORMS::CPU:{
            assert(false);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA:{
            assert(false);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL:{
            return _ResolveMemoryBankOclXilinx(name);
            break;
        }
#endif
        default:{
            assert(false);
            break;
        }
    }
}


/**
 * @brief      Returns the right bank index for the given weight name based on the layer that is going to use it.
 *             The layers that are involved with the weights are:
 *                   1. Conv2
 *                   2. MatOps
 *                   3. MatMul 
 *             WARNING: This method considers that all of m_axi of the kernels are tied to the same memory bank. 
 *
 * @param[in]  name  The weight name
 *
 * @return     DDR bank index for xilinx ocl platform
 */
int WeightsLoader::_ResolveMemoryBankOclXilinx(string name){
    assert(
        ConfigTaskConv2::BankIndex_inputTn == ConfigTaskConv2::BankIndex_weightTn &&
        ConfigTaskConv2::BankIndex_weightTn == ConfigTaskConv2::BankIndex_biasTn &&
        ConfigTaskConv2::BankIndex_biasTn == ConfigTaskConv2::BankIndex_outputTn
    );

    assert(
        ConfigTaskMatOps::BankIndex_inputTn1 == ConfigTaskMatOps::BankIndex_inputTn2 &&
        ConfigTaskMatOps::BankIndex_inputTn2 == ConfigTaskMatOps::BankIndex_outputTn
    );

    assert(
        ConfigTaskMatMul::BankIndex_inputTn1 == ConfigTaskMatMul::BankIndex_inputTn2 &&
        ConfigTaskMatMul::BankIndex_inputTn2 == ConfigTaskMatMul::BankIndex_outputTn
    );

    int indexConv2 = ConfigTaskConv2::BankIndex_inputTn;
    int indexMatOps = ConfigTaskMatOps::BankIndex_inputTn1;
    int indexMatMul = ConfigTaskMatMul::BankIndex_inputTn1;

    bool isConv[] = {
        name == "transform_net1.tconv1.weights.npy",
        name == "transform_net1.tconv1.biases.npy",
        name == "transform_net1.tconv2.weights.npy",
        name == "transform_net1.tconv2.biases.npy",
        name == "transform_net1.tconv3.weights.npy",
        name == "transform_net1.tconv3.biases.npy",
        name == "dgcnn1.weights.npy",
        name == "dgcnn1.biases.npy",
        name == "dgcnn2.weights.npy",
        name == "dgcnn2.biases.npy",
        name == "dgcnn3.weights.npy",
        name == "dgcnn3.biases.npy",
        name == "dgcnn4.weights.npy",
        name == "dgcnn4.biases.npy",
        name == "agg.weights.npy",
        name == "agg.biases.npy"
    };

    bool isMatOps[] = {
        name == "transform_net1.tconv1.bn.gamma.npy",
        name == "transform_net1.tconv1.bn.beta.npy",
        name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "transform_net1.tconv2.bn.gamma.npy",
        name == "transform_net1.tconv2.bn.beta.npy",
        name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "transform_net1.tconv3.bn.gamma.npy",
        name == "transform_net1.tconv3.bn.beta.npy",
        name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "transform_net1.tfc1.bn.gamma.npy",
        name == "transform_net1.tfc1.bn.beta.npy",
        name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "transform_net1.tfc2.bn.gamma.npy",
        name == "transform_net1.tfc2.bn.beta.npy",
        name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "dgcnn1.bn.gamma.npy",
        name == "dgcnn1.bn.beta.npy",
        name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "dgcnn2.bn.gamma.npy",
        name == "dgcnn2.bn.beta.npy",
        name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "dgcnn3.bn.gamma.npy",
        name == "dgcnn3.bn.beta.npy",
        name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "dgcnn4.bn.gamma.npy",
        name == "dgcnn4.bn.beta.npy",
        name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "agg.bn.gamma.npy",
        name == "agg.bn.beta.npy",
        name == "agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "fc1.bn.gamma.npy",
        name == "fc1.bn.beta.npy",
        name == "fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "fc2.bn.gamma.npy",
        name == "fc2.bn.beta.npy",
        name == "fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
        name == "fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
        name == "transform_net1.tfc1.biases.npy",
        name == "transform_net1.tfc2.biases.npy",
        name == "fc1.biases.npy",
        name == "fc2.biases.npy",
        name == "fc3.biases.npy"
    };

    bool isMatMul[] = {
        name == "transform_net1.tfc1.weights.npy",
        name == "transform_net1.tfc2.weights.npy",
        name == "fc1.weights.npy",
        name == "fc2.weights.npy",
        name == "fc3.weights.npy"
    };

    bool rsltConv = false;
    bool rsltMatOps = false;
    bool rsltMatMul = false;

    for(bool item:isConv){
        rsltConv = rsltConv | item;
    }

    for(bool item:isMatOps){
        rsltMatOps = rsltMatOps | item;
    }

    for(bool item:isMatMul){
        rsltMatMul = rsltMatMul | item;
    }

    if(rsltConv){
        SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"Conv2\" and will be transfered to the DDR bank {}", name, indexConv2);
        return indexConv2;
    }

    if(rsltMatOps){
        SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"MatOps\" and will be transfered to the DDR bank {}", name, indexMatOps);
        return indexMatOps;
    }

    if(rsltMatMul){
        SPDLOG_LOGGER_DEBUG(logger,"The weight tensor \"{}\" is considered to be related to the layer \"MatMul\" and will be transfered to the DDR bank {}", name, indexMatMul);
        return indexMatMul;
    }
}