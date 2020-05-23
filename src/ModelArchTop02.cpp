#include "build_config.h"
#include "ModelArchTop02.h"

ModelArchTop02::ModelArchTop02(int dataset_offset, int batchsize, int pointcount, int knn_k) {
    platformSelector = new PlatformSelector(PLATFORMS::GPU_OCL,{PLATFORMS::CPU,PLATFORMS::GPU_OCL},true);
    DB_OFFSET = dataset_offset;
    B = (unsigned)batchsize;
    N = (unsigned)pointcount;
    K = (unsigned)knn_k;
#ifndef USE_OCL
    SPDLOG_LOGGER_ERROR(logger,"This ModelArch needs OpenCL enabled, Terminating...");
    exit(1);
#endif
}

ModelInfo ModelArchTop02::GetModelInfo() {
    ModelInfo tmplt;
    tmplt.ModelType="Classifier";
    tmplt.Version="2.0";
    tmplt.DesignNotes=
            "1) FPGA only.\n"
            "2) Weights are on FPGA.";
    tmplt.ExperimentNotes="";
    tmplt.ToDo=""
               "";
    tmplt.Date="99.3.2";
    return tmplt;
}

void ModelArchTop02::SetModelInput_data(string npy_pcl) {
    _npy_pcl = cnpy::npy_load(npy_pcl);
    input_pcl_BxNxD = new TensorF({B,N,3}, _npy_pcl.data<float>());
    input_pcl_BxNxD += DB_OFFSET*N*3;///TODO: CHECK COMPATIBILITY FOR GPU TENSOR(CUDA & OCL)
    assert( N == (int)(_npy_pcl.shape[1]) );
}

void ModelArchTop02::SetModelInput_labels(string npy_labels) {
    // dataType of npy file should be int32, NOT uchar8!
    // use dataset_B5_labels_int32.npy
    _npy_labels = cnpy::npy_load(npy_labels);
    input_labels_B = new TensorI({B},_npy_labels.data<int>());
    input_labels_B->_buff += DB_OFFSET;
}

TensorF* ModelArchTop02::FullyConnected_Forward(WorkScheduler scheduler, TensorF* input_BxD, TensorF* weights, TensorF* biases){
    int ch_out = (int)weights->getShape().back();
    TensorF* tmp = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,input_BxD,weights);
    TensorF* rsltTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tmp,biases,MAT_OPS::ADD);
    delete(tmp);
    return rsltTn;
}

TensorF* ModelArchTop02::Batchnorm_Forward(WorkScheduler scheduler, TensorF* input, TensorF* gamma, TensorF* beta, TensorF* ema_ave, TensorF* ema_var){
    int rank;
    float bn_decay = 0.5f;


    TensorF* mu;
    TensorF* var;

    rank = input->getRank();

    ///TODO: Rank4 and Rank2 branches could be merged into 1 branch with just mu and var lines conditioned!
    if(rank==4){
        /*
        {// JUST FOR DEBUGGING --- STILL RAISES EXCEPTION ON COMPARE TENSORS FOR REDUCESUM4D
                platformSelector->DumpMatrix(PLATFORMS::GPU_OCL,scheduler,"dbg4d_src.npy",input);
                TensorF * dbgTnCPU = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
                TensorF * dbgTn = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
                assert(platformSelector->CompareTensors(PLATFORMS::GPU_OCL,scheduler,dbgTnCPU,dbgTn));
                platformSelector->DumpMatrix(PLATFORMS::GPU_OCL,scheduler,"dbg4d_dst.npy",dbgTn);
        }*/

        //mu and var is of shape (dim3)
        mu = platformSelector->Mean(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
        /*
        {// JUST FOR DEBUGGING
            TensorF *mu_cpu = platformSelector->Mean(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
            assert(platformSelector->CompareTensors(PLATFORMS::GPU_OCL,scheduler,mu,mu_cpu));
        }*/

        var = platformSelector->Variance(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);

        // Exponential Moving Average for mu and var
        TensorF *update_delta_ave, *update_delta_var;
        TensorF *update_delta_ave2, *update_delta_var2;

        update_delta_ave = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_ave, mu, MAT_OPS::SUB);
        update_delta_ave2 = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, update_delta_ave, bn_decay, MAT_OPS::MUL_ELEMENTWISE);
        update_delta_var = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_var, var, MAT_OPS::SUB);
        update_delta_var2 = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, update_delta_var, bn_decay, MAT_OPS::MUL_ELEMENTWISE);

        TensorF *final_ave =  platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_ave, update_delta_ave2, MAT_OPS::SUB);
        TensorF *final_var =  platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_var, update_delta_var2, MAT_OPS::SUB);

        TensorF* xNormTmp1 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,input,final_ave, MAT_OPS::SUB);
        TensorF* xNormTmp2 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,final_var,1e-8, MAT_OPS::ADD);
        TensorF* xNormTmp3 = platformSelector->Sqrt(PLATFORMS::GPU_OCL,scheduler,xNormTmp2);
        TensorF* xNorm = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,xNormTmp1,xNormTmp3, MAT_OPS::DIV_ELEMENTWISE);
        TensorF* rsltTmp1 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,xNorm,gamma, MAT_OPS::MUL_ELEMENTWISE);
        TensorF* rsltTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,rsltTmp1,beta, MAT_OPS::ADD);

        delete(mu);
        delete(var);
        delete(update_delta_ave);
        delete(update_delta_ave2);
        delete(update_delta_var);
        delete(update_delta_var2);
        delete(final_ave);
        delete(final_var);
        delete(xNormTmp1);
        delete(xNormTmp2);
        delete(xNormTmp3);
        delete(xNorm);
        delete(rsltTmp1);

        return rsltTn;
    }

    if(rank==2){
        /*
        {// JUST FOR DEBUGGING
            platformSelector->DumpMatrix(PLATFORMS::GPU_OCL,scheduler,"dbg2d_src.npy",input);
            input->ExpandDimZero();
            input->ExpandDimZero();
            TensorF * dbgTnCPU = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
            TensorF * dbgTn = platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL, scheduler, input, true, true, true, false);
            assert(platformSelector->CompareTensors(PLATFORMS::GPU_OCL,scheduler,dbgTnCPU,dbgTn));
            input->SqueezeDimZero();
            input->SqueezeDimZero();
            platformSelector->DumpMatrix(PLATFORMS::GPU_OCL,scheduler,"dbg2d_dst.npy",dbgTn);
        }*/

        //mu and var is of shape (dim1)
        mu = platformSelector->Mean(PLATFORMS::GPU_OCL, scheduler, input, true, false, false, false);
        /*
        {// JUST FOR DEBUGGING
            TensorF *mu_cpu = platformSelector->Mean(PLATFORMS::GPU_OCL, scheduler, input, true, false, false, false);
            assert(platformSelector->CompareTensors(PLATFORMS::GPU_OCL,scheduler,mu,mu_cpu));
        }*/

        var = platformSelector->Variance(PLATFORMS::GPU_OCL, scheduler, input, true, false, false, false);

        // Exponential Moving Average for mu and var
        TensorF *update_delta_ave, *update_delta_var;
        TensorF *update_delta_ave2, *update_delta_var2;

        update_delta_ave = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_ave, mu, MAT_OPS::SUB);
        update_delta_ave2 = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, update_delta_ave, bn_decay, MAT_OPS::MUL_ELEMENTWISE);
        update_delta_var = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_var, var, MAT_OPS::SUB);
        update_delta_var2 = platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, update_delta_var, bn_decay, MAT_OPS::MUL_ELEMENTWISE);

        TensorF *final_ave =  platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_ave, update_delta_ave2, MAT_OPS::SUB);
        TensorF *final_var =  platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, ema_var, update_delta_var2, MAT_OPS::SUB);

        TensorF* xNormTmp1 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,input,final_ave, MAT_OPS::SUB);
        TensorF* xNormTmp2 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,final_var,1e-8, MAT_OPS::ADD);
        TensorF* xNormTmp3 = platformSelector->Sqrt(PLATFORMS::GPU_OCL,scheduler,xNormTmp2);
        TensorF* xNorm = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,xNormTmp1,xNormTmp3, MAT_OPS::DIV_ELEMENTWISE);
        TensorF* rsltTmp1 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,xNorm,gamma, MAT_OPS::MUL_ELEMENTWISE);
        TensorF* rsltTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,rsltTmp1,beta, MAT_OPS::ADD);

        delete(mu);
        delete(var);
        delete(update_delta_ave);
        delete(update_delta_ave2);
        delete(update_delta_var);
        delete(update_delta_var2);
        delete(final_ave);
        delete(final_var);
        delete(xNormTmp1);
        delete(xNormTmp2);
        delete(xNormTmp3);
        delete(xNorm);
        delete(rsltTmp1);

        return rsltTn;
    }
    return nullptr;
}

TensorF* ModelArchTop02::GetEdgeFeatures(WorkScheduler scheduler, TensorF *input_BxNxD, TensorI *knn_output_BxNxK) {
    //Gather knn's indices from input array.
    TensorF* point_cloud_neighbors = platformSelector->Gather(PLATFORMS::GPU_OCL,scheduler,input_BxNxD,knn_output_BxNxK,1);

    //tile ing input of shape BxNxD into BxNxKxD..
    //input_BxNxD->ExpandDims(2);
    TensorF* point_cloud_central = platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,input_BxNxD,2,K);
    //input_BxNxD->SqueezeDims();

    TensorF* features = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler, point_cloud_neighbors,point_cloud_central,MAT_OPS::SUB);

    //concatenate centrals and features (BxNxKxD) and (BxNxKxD)
    TensorF* edge_feature = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler, point_cloud_central,features,3);

    delete(point_cloud_neighbors);
    delete(point_cloud_central);
    delete(features);

    return edge_feature;
}

TensorF* ModelArchTop02::PairwiseDistance(WorkScheduler scheduler, TensorF *input_BxNxD) {
    /*
    TensorF* point_cloud_transpose = platformSelector->Transpose(PLATFORMS::CPU,scheduler,input_BxNxD);
    TensorF* point_cloud_inner =  platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,input_BxNxD,point_cloud_transpose);
    TensorF* point_cloud_inner2 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,point_cloud_inner,-2.0f,MAT_OPS::MUL_ELEMENTWISE);
    TensorF* point_cloud_inner2p2 = platformSelector->Square(PLATFORMS::GPU_OCL,scheduler,input_BxNxD);
    TensorF* point_cloud_sum = platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,point_cloud_inner2p2,false,false,true);
    point_cloud_sum->ExpandDims(-1);
    TensorF* point_cloud_sum_transpose = platformSelector->Transpose(PLATFORMS::CPU,scheduler,point_cloud_sum);  //changed dims
    point_cloud_sum_transpose->SqueezeDims();
    point_cloud_sum->SqueezeDims();
    TensorF* point_cloud_sum_tiled =  platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum,2,N); //result is BxNxK for k=N
    TensorF* point_cloud_sum_transpose_tiled =  platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum_transpose,1,N); //result is BxkxN for k=N
    TensorF* rsltTmpTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum_tiled,point_cloud_sum_transpose_tiled, MAT_OPS::ADD); //both input tensors are BxNxN
    TensorF* rsltTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,rsltTmpTn,point_cloud_inner2, MAT_OPS::ADD); //both input tensors are BxNxN
    */

    TensorF* point_cloud_transpose = platformSelector->Transpose(PLATFORMS::GPU_OCL,scheduler,input_BxNxD);
    TensorF* point_cloud_inner =  platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,input_BxNxD,point_cloud_transpose);
    TensorF* point_cloud_inner2 = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,point_cloud_inner,-2.0f,MAT_OPS::MUL_ELEMENTWISE);
    TensorF* point_cloud_inner2p2 = platformSelector->Square(PLATFORMS::GPU_OCL,scheduler,input_BxNxD);
    TensorF* point_cloud_sum = platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,point_cloud_inner2p2,false,false,true);
    TensorF* point_cloud_sum_tiled =  platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum,2,N); //result is BxNxK for k=N
    TensorF* point_cloud_sum_transpose_tiled =  platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum,1,N); //result is BxkxN for k=N
    TensorF* rsltTmpTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,point_cloud_sum_tiled,point_cloud_sum_transpose_tiled, MAT_OPS::ADD); //both input tensors are BxNxN
    TensorF* rsltTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,rsltTmpTn,point_cloud_inner2, MAT_OPS::ADD); //both input tensors are BxNxN

    delete(point_cloud_transpose);
    delete(point_cloud_inner);
    delete(point_cloud_inner2);
    delete(point_cloud_inner2p2);
    delete(point_cloud_sum);
    //delete(point_cloud_sum_transpose);
    delete(point_cloud_sum_tiled);
    delete(point_cloud_sum_transpose_tiled);
    delete(rsltTmpTn);

    return rsltTn;
}

TensorF* ModelArchTop02::TransformNet(WorkScheduler scheduler, TensorF* edgeFeatures){

    TensorF* net;
    {
        TensorF* net1 = platformSelector->Conv2D(
                PLATFORMS::GPU_OCL,
                scheduler,
                edgeFeatures,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.weights.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.biases.npy")
                        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A01_tnet_conv.npy",net1);

        TensorF* net2 = Batchnorm_Forward(
                scheduler,net1,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.bn.gamma.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.bn.beta.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A02_tnet_bn.npy",net2);


        TensorF *net3 = platformSelector->ReLU(
                PLATFORMS::GPU_OCL,
                scheduler,
                net2);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A03_tnet_relu.npy",net3);

        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------
    {
        TensorF* net1 = platformSelector->Conv2D(
                PLATFORMS::GPU_OCL,
                scheduler,
                net,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.weights.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.biases.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A04_tnet_conv.npy",net1);

        TensorF* net2 = Batchnorm_Forward(
                scheduler,net1,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.bn.gamma.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.bn.beta.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A05_tnet_bn.npy",net2);


        TensorF *net3 = platformSelector->ReLU(
                PLATFORMS::GPU_OCL,
                scheduler,
                net2);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A06_tnet_relu.npy",net3);
        delete(net);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------
    {
        TensorF *net1 = platformSelector->ReduceMax(
                PLATFORMS::GPU_OCL,
                scheduler,net,2);
        net1->ExpandDims(2);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A07_tnet_pool.npy",net1);
        delete(net);
        net = net1;
    }

    //----------------------------------------------------------------------------
    {
        TensorF* net1 = platformSelector->Conv2D(
                PLATFORMS::GPU_OCL,
                scheduler,
                net,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.weights.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.biases.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A08_tnet_conv.npy",net1);

        TensorF* net2 = Batchnorm_Forward(
                scheduler,net1,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.bn.gamma.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.bn.beta.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A09_tnet_bn.npy",net2);


        TensorF *net3 = platformSelector->ReLU(
                PLATFORMS::GPU_OCL,
                scheduler,
                net2);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A10_tnet_relu.npy",net3);
        delete(net);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------
    {
        TensorF *net1 = platformSelector->ReduceMax(
                PLATFORMS::GPU_OCL,
                scheduler,net,1);
        net1->SqueezeDims();
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A11_tnet_pool.npy",net1);
        delete(net);
        net = net1;
    }

    //----------------------------------------------------------------------------
    //FC
    // net is Bx1024
    {
        TensorF* net1 = FullyConnected_Forward(
                scheduler,
                net,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.weights.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.biases.npy")
                );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A12_tnet_fc.npy",net1);

        TensorF* net2 = Batchnorm_Forward(
                scheduler,net1,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.bn.gamma.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.bn.beta.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A13_tnet_bn.npy",net2);

        TensorF *net3 = platformSelector->ReLU(
                PLATFORMS::GPU_OCL,
                scheduler,
                net2);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A13_tnet_relu.npy",net3);
        delete(net);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------
    //FC
    // net is Bx1024
    {
        TensorF* net1 = FullyConnected_Forward(
                scheduler,
                net,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.weights.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.biases.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A14_tnet_fc.npy",net1);

        TensorF* net2 = Batchnorm_Forward(
                scheduler,net1,
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.bn.gamma.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.bn.beta.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                platformSelector->weightsLoader->AccessWeights(
                        PLATFORMS::GPU_OCL,"transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A15_tnet_bn.npy",net2);

        TensorF *net3 = platformSelector->ReLU(
                PLATFORMS::GPU_OCL,
                scheduler,
                net2);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A16_tnet_relu.npy",net3);
        delete(net);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------
    {
        TensorF* weights = platformSelector->weightsLoader->AccessWeights(
                PLATFORMS::GPU_OCL,"transform_net1.transform_XYZ.weights.npy");
        TensorF* _biases = platformSelector->weightsLoader->AccessWeights(
                PLATFORMS::GPU_OCL,"transform_net1.transform_XYZ.biases.npy");

        float eyeData[] = {1,0,0,
                           0,1,0,
                           0,0,1};
        TensorF* eye = new TensorF({9},eyeData);

        TensorF* biases = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,_biases,eye, MAT_OPS::ADD);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A17_biass_added.npy",biases);

        TensorF* transformTn = platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,net,weights);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A18_transform_batch.npy",transformTn);

        TensorF* transformFinalTn = platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,transformTn,biases, MAT_OPS::ADD);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"A19_transform_batch_bias.npy",transformFinalTn);

        delete(biases);
        delete(transformTn);

        return platformSelector->CrossThePlatform(transformFinalTn, PLATFORMS::CPU); // Force CPU
    }
}

/// Returns per-class score tensor of shape=40 and rank=1
/// \param scheduler
/// \return
TensorF* ModelArchTop02::Execute(WorkScheduler scheduler) {
    TensorF* net;
    TensorF* net_BxNx3;

    TensorF* endpoint_0;TensorF* endpoint_1;TensorF* endpoint_2;TensorF* endpoint_3;

    bool *correct = new bool[B];
    float accu =0;

    //----------------------------------------------------------------------------------------
    SPDLOG_LOGGER_INFO(logger,"Starting Process...");
    SPDLOG_LOGGER_INFO(logger,"Batch Size: {}", B);
    SPDLOG_LOGGER_INFO(logger,"Point Count: {}", N);

    //----------------------------------------------------------------------------------------
    // TransferNet(net_BxNx3 is this layer's input)
    {
        net_BxNx3 = input_pcl_BxNxD;

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B00_input_pcl_BxNxD.npy",input_pcl_BxNxD);

        TensorF *adj_matrix = PairwiseDistance(scheduler, net_BxNx3);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B01_tnet_adj_matrix.npy",adj_matrix);

        TensorI *nn_idx = platformSelector->TopK(PLATFORMS::GPU_OCL,scheduler,adj_matrix,2,K);


        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B02_tnet_nn_idx.npy",nn_idx);


        TensorF *edge_features = GetEdgeFeatures(scheduler,net_BxNx3, nn_idx);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B03_tnet_edgef.npy",edge_features);

        TensorF *transform = TransformNet(scheduler,edge_features);
        transform->Reshape({B,3,3});
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B04_tnet_3x3.npy",transform);

        net = platformSelector->MatMul(PLATFORMS::GPU_OCL, scheduler, net_BxNx3, transform);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"C01_pcl.npy",net);

        delete(adj_matrix);
        delete(nn_idx);
        delete(edge_features);
        delete(transform);
    }

    //----------------------------------------------------------------------------------------
    // DGCNN Layer #0
    SPDLOG_LOGGER_INFO(logger,"DGCCN0 Started...");
    {
        TensorF *adj_matrix = PairwiseDistance(scheduler,net);
        TensorI *nn_idx = platformSelector->TopK(PLATFORMS::GPU_OCL,scheduler,adj_matrix,2,K);
        TensorF* edge_features = GetEdgeFeatures(scheduler,net,nn_idx);
        TensorF* net1 = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,
                                                 edge_features,
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn1.weights.npy"),
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn1.biases.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"C02_dg1_conv.npy",net1);

        TensorF* net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn1.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn1.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"C03_dg1_bn.npy",net2);

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);

        TensorF* net4 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,net3,2);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B05_dg1_pool.npy",net4);

        net = net4;
        endpoint_0 = net;

        delete(adj_matrix);
        delete(nn_idx);
        delete(edge_features);
        delete(net1);
        delete(net2);
        delete(net3);
    }

    //----------------------------------------------------------------------------------------
    // DGCNN Layer #1
    SPDLOG_LOGGER_INFO(logger,"DGCCN1 Started...");
    {
        TensorF *adj_matrix = PairwiseDistance(scheduler,net);
        TensorI *nn_idx = platformSelector->TopK(PLATFORMS::GPU_OCL,scheduler,adj_matrix,2,K);
        TensorF* edge_features = GetEdgeFeatures(scheduler,net,nn_idx);
        TensorF* net1 = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,
                                                 edge_features,
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn2.weights.npy"),
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn2.biases.npy")
        );

        TensorF* net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn2.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn2.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);

        TensorF* net4 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,net3,2);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B06_dg2_pool.npy",net4);

        net = net4;
        endpoint_1 = net;

        delete(adj_matrix);
        delete(nn_idx);
        delete(edge_features);
        delete(net1);
        delete(net2);
        delete(net3);
    }

    //----------------------------------------------------------------------------------------
    // DGCNN Layer #2
    SPDLOG_LOGGER_INFO(logger,"DGCCN2 Started...");
    {
        TensorF *adj_matrix = PairwiseDistance(scheduler,net);
        TensorI *nn_idx = platformSelector->TopK(PLATFORMS::GPU_OCL, scheduler,adj_matrix,2,K);
        TensorF* edge_features = GetEdgeFeatures(scheduler,net,nn_idx);
        TensorF* net1 = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,
                                                 edge_features,
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn3.weights.npy"),
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn3.biases.npy")
        );

        TensorF* net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn3.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn3.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);

        TensorF* net4 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,net3,2);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B07_dg3_pool.npy",net4);

        net = net4;
        endpoint_2 = net;

        delete(adj_matrix);
        delete(nn_idx);
        delete(edge_features);
        delete(net1);
        delete(net2);
        delete(net3);
    }

    //----------------------------------------------------------------------------------------
    // DGCNN Layer #3
    SPDLOG_LOGGER_INFO(logger,"DGCCN3 Started...");
    {
        TensorF *adj_matrix = PairwiseDistance(scheduler,net);
        TensorI *nn_idx = platformSelector->TopK(PLATFORMS::GPU_OCL,scheduler,adj_matrix,2,K);
        TensorF* edge_features = GetEdgeFeatures(scheduler,net,nn_idx);
        TensorF* net1 = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,
                                                 edge_features,
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn4.weights.npy"),
                                                 platformSelector->weightsLoader->AccessWeights(
                                                         PLATFORMS::GPU_OCL,"dgcnn4.biases.npy")
        );

        TensorF* net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn4.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn4.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);

        TensorF* net4 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,net3,2);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B08_dg4_pool.npy",net4);

        net = net4;
        endpoint_3 = net;

        delete(adj_matrix);
        delete(nn_idx);
        delete(edge_features);
        delete(net1);
        delete(net2);
        delete(net3);
    }

    //----------------------------------------------------------------------------------------
    SPDLOG_LOGGER_INFO(logger,"Agg Layer Started...");
    {
        endpoint_0->ExpandDims(2);
        endpoint_1->ExpandDims(2);
        endpoint_2->ExpandDims(2);
        endpoint_3->ExpandDims(2);
        TensorF *concatA = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler, endpoint_0, endpoint_1, 3);
        delete(endpoint_0);
        delete(endpoint_1);
        TensorF *concatB = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler, concatA, endpoint_2, 3);
        delete(endpoint_2);
        delete(concatA);
        TensorF *concatC = platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler, concatB, endpoint_3, 3);
        delete(endpoint_3);
        delete(concatB);

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B09_agg_concat.npy",concatC);

        // DIM2(K) of concatenated matrix is ONE, NOT 'K'
        TensorF* net1 = platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,
                 concatC,
                 platformSelector->weightsLoader->AccessWeights(
                         PLATFORMS::GPU_OCL,"agg.weights.npy"),
                 platformSelector->weightsLoader->AccessWeights(
                         PLATFORMS::GPU_OCL,"agg.biases.npy")
                 );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B10_agg_conv.npy",net1);

        TensorF* net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"agg.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"agg.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );


        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B11_agg_bn.npy",net2);
        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);

        TensorF* net4 = platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,net3,1);
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B12_agg_pool.npy",net4);

        net = net4;




        delete(concatC);
        delete(net1);
        delete(net2);
        delete(net3);
    }

    //----------------------------------------------------------------------------------------
    //RESHAPING TO (Bx-1)
    {
        net->SqueezeDims();
    }

    //----------------------------------------------------------------------------------------
    //FC1
    //net is of shape Bx1x1x1024
    SPDLOG_LOGGER_INFO(logger,"FC Layer1 Started...");
    {
        TensorF *net1 = FullyConnected_Forward(scheduler,net,
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc1.weights.npy"),
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc1.biases.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B13_fc.npy",net1);

        //net1 is of shape Bx1x1x512
        TensorF *net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc1.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc1.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B14_fc.npy",net2);

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------------------
    //FC2
    //net is of shape Bx1x1x512
    SPDLOG_LOGGER_INFO(logger,"FC Layer2 Started...");
    {
        TensorF *net1 = FullyConnected_Forward(scheduler,net,
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc2.weights.npy"),
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc2.biases.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B15_fc.npy",net1);

        //net1 is of shape Bx1x1x512
        TensorF *net2 = Batchnorm_Forward(scheduler,net1,
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc2.bn.gamma.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc2.bn.beta.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy"),
                                          platformSelector->weightsLoader->AccessWeights(
                                                  PLATFORMS::GPU_OCL,"fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy")
        );

        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B16_fc.npy",net2);

        TensorF* net3 = platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,net2);
        net = net3;

        delete(net1);
        delete(net2);
    }

    //----------------------------------------------------------------------------------------
    //FC3
    //net is of shape Bx1x1x256
    SPDLOG_LOGGER_INFO(logger,"FC Layer3 Started...");
    {
        TensorF *net1 = FullyConnected_Forward(scheduler,net,
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc3.weights.npy"),
                                             platformSelector->weightsLoader->AccessWeights(
                                                     PLATFORMS::GPU_OCL,"fc3.biases.npy")
        );
        platformSelector->DumpMatrix(platformSelector->defaultPlatform,scheduler,"B17_fc.npy",net1);
        net = net1;
    }

    //----------------------------------------------------------------------------------------
    //force output tensor platform to be CPU
    {
#ifdef USE_CUDA
        if(net->getPlatform()==PLATFORMS::GPU_CUDA) return ((CudaTensorF*)net)->TransferToHost();
#endif
#ifdef USE_OCL
        if(net->getPlatform()==PLATFORMS::GPU_OCL) return platformSelector->CrossThePlatform(net,PLATFORMS::CPU);
#endif
    }

    return net;
}

TensorI* ModelArchTop02::GetLabels(){
    return input_labels_B;
}

int ModelArchTop02::GetBatchSize(){
    return B;
}
