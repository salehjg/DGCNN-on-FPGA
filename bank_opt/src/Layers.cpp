#include "Layers.h"
#include "config.h"

Layers::Layers(PLATFORMS defaultPlatform, vector<PLATFORMS> neededPlatforms, bool loadWeights) {

}

TensorI *Layers::CreateDummyTensorI(int bank, string tag) {
    TensorI *tn = new TensorI({1,2,3},bank, tag);
    return tn;
}

TensorF *Layers::CreateDummyTensorF(int bank, string tag) {
    TensorF *tn = new TensorF({1,2,3},bank, tag);
    return tn;
}

TensorF *Layers::UnpadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, unsigned lastDimUnpadded) {
    ChangeBankIfNeeded(inputTn, ConfigTaskPadUnpad::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-padunpad_in)");

    return CreateDummyTensorF(ConfigTaskPadUnpad::BankIndex_outputTn, "padunpad_out");
}

TensorF *Layers::PadLastDim(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, unsigned lastDimPadded) {
    ChangeBankIfNeeded(inputTn, ConfigTaskPadUnpad::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-padunpad_in)");

    return CreateDummyTensorF(ConfigTaskPadUnpad::BankIndex_outputTn, "padunpad_out");
}

TensorF *Layers::Tile(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    ChangeBankIfNeeded(inputTn, ConfigTaskTile::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-tile_in)");

    return CreateDummyTensorF(ConfigTaskTile::BankIndex_outputTn, "tile_out");
}

TensorF *Layers::ReLU(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn) {
    ChangeBankIfNeeded(inputTn, ConfigTaskReluSqrtSquare::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-relusqrtsquare_in)");

    return CreateDummyTensorF(ConfigTaskReluSqrtSquare::BankIndex_outputTn, "relusqrtsquare_out");
}

TensorF * Layers::Conv2D(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, TensorF *weights, TensorF *biases,
               int overrideDim2) {
    TensorF* _weights = PadLastDim(platform,scheduler,weights,123);

    ChangeBankIfNeeded(inputTn, ConfigTaskConv2::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-conv_in)");
    ChangeBankIfNeeded(_weights, ConfigTaskConv2::BankIndex_inputTn);
    objective.push_back("abs("+ _weights->tag +"-conv_w)");
    ChangeBankIfNeeded(biases, ConfigTaskConv2::BankIndex_inputTn);
    objective.push_back("abs("+ biases->tag +"-conv_b)");

    TensorF* _rsltTn = CreateDummyTensorF(ConfigTaskConv2::BankIndex_outputTn,"conv_out");
    TensorF* rsltTn = UnpadLastDim(platform,scheduler,_rsltTn,123);
    return rsltTn;
}

TensorF *
Layers::Gather(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, TensorI *indices, int indices_axis) {
    ChangeBankIfNeeded(inputTn, ConfigTaskGather::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-gather_in1)");
    ChangeBankIfNeeded(indices, ConfigTaskGather::BankIndex_indicesTn);
    objective.push_back("abs("+ indices->tag +"-gather_in2)");

    return CreateDummyTensorF(ConfigTaskGather::BankIndex_outputTn,"gather_out");
}

TensorI *Layers::TopK(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat, int axis, int k) {
    ChangeBankIfNeeded(batchedMat, ConfigTaskTopK::BankIndex_inputTn);
    objective.push_back("abs("+ batchedMat->tag +"-topk_in)");

    return CreateDummyTensorI(ConfigTaskTopK::BankIndex_indicesSplitedTn,"topk_out");
}

TensorF *Layers::ReduceMax(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int reductionDim) {
    ChangeBankIfNeeded(inputTn, ConfigTaskReduce::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-reduce_in)");

    return CreateDummyTensorF(ConfigTaskReduce::BankIndex_outputTn,"reduce_out");
}

TensorF *
Layers::Concat2(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, int concatDim) {
    ChangeBankIfNeeded(inputTn1, ConfigTaskConcat::BankIndex_inputTn1);
    objective.push_back("abs("+ inputTn1->tag +"-concat_in1)");
    ChangeBankIfNeeded(inputTn2, ConfigTaskConcat::BankIndex_inputTn2);
    objective.push_back("abs("+ inputTn2->tag +"-concat_in2)");

    return CreateDummyTensorF(ConfigTaskConcat::BankIndex_outputTn,"concat_out");
}

TensorF *Layers::Sqrt(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn) {
    ChangeBankIfNeeded(inputTn, ConfigTaskReluSqrtSquare::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-relusqrtsquare_in)");

    return CreateDummyTensorF(ConfigTaskReluSqrtSquare::BankIndex_outputTn,"relusqrtsquare_out");
}

TensorF *
Layers::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode) {
    ChangeBankIfNeeded(inputTn1, ConfigTaskMatOps::BankIndex_inputTn1);
    objective.push_back("abs("+ inputTn1->tag +"-matops_in1)");
    ChangeBankIfNeeded(inputTn2, ConfigTaskMatOps::BankIndex_inputTn2);
    objective.push_back("abs("+ inputTn2->tag +"-matops_in2)");

    return CreateDummyTensorF(ConfigTaskMatOps::BankIndex_outputTn,"matops_out");
}

TensorF *
Layers::ReduceSum4D(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, bool over_axis0, bool over_axis1,
                    bool over_axis2, bool over_axis3) {
    ChangeBankIfNeeded(inputTn, ConfigTaskReduce::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-reduce_in)");

    return CreateDummyTensorF(ConfigTaskReduce::BankIndex_outputTn,"reduce_out");
}

TensorF *
Layers::ReduceSum(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, bool over_axis0, bool over_axis1,
                  bool over_axis2) {
    ChangeBankIfNeeded(inputTn, ConfigTaskReduce::BankIndex_inputTn);
    objective.push_back("abs("+ inputTn->tag +"-reduce_in)");

    return CreateDummyTensorF(ConfigTaskReduce::BankIndex_outputTn,"reduce_out");
}

TensorF *Layers::Square(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat) {
    ChangeBankIfNeeded(batchedMat, ConfigTaskReluSqrtSquare::BankIndex_inputTn);
    objective.push_back("abs("+ batchedMat->tag +"-relusqrtsquare_in)");

    return CreateDummyTensorF(ConfigTaskReluSqrtSquare::BankIndex_outputTn,"relusqrtsquare_out");
}

TensorF *Layers::MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat1, TensorF *batchedMat2) {
    ChangeBankIfNeeded(batchedMat1, ConfigTaskMatMul::BankIndex_inputTn1);
    objective.push_back("abs("+ batchedMat1->tag +"-matmul_in1)");
    ChangeBankIfNeeded(batchedMat2, ConfigTaskMatMul::BankIndex_inputTn2);
    objective.push_back("abs("+ batchedMat2->tag +"-matmul_in2)");

    return CreateDummyTensorF(ConfigTaskMatMul::BankIndex_outputTn,"matmul_out");
}

TensorF *Layers::Transpose(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat) {
    ChangeBankIfNeeded(batchedMat, ConfigTaskTranspose::BankIndex_inputTn);
    objective.push_back("abs("+ batchedMat->tag +"-transpose_in)");

    return CreateDummyTensorF(ConfigTaskTranspose::BankIndex_outputTn,"transpose_out");
}

void Layers::ChangeBankIfNeeded(TensorF *tn, int dstBank) {
    if(tn->bank!=dstBank){
        tn->bank=dstBank;
        this->dataMoverLaunches++;
    }
}

void Layers::ChangeBankIfNeeded(TensorI *tn, int dstBank) {
    if(tn->bank!=dstBank){
        tn->bank=dstBank;
        this->dataMoverLaunches++;
    }
}

TensorF *Layers::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode) {
    TensorF* tn = CreateDummyTensorF(ConfigTaskMatOps::BankIndex_inputTn2,"matops_in2");
    return MatOps(platform,scheduler,inputTn1,tn,mode);
}

TensorF *Layers::Variance(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, bool variance_axis0,
                          bool variance_axis1, bool variance_axis2, bool variance_axis3) {

    TensorF* tmpTn = ReduceSum4D(platform,scheduler,inputTn,false,false,false,false);
    TensorF* varianceXi2Tn = ReduceSum4D(platform,scheduler,inputTn,false,false,false,false);

    float coef = 0;
    TensorF* meanTn = MatOps(platform,scheduler,tmpTn,coef,MAT_OPS::DIV_ELEMENTWISE);
    TensorF* tmp2Tn = MatOps(platform,scheduler,varianceXi2Tn,coef,MAT_OPS::DIV_ELEMENTWISE);
    TensorF* tmp3Tn = MatOps(platform,scheduler,meanTn,meanTn,MAT_OPS::MUL_ELEMENTWISE);
    TensorF* rsltTn = MatOps(platform,scheduler,tmp2Tn,tmp3Tn,MAT_OPS::SUB);

    delete(tmpTn);
    delete(tmp2Tn);
    delete(tmp3Tn);
    delete(varianceXi2Tn);
    delete(meanTn);

    return rsltTn;
}

TensorF *Layers::Mean(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, bool mean_axis0, bool mean_axis1,
                      bool mean_axis2, bool mean_axis3) {

    TensorF* reducedTn = ReduceSum4D(platform,scheduler,inputTn, false,false,false,false);
    float coef = 0;
    TensorF* rsltTn = MatOps(platform,scheduler,reducedTn,coef,MAT_OPS::DIV_ELEMENTWISE);

    return rsltTn;
}

