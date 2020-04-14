//
// Created by saleh on 8/22/18.
//

#include "../../inc/PlatformImplementation.h"
#include "../../inc/PlatformSelector.h"
#include "../../inc/cpu_imp/CpuImplementation.h"
#include <iostream>
#include "../../inc/TensorF.h"
#include <cmath>
#include <algorithm>

using namespace std;
///
CpuImplementation::CpuImplementation(){

}

void CpuImplementation::PrintInfo(
        string opName,
        const string &setting1, int val1,
        const string &setting2, int val2,
        const string &setting3, float val3,
        vector<unsigned> shape1,
        vector<unsigned> shape2,
        vector<bool> comb){

    string finalStr ;
    if(!setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1)+ ", " + setting2+ "=" + to_string(val2);
    }else if(!setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1);
    }else if(setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting2+ "=" + to_string(val2);
    }else if(setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " ;
    }

    if(!setting3.empty()){
        finalStr += ", " + setting3 + ": " + to_string(val3);
    }

    if(!shape1.empty()){
        finalStr += ", Shape1=";
        for(unsigned i : shape1){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!shape2.empty()){
        finalStr += ", Shape2=";
        for(unsigned i : shape2){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!comb.empty()){
        finalStr += ", Combination=";
        for(bool i : comb){ finalStr += to_string(i) + "-"; }
        finalStr += ", ";
    }
    finalStr+="\n";
    cout<<finalStr;
}

//float* CpuImplementation::Transpose(float* input,int batchsize, int matrix_rank, int matrixH, int matrixW){
TensorF* CpuImplementation::Transpose(WorkScheduler scheduler, TensorF *batchedMat){
    PrintInfo("Transpose","",0,"",0,"",0,batchedMat->getShape(),{});
    if(batchedMat->getRank()!=3){cout<<"Transpose: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    unsigned B, dim0, dim1;
    B = batchedMat->getShape()[0];
    dim0 = batchedMat->getShape()[1];
    dim1 = batchedMat->getShape()[2];
    float* rslt = new float[B*dim0*dim1];
    unsigned indxS=0;
    unsigned indxD=0;

    for(int b=0;b<B;b++)
    {
        for (int j = 0; j < dim0; j++) {
            for (int i = 0; i < dim1 ; i++) {
                indxS = b * dim0 * dim1 + j * dim1 + i;
                indxD = b * dim0 * dim1 + i * dim0 + j;
                rslt[indxD] = batchedMat->_buff[indxS];
            }
        }
    }
    TensorF *rsltTn = new TensorF({B,dim1,dim0},rslt);
    return rsltTn;
}


/// // rslt = MAT1 * MAT2
/// \param mat1
/// \param mat2
/// \param batchsize
/// \param matrix_rank
/// \param matrixH1
/// \param matrixW1
/// \param matrixH2
/// \param matrixW2
/// \return
/*float* CpuImplementation::MatMul(float* mat1,float* mat2,
                              int batchsize, int matrix_rank,
                              int matrixH1,int matrixW1,
                              int matrixH2,int matrixW2){*/
TensorF* CpuImplementation::MatMul(WorkScheduler scheduler,
    TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());
    int rankDiff = 3 - batchedMat1->getRank();

    for(int i=0;i<rankDiff;i++){
        batchedMat1->ExpandDimZero();
        batchedMat2->ExpandDimZero();
    }

    unsigned matrixH1  = batchedMat1->getShape()[1];
    unsigned matrixW1  = batchedMat1->getShape()[2];
    unsigned matrixH2  = batchedMat2->getShape()[1];
    unsigned matrixW2  = batchedMat2->getShape()[2];
    unsigned batchsize = batchedMat1->getShape()[0];
    unsigned matrix_rank = batchedMat1->getRank();

    if(matrix_rank!=3){cout<<"MatMul: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    if(matrixW1!=matrixH2){cout<<"MatMul: ERROR_BAD_MATRIX_DIMs"<<endl;return nullptr;}
    if(batchedMat1->getShape()[0]!=batchedMat2->getShape()[0]){cout<<"MatMul: ERROR_BAD_MATRIX_DIM0s"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixH1*matrixW2];
    int indxS1=0;
    int indxS2=0;
    int indxD=0;

    for(int b=0;b<batchsize;b++) {
        // for element of output of matrixH1 x matrixW2
        for(int j=0;j<matrixH1;j++){
            for(int i=0;i<matrixW2;i++){
                //mat1: select row j
                //mat2: select col i
                float sum=0;
                for(int mat1_x=0;mat1_x<matrixW1;mat1_x++)
                {
                    indxS1 = b*matrixH1*matrixW1 +
                             j*matrixW1 + mat1_x;
                    /*indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW1 + j;*/
                    indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW2 + i;

                    sum += batchedMat1->_buff[indxS1] * batchedMat2->_buff[indxS2];
                }
                // for element of output of matrixH1 x matrixW2
                indxD = b*matrixH1*matrixW2 +
                        j*matrixW2 + i;
                rslt[indxD] = sum;

            }
        }
    }

    TensorF* rsltTn = new TensorF({batchsize,matrixH1,matrixW2},rslt);
    for(int i=0;i<rankDiff;i++){
        batchedMat1->SqueezeDimZero();
        batchedMat2->SqueezeDimZero();
        rsltTn->SqueezeDimZero();

    }
    return rsltTn;
}

/*
float* CpuImplementation::MatMul(float* mat1,float scalar,
                              int batchsize, int matrix_rank,
                              int matrixH1,int matrixW1){*/
TensorF* CpuImplementation::MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar){
    PrintInfo("MatMul_Scalar","",0,"",0,"scalarVal",scalar,batchedMat->getShape(),{});
    float* rslt = new float[batchedMat->getLength()];
    unsigned limit = batchedMat->getLength();

    for(unsigned b=0;b<limit;b++) {
        rslt[b] = batchedMat->_buff[b] * scalar;
    }
    TensorF* rsltTn = new TensorF(batchedMat->getShape(),rslt);
    return rsltTn;
}

/*float* CpuImplementation::Square(float* mat1,
                              int batchsize, int matrix_rank,
                              int matrixH1,int matrixW1){*/
TensorF* CpuImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});
    if(batchedMat->getRank()!=3){cout<<"Square: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    float* rslt = new float[batchedMat->getLength()];
    unsigned limit = batchedMat->getLength();
    ///TODO: Check index variables to be unsigned!
    for(unsigned b=0;b<limit;b++) {
        rslt[b] = batchedMat->_buff[b] * batchedMat->_buff[b];
    }
    TensorF* rsltTn = new TensorF(batchedMat->getShape(),rslt);
    return rsltTn;
}

/*
float* CpuImplementation::ReduceSum(float* mat1,
                           bool over_axis0,
                           bool over_axis1,
                           bool over_axis2,
                           int dim0,
                           int dim1,
                           int dim2){*/
TensorF* CpuImplementation::ReduceSum(WorkScheduler scheduler,
        TensorF* inputTn,
        bool over_axis0,
        bool over_axis1,
        bool over_axis2){
    PrintInfo("ReduceSum","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2});

    int rankDiff = 3 - inputTn->getRank();
    for(int i=0;i<rankDiff;i++){
        inputTn->ExpandDimZero();
    }

    unsigned indxS=0;
    unsigned indxD=0;
    float* rslt;
    unsigned dim0 = inputTn->getShape()[0],
        dim1 = inputTn->getShape()[1],
        dim2 = inputTn->getShape()[2];

    if(inputTn->getRank() != 3){cout<<"ReduceSum: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}

    if(over_axis0==true &&
       over_axis1==true &&
       over_axis2==true )
    {
        float sum = 0;
        rslt = &sum;
        unsigned limit = inputTn->getLength();

        for(unsigned b=0;b<limit;b++) {
            (*rslt) += inputTn->_buff[b];
        }
        TensorF* rsltTn = new TensorF({1},rslt);
        for(int i=0;i<rankDiff;i++){
            inputTn->SqueezeDimZero();
        }
        return rsltTn;
    }

    if(over_axis0==true &&
       over_axis1==false &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim1];
        float sum=0;
        for(int d1=0; d1<dim1;d1++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d1 * dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim0;dx++)
                {
                    indxS = dx * dim1*dim2 + d1 * dim2 + d2;
                    sum += inputTn->_buff[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        TensorF* rsltTn = new TensorF({dim1,dim2},rslt);
        for(int i=0;i<rankDiff;i++){
            inputTn->SqueezeDimZero();
        }
        return rsltTn;
    }

    if(over_axis0==false &&
       over_axis1==true &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d0 *dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim1;dx++)
                {
                    indxS = d0 * dim1*dim2 + dx * dim2 + d2;
                    sum+=inputTn->_buff[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        TensorF* rsltTn = new TensorF({dim0,dim2},rslt);
        for(int i=0;i<rankDiff;i++){
            inputTn->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }
        return rsltTn;
    }

    if(over_axis0==false &&
       over_axis1==false &&
       over_axis2==true )
    {
        rslt = new float[dim1*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                sum=0;
                indxD = d0 * dim1 + d1;
                //sum over dim of interest
                for(int dx=0;dx<dim2;dx++)
                {
                    indxS = d0 * dim1*dim2 + d1 * dim2 + dx;
                    sum+=inputTn->_buff[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        TensorF* rsltTn = new TensorF({dim0,dim1},rslt);
        for(int i=0;i<rankDiff;i++){
            inputTn->SqueezeDimZero();
            rsltTn->SqueezeDimZero();
        }
        return rsltTn;
    }

    cout<<"ReduceSum:ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
    return nullptr;
}

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is(as a matrix)
/*float* CpuImplementation::ReduceSum4D(float* mat1,
                             bool over_axis0,
                             bool over_axis1,
                             bool over_axis2,
                             bool over_axis3,
                             int dim0,
                             int dim1,
                             int dim2,
                             int dim3){*/
TensorF* CpuImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){

    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
    //cout<<"**LA_SUM4D: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<";\n";
    unsigned indxS=0;
    unsigned indxD=0;
    float* rslt;
    unsigned dim0 = inputTn->getShape()[0],
        dim1 = inputTn->getShape()[1],
        dim2 = inputTn->getShape()[2],
        dim3 = inputTn->getShape()[3];

    if(inputTn->getRank() != 4){cout<<"ReduceSum4D: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}

    if(over_axis0==true &&
       over_axis1==true &&
       over_axis2==true &&
       over_axis3==false )
    {
        rslt = new float[dim3];
        float sum=0;
        for (int d3 = 0; d3 < dim3; d3++)
        {
            sum=0;
            indxD = d3;
            for (int d0 = 0; d0 < dim0; d0++) {
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = 0; d2 < dim2; d2++) {

                        indxS = d0*dim1*dim2*dim3+
                                d1*dim2*dim3+
                                d2*dim3+
                                d3;

                        sum += inputTn->_buff[indxS];
                    }
                }
            }

            rslt[indxD] = sum;
        }
        TensorF* rsltTn = new TensorF({dim3},rslt);
        return rsltTn;
    }

    cout<<"ReduceSum4D:ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
    return nullptr;
}

/*
float* CpuImplementation::Mean(
        float* mat,
        int rank,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3

){*/
TensorF* CpuImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){

    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});
    unsigned dim0 = inputTn->getShape()[0],
        dim1 = inputTn->getShape()[1],
        dim2 = inputTn->getShape()[2],
        dim3 = inputTn->getShape()[3];
    int rank = inputTn->getRank();

    //cout      <<"**LA_Mean: Rank: "<< rank << "  dims: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<
    //            "  overaxes: "<<mean_axis0<<","<<mean_axis1<<","<<mean_axis2<<","<<mean_axis3<<";\n";

    if(rank==4){
        if(!mean_axis3 && mean_axis0 && mean_axis1 && mean_axis2){
            ///TODO: Change method call for LA_Sum4D: ******************DONE
            TensorF* reduced = ReduceSum4D(scheduler,inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            float* sum = reduced->_buff;
            /*
            float* sum =
            LA_Sum4D(mat,
                          mean_axis0,
                          mean_axis1,
                          mean_axis2,
                          mean_axis3,
                          dim0,
                          dim1,
                          dim2,
                          dim3);*/

            float *mean = new float[dim3];

            for(int d3=0;d3<dim3;d3++){
                mean[d3] = (sum[d3])/(float)(dim0*dim1*dim2);
            }
            //delete(reduced);

            TensorF* rsltTn = new TensorF({dim3}, mean);
            return rsltTn;
        }
        cout<<"Mean: ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){ //dim0 is batch, dim1 is fc layer output, ex.: for B=1 --> output=[1,256]
        if(!mean_axis1 && mean_axis0 ){
            ///TODO: CHANGE LA_Sum method call: **********************DONE
            /*float* sum = LA_Sum(mat,false,true,false,1,dim0,dim1);*/ //result is of shape dim1
            TensorF* reduced = ReduceSum(scheduler,inputTn, false, true, false);

            ///TODO: CHANGE LA_MatMul method call:*******************DONE
            //mean = LA_MatMul(sum,(1.0f/dim0),1,3,1,dim1); //B=1, Rank=3, H=0,W=dim1
            TensorF* mean = MatMul(scheduler,reduced,(1.0f/dim0));

            //delete(reduced);

            return mean;
        }
        cout<<"Mean: ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        ///TODO: Change LA_Sum method call: ***********************DONE
        //float* sum = LA_Sum(mat,true,true,true,1,1,dim0);
        TensorF* reduced = ReduceSum(scheduler,inputTn,true,true,true);
        TensorF* mean = MatMul(scheduler,reduced,1.0f/(float)(dim0));
        //float *mean = (float*)malloc(sizeof(float) * 1);
        //*mean = (*sum)/(float)(dim0);
        //free(sum);
        //delete(reduced);

        return mean;
    }
    cout<<"Mean: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

/*
float* CpuImplementation::Variance(
        float* mat,
        int rank,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3

){*/
TensorF* CpuImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){

    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});
    unsigned dim0 = inputTn->getShape()[0],
        dim1 = inputTn->getShape()[1],
        dim2 = inputTn->getShape()[2],
        dim3 = inputTn->getShape()[3];
    int rank = inputTn->getRank();

    //cout      <<"**LA_Variance: Rank: "<< rank << "  dims: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<
    //          "  overaxes: "<<variance_axis0<<","<<variance_axis1<<","<<variance_axis2<<","<<variance_axis3<<";\n";
    if(rank==4){
        if(!variance_axis3 && variance_axis0 && variance_axis1 && variance_axis2) {
            ///TODO: Change LA_Mean method call: *********************DONE
            /*
            float *mean = LA_Mean(mat,
                                  rank,
                                  variance_axis0,
                                  variance_axis1,
                                  variance_axis2,
                                  false,
                                  dim0,
                                  dim1,
                                  dim2,
                                  dim3);*/

            TensorF* mean = Mean(scheduler,
                                 inputTn,
                                 variance_axis0,
                                 variance_axis1,
                                 variance_axis2,
                                 false);

            TensorF* variance = new TensorF({dim3});

            unsigned indxS1,indxS2,indxD;
            for (int d3 = 0; d3 < dim3; d3++) { //over last-dim
                variance->_buff[d3]=0;


                for (int d0 = 0; d0 < dim0; d0++) {
                    for (int d1 = 0; d1 < dim1; d1++) {
                        for (int d2 = 0; d2 < dim2; d2++) {
                            indxS1 = d0*dim1*dim2*dim3+
                                     d1*dim2*dim3+
                                     d2*dim3+
                                     d3;

                            float delta = (inputTn->_buff[indxS1] - mean->_buff[d3]);
                            variance->_buff[d3] += delta*delta;
                        }
                    }
                }
            }

            ///TODO: CHANGE LA_MatMul method call: *****************DONE
            TensorF* variance_final = MatMul(scheduler,variance,(float)(1.0f/(dim0*dim1*dim2)));
            
            return variance_final;
        }
        cout<<"Variance: ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){
        if(!variance_axis1 && variance_axis0 ) {
            ///TODO: Change LA_Mean method call: ******************DONE
            /*
            float *mean = LA_Mean(mat,
                                  2,
                                  true,
                                  false,
                                  false,
                                  false,
                                  dim0,
                                  dim1,
                                  0,
                                  0);*/
            TensorF* mean = Mean(scheduler,inputTn,true,false,false,false);
            TensorF* variance = new TensorF({dim1});

            unsigned indxS1,indxS2,indxD;
            for (int d1 = 0; d1 < dim1; d1++) { //over last-dim
                variance->_buff[d1]=0;

                for (int d0 = 0; d0 < dim0; d0++) {
                    indxS1 = d0*dim1 + d1;

                    float delta = (inputTn->_buff[indxS1]-mean->_buff[d1]);
                    variance->_buff[d1] += delta*delta;
                }
            }
            ///TODO: Change LA_Mean method call: **********************DONE
            //float* variance_final = LA_MatMul(variance,(float)(1.0f/dim0),1,3,1,dim1);
            TensorF* variance_final = MatMul(scheduler,variance,(float)(1.0f/dim0));

            //delete(variance);
            //delete(mean);

            return variance_final;
        }
        cout<<"Variance: ERROR_UNIMPLEMENTED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        ///TODO: Change LA_Mean method call: ********************DONE
        /*float *mean = LA_Mean(mat,
                              rank,
                              true,
                              true,
                              true,
                              true,
                              dim0,
                              1,
                              1,
                              1);*/
        TensorF* mean = Mean(scheduler, inputTn, true, true, true, true);
        TensorF* variance = new TensorF({1});
        //float *variance = (float*)malloc(sizeof(float) * 1);

        for (int d0 = 0; d0 < dim0; d0++) {
            float delta = (inputTn->_buff[d0] - mean->_buff[0]);
            variance->_buff[0] += delta*delta;
        }

        //*variance = *variance * (float)(dim0);
        TensorF* variance_final = MatMul(scheduler,variance,1.0f/(float)(dim0));

        //delete(mean);
        //delete(variance);

        return variance_final;

    }
    cout<<"Variance: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}
/*
float* CpuImplementation::MatAdd(
        float* mat1,
        float* mat2,
        int    rank,
        int    dim0,
        int    dim1,
        int    dim2){*/
TensorF* CpuImplementation::MatAdd(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("MatAdd","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    if(inputTn1->getShape() != inputTn2->getShape()){cout<<"MatAdd: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    unsigned limit = inputTn1->getLength();
    float* rslt=new float[limit];

    for(unsigned d0=0;d0<limit;d0++){
        rslt[d0] = inputTn1->_buff[d0] + inputTn2->_buff[d0];
    }

    TensorF* rsltTn = new TensorF(inputTn1->getShape(),rslt);
    return rsltTn;
}

/*
float* CpuImplementation::MatSub(
        float* mat1,
        float* mat2,
        int    rank,
        int    dim0,
        int    dim1,
        int    dim2,
        int    dim3
){*/

TensorF* CpuImplementation::MatSub(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2){
    PrintInfo("MatSub","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    unsigned limit;
    if(inputTn1->getShape() != inputTn2->getShape()){cout<<"MatSub: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}
    limit = inputTn1->getLength();

    float *rslt = new float[limit];

    for (unsigned d0 = 0; d0 < limit; d0++) {
        rslt[d0] = inputTn1->_buff[d0] - inputTn2->_buff[d0];
    }
    TensorF* rsltTn = new TensorF(inputTn1->getShape(),rslt);
    return rsltTn;


    cout<<"MatSub: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

TensorF* CpuImplementation::MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    PrintInfo("MatAddTiled","",0,"",0,"",0,inputTn1->getShape(),inputSmallTn2->getShape(),{});
    if(!(inputTn1->getRank()==4 || inputTn1->getRank()==2)){cout<<"MatAddTiled: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}
    if(inputSmallTn2->getRank()!=1){cout<<"MatAddTiled: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    unsigned indxS1,indxS2,indxD;
    TensorF* rsltTn = new TensorF(inputTn1->getShape());

    unsigned dim0, dim1, dim2, dim3;

    if(inputTn1->getRank()==4 && inputSmallTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        dim2 = inputTn1->getShape()[2];
        dim3 = inputTn1->getShape()[3];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                for(int d2=0;d2<dim2;d2++) {
                    for(int d3=0;d3<dim3;d3++) {
                        indxS1 = d0*dim1*dim2*dim3+
                                 d1*dim2*dim3+
                                 d2*dim3+
                                 d3;
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] + inputSmallTn2->_buff[d3];
                    }
                }
            }
        }
        return rsltTn;
    }

    if(inputTn1->getRank()==2 && inputSmallTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                indxS1 = d0*dim1 + d1;
                rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] + inputSmallTn2->_buff[d1];
            }
        }
        return rsltTn;
    }

    cout<<"MatAddTiled: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

TensorF* CpuImplementation::MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn1->getShape(),inputSmallTn2->getShape(),{});
    if(!(inputTn1->getRank()==4 || inputTn1->getRank()==2)){cout<<"MatAddTiled: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}
    if(inputSmallTn2->getRank()!=1){cout<<"MatAddTiled: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    unsigned indxS1;
    unsigned dim0, dim1, dim2, dim3;

    if(inputTn1->getRank()==4 && inputSmallTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        dim2 = inputTn1->getShape()[2];
        dim3 = inputTn1->getShape()[3];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                for(int d2=0;d2<dim2;d2++) {
                    for(int d3=0;d3<dim3;d3++) {
                        indxS1 = d0*dim1*dim2*dim3+
                                 d1*dim2*dim3+
                                 d2*dim3+
                                 d3;
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] - inputSmallTn2->_buff[d3];
                    }
                }
            }
        }
        return rsltTn;
    }

    if(inputTn1->getRank()==2 && inputSmallTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                indxS1 = d0*dim1 + d1;
                rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] - inputSmallTn2->_buff[d1];
            }
        }
        return rsltTn;
    }

    cout<<"MatAddTiled: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

TensorF* CpuImplementation::MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    PrintInfo("MatAddTiled","",0,"",0,"scalar",scalar,inputTn1->getShape(),{},{});
    TensorF* rsltTn = new TensorF(inputTn1->getShape());
    unsigned len = inputTn1->getLength();
    for(unsigned d=0;d<len;d++) {
        rsltTn->_buff[d] = inputTn1->_buff[d] + scalar;
    }
    return rsltTn;
}

TensorF* CpuImplementation::MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    PrintInfo("MatSubTiled","",0,"",0,"scalar",scalar,inputTn1->getShape(),{},{});
    TensorF* rsltTn = new TensorF(inputTn1->getShape());
    unsigned len = inputTn1->getLength();
    for(unsigned d=0;d<len;d++) {
        rsltTn->_buff[d] = inputTn1->_buff[d] - scalar;
    }
    return rsltTn;
}

TensorF* CpuImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode) {
    PrintInfo("MatOps",
            "mode",(mode==MAT_OPS::ADD ? 0 :
                    mode==MAT_OPS::SUB ? 1 :
                    mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                    3),
            "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    int rankDiff;

    if(!(inputTn1->getRank()<=4 && inputTn1->getRank()>=1 && inputTn2->getRank()<=4 && inputTn2->getRank()>=1 )){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E1"<<endl;
        return nullptr;
    }

    if(inputTn1->getRank() < inputTn2->getRank()){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E2"<<endl;
    return nullptr;
    }

    //forcing inputTn1 to be of rank 4. (always)
    rankDiff = 4- inputTn1->getRank();
    while(inputTn1->getRank()<4){
        inputTn1->ExpandDimZero();
    }

    unsigned indxS1;
    unsigned indxS2;
    unsigned dim0, dim1, dim2, dim3;
    unsigned dim0B, dim1B, dim2B, dim3B;
    int dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;

    TensorF* rsltTn = new TensorF( inputTn1->getShape() );

    dim0 = inputTn1->getShape()[0];
    dim1 = inputTn1->getShape()[1];
    dim2 = inputTn1->getShape()[2];
    dim3 = inputTn1->getShape()[3];

    if(inputTn2->getRank()==4){
        dim0B=inputTn2->getShape()[0];
        dim1B=inputTn2->getShape()[1];
        dim2B=inputTn2->getShape()[2];
        dim3B=inputTn2->getShape()[3];
    }
    if(inputTn2->getRank()==3){
        dim0B=0                     ;
        dim1B=inputTn2->getShape()[0];
        dim2B=inputTn2->getShape()[1];
        dim3B=inputTn2->getShape()[2];
    }
    if(inputTn2->getRank()==2){
        dim0B=0;
        dim1B=0;
        dim2B=inputTn2->getShape()[0];
        dim3B=inputTn2->getShape()[1];
    }
    if(inputTn2->getRank()==1 && inputTn2->getShape()[0]!=1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=inputTn2->getShape()[0];
    }
    if(inputTn2->getShape()[0]==1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=1; //and rank should be 1 which already is
    }


    int tmp =15>>(4-inputTn2->getRank());
    dim0B_IsNotZero = (tmp >> 3) & 1;
    dim1B_IsNotZero = (tmp >> 2) & 1;
    dim2B_IsNotZero = (tmp >> 1) & 1;
    dim3B_IsNotZero = (tmp >> 0) & 1;

    if(inputTn2->getRank()==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dim3B_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }

    for(int d0=0;d0<dim0;d0++){
        for(int d1=0;d1<dim1;d1++) {
            for(int d2=0;d2<dim2;d2++) {
                for(int d3=0;d3<dim3;d3++) {
                    indxS1 = d0*dim1*dim2*dim3+
                             d1*dim2*dim3+
                             d2*dim3+
                             d3;
                    indxS2 = d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
                             d1 * dim2B * dim3B * dim1B_IsNotZero +
                             d2 * dim3B * dim2B_IsNotZero +
                             d3 * dim3B_IsNotZero;

                    if(mode==MAT_OPS::ADD)                      //Add
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] + inputTn2->_buff[indxS2];
                    else if(mode==MAT_OPS::SUB)                 //Sub
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] - inputTn2->_buff[indxS2];
                    else if(mode==MAT_OPS::MUL_ELEMENTWISE)     //Mul (element wise)
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] * inputTn2->_buff[indxS2];
                    else if(mode==MAT_OPS::DIV_ELEMENTWISE)     //Div (element wise)
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] / inputTn2->_buff[indxS2];
                }
            }
        }
    }

    for(int i =0;i<rankDiff;i++){
        inputTn1->SqueezeDimZero();
        rsltTn->SqueezeDimZero();
    }

    return rsltTn;
}

TensorF* CpuImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode) {
    PrintInfo("MatOps-Scalar",
            "mode",(mode==MAT_OPS::ADD ? 0 :
                    mode==MAT_OPS::SUB ? 1 :
                    mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                    3),
            "",0,"",0,inputTn1->getShape(),{},{});
    float* val = new float[1]; val[0] = scalar;
    TensorF* tmpTn = new TensorF({1},val);
    return MatOps(scheduler,inputTn1,tmpTn,mode);
}

TensorF* CpuImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn->getShape(),{},{});
    TensorF* rsltTn = new TensorF(inputTn->getShape());
    unsigned len = inputTn->getLength();
    for(unsigned d=0;d<len;d++) {
        rsltTn->_buff[d] = sqrt(inputTn->_buff[d]);
    }
    return rsltTn;
}

TensorF* CpuImplementation::Multiply(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("Multiply","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    TensorF* rsltTn = new TensorF(inputTn1->getShape());
    unsigned len = inputTn1->getLength();
    for(unsigned d=0;d<len;d++) {
        rsltTn->_buff[d] = (inputTn1->_buff[d]) * (inputTn2->_buff[d]);
    }
    return rsltTn;
}

TensorF* CpuImplementation::Divide(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("Divide","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    TensorF* rsltTn = new TensorF(inputTn1->getShape());
    unsigned len = inputTn1->getLength();
    for(unsigned d=0;d<len;d++) {
        rsltTn->_buff[d] = (inputTn1->_buff[d]) / (inputTn2->_buff[d]);
    }
    return rsltTn;
}

TensorF* CpuImplementation::MultiplyTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("MultiplyTiled","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    if(!(inputTn1->getRank()==4 || inputTn1->getRank()==2)){cout<<"MultiplyTiled: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}
    if(inputTn2->getRank()!=1){cout<<"MultiplyTiled: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    unsigned indxS1;
    int dim0, dim1, dim2, dim3;

    TensorF* rsltTn = new TensorF(inputTn1->getShape());



    if(inputTn1->getRank()==4 && inputTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        dim2 = inputTn1->getShape()[2];
        dim3 = inputTn1->getShape()[3];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                for(int d2=0;d2<dim2;d2++) {
                    for(int d3=0;d3<dim3;d3++) {
                        indxS1 = d0*dim1*dim2*dim3+
                                 d1*dim2*dim3+
                                 d2*dim3+
                                 d3;
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] * inputTn2->_buff[d3];
                    }
                }
            }
        }
        return rsltTn;
    }

    if(inputTn1->getRank()==2 && inputTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                indxS1 = d0*dim1 + d1;
                rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] * inputTn2->_buff[d1];
            }
        }
        return rsltTn;
    }

    cout<<"MultiplyTiled: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

TensorF* CpuImplementation::DivideTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("DivideTiled","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    if(!(inputTn1->getRank()==4 || inputTn1->getRank()==2)){cout<<"DivideTiled: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}
    if(inputTn2->getRank()!=1){cout<<"DivideTiled: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    unsigned indxS1;
    unsigned dim0, dim1, dim2, dim3;

    if(inputTn1->getRank()==4 && inputTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        dim2 = inputTn1->getShape()[2];
        dim3 = inputTn1->getShape()[3];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                for(int d2=0;d2<dim2;d2++) {
                    for(int d3=0;d3<dim3;d3++) {
                        indxS1 = d0*dim1*dim2*dim3+
                                 d1*dim2*dim3+
                                 d2*dim3+
                                 d3;
                        rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] / inputTn2->_buff[d3];
                    }
                }
            }
        }
        return rsltTn;
    }

    if(inputTn1->getRank()==2 && inputTn2->getRank()==1){
        TensorF* rsltTn = new TensorF(inputTn1->getShape());
        dim0 = inputTn1->getShape()[0];
        dim1 = inputTn1->getShape()[1];
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++) {
                indxS1 = d0*dim1 + d1;
                rsltTn->_buff[indxS1] = inputTn1->_buff[indxS1] / inputTn2->_buff[d1];
            }
        }
        return rsltTn;
    }

    cout<<"DivideTiled: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

///concat 2 matrices
/// [matA, matB]
/*
float* CpuImplementation::Concat2(
        float* matA,
        float* matB,
        int rank,
        int concat_dim,
        int dimA0,
        int dimA1,
        int dimA2,
        int dimA3,
        int dimB0,
        int dimB1,
        int dimB2,
        int dimB3
){*/
TensorF* CpuImplementation::Concat2(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2,
        int concatDim){
    PrintInfo("Concat2","concatDim",concatDim,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

    //cout      <<"**Concat2: Rank: "<< rank << "  concatDim: "<<concatDim<<
    //          "  dimA: "<<dimA0<<","<<dimA1<<","<<dimA2<<","<<dimA3<<
    //          "  dimB: "<<dimB0<<","<<dimB1<<","<<dimB2<<","<<dimB3<<";\n";

    if(inputTn1->getRank() != inputTn2->getRank()){cout<<"Concat2: ERROR_BAD_TENSOR_RANK"<<endl;return nullptr;}

    int rank  = inputTn1->getRank();
    unsigned dimA0 = inputTn1->getShape()[0];
    unsigned dimA1 = inputTn1->getShape()[1];
    unsigned dimA2 = inputTn1->getShape()[2];
    unsigned dimA3 = inputTn1->getShape()[3];
    unsigned dimB0 = inputTn2->getShape()[0];
    unsigned dimB1 = inputTn2->getShape()[1];
    unsigned dimB2 = inputTn2->getShape()[2];
    unsigned dimB3 = inputTn2->getShape()[3];

    if(rank==4){
        unsigned  dimR0=0,dimR1=0,dimR2=0,dimR3=0;
        int mat2_offset_dim0=0;
        int mat2_offset_dim1=0;
        int mat2_offset_dim2=0;
        int mat2_offset_dim3=0;

        if(concatDim==0){
            dimR0 = dimA0 + dimB0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim0=dimA0;
        }
        if(concatDim==1){
            dimR0 = dimA0;
            dimR1 = dimA1 + dimB1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim1=dimA1;
        }
        if(concatDim==2){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2 + dimB2;
            dimR3 = dimA3;
            mat2_offset_dim2=dimA2;
        }
        if(concatDim==3){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3 + dimB3;
            mat2_offset_dim3=dimA3;
        }

        float* rslt = new float[dimR0*dimR1*dimR2*dimR3];
        int indxS1,indxS2,indxD;

        for(int d0=0;d0<dimA0;d0++){
            for(int d1=0;d1<dimA1;d1++){
                for(int d2=0;d2<dimA2;d2++){
                    for(int d3=0;d3<dimA3;d3++){
                        indxS1 = d0*dimA1*dimA2*dimA3 +
                                 d1*dimA2*dimA3+
                                 d2*dimA3+
                                 d3;
                        indxD = (d0)*dimR1*dimR2*dimR3 +
                                (d1)*dimR2*dimR3+
                                (d2)*dimR3+
                                (d3);
                        rslt[indxD] = inputTn1->_buff[indxS1];
                    }
                }
            }
        }

        for(int d0=0;d0<dimB0;d0++){
            for(int d1=0;d1<dimB1;d1++){
                for(int d2=0;d2<dimB2;d2++){
                    for(int d3=0;d3<dimB3;d3++){
                        indxS2 = d0*dimB1*dimB2*dimB3 +
                                 d1*dimB2*dimB3+
                                 d2*dimB3+
                                 d3;
                        indxD  = (d0+mat2_offset_dim0)*dimR1*dimR2*dimR3 +
                                 (d1+mat2_offset_dim1)*dimR2*dimR3+
                                 (d2+mat2_offset_dim2)*dimR3+
                                 (d3+mat2_offset_dim3);
                        rslt[indxD] = inputTn2->_buff[indxS2];
                    }
                }
            }
        }

        TensorF* rsltTn = new TensorF({dimR0,dimR1,dimR2,dimR3},rslt);
        return rsltTn;
    }

    cout<<"Concat2: ERROR_UNIMPLEMENTED_TENSOR_RANK"<<endl;
    return nullptr;
}

/*
float* CpuImplementation::ReduceMax(
        float* input,
        int axis,
        int rank,
        int dim0,
        int dim1,
        int dim2,
        int dim3){*/

TensorF* CpuImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});
    
    //cout<< "ReduceMax: "<< "Rank: " << rank << ", Axis: "<< reductionDim <<", Dim: "
    // << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << endl;

    if(inputTn->getRank()==4){
        unsigned
            dim0 = inputTn->getShape()[0],
            dim1 = inputTn->getShape()[1],
            dim2 = inputTn->getShape()[2],
            dim3 = inputTn->getShape()[3];
        
        if(reductionDim==3){
            float* rslt= new float[dim0*dim1*dim2];
            unsigned indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= -numeric_limits<float>::infinity();

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d2=0;d2<dim2;d2++){
                        indxD = d0*dim1*dim2+
                                d1*dim2+
                                d2;
                        max = max_cte;
                        for(int d3=0;d3<dim3;d3++){
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<inputTn->_buff[indxS]){
                                max = inputTn->_buff[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            TensorF* rsltTn = new TensorF({dim0,dim1,dim2},rslt);
            return rsltTn;
        }




        if(reductionDim==2){
            float* rslt= new float[dim0*dim1*dim3];
            unsigned indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim1*dim3+
                                d1*dim3+
                                d3;
                        max = max_cte;

                        for(int d2=0;d2<dim2;d2++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<inputTn->_buff[indxS]){
                                max = inputTn->_buff[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            TensorF* rsltTn = new TensorF({dim0,dim1,dim3},rslt);
            return rsltTn;
        }

        if(reductionDim==1){
            float* rslt= new float[dim0*dim2*dim3];
            unsigned indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim2*dim3+
                                d2*dim3+
                                d3;
                        max = max_cte;


                        for(int d1=0;d1<dim1;d1++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<inputTn->_buff[indxS]){
                                max = inputTn->_buff[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            TensorF* rsltTn = new TensorF({dim0,dim2,dim3},rslt);
            return rsltTn;
        }
        cout<<"ReduceMax: ERROR_UNIMPLEMENTED_REDUCTION_AXIS"<<endl;
        return nullptr;
    }

    cout<<"ReduceMax: ERROR_UNIMPLEMENTED_MATRIX_RANK"<<endl;
    return nullptr;
}

TensorI* CpuImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    
    ///TODO: VALIDATE THIS KERNEL!!!

    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});
    if(batchedMat->getRank() != 3){cout<<"TopK: ERROR_UNIMPLEMENTED_TENSOR_RANK"<<endl;return nullptr;}
    if(axis != 2){cout<<"TopK: ERROR_UNIMPLEMENTED_AXIS"<<endl;return nullptr;}
    if(k < 1 || k > batchedMat->getShape()[2]){cout<<"TopK: ERROR_BAD_K"<<endl;return nullptr;}
    if(batchedMat->getShape()[1] != batchedMat->getShape()[2]){cout<<"TopK: ERROR_BAD_TENSOR_SHAPE"<<endl;return nullptr;}

    //batchedMat is considered as BxNxN
    //we will use std::sort in ascending order.
    unsigned indxS=0;
    unsigned B = batchedMat->getShape()[0], N = batchedMat->getShape()[1], K = (unsigned)k;

    TensorI* rslt = new TensorI({B,N,K});

    float tmp_array[N];
    unsigned indices[N];

    for(unsigned b=0;b<B*N;b++){
        for(unsigned i = 0 ;i<N;i++){
            indices[i]=i;
        }
        indxS = b*N + 0;
        std::copy(batchedMat->_buff +indxS, batchedMat->_buff+indxS+N, tmp_array);
        std::sort(  indices,
                    indices+N,
                    [&](int i1, int i2) { return tmp_array[i1] < tmp_array[i2]; } );

        std::copy(indices, indices+K, rslt->_buff+(b*K));
    }

    return rslt;
}

TensorF* CpuImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});
    if(inputTn->getRank() != 3){cout<<"Gather: ERROR_UNIMPLEMENTED_TENSOR_RANK"<<endl;return nullptr;}
    if(indices->getRank() != 3){cout<<"Gather: ERROR_UNIMPLEMENTED_INDICES_RANK"<<endl;return nullptr;}
    if(indices_axis != 1){cout<<"Gather: ERROR_UNIMPLEMENTED_INDICES_AXIS"<<endl;return nullptr;}
    ///TODO: Check 2 tensor shape compatibility!
    //inputTn       is considered as BxNxD
    //indices       is considered as BxNxK
    //indices_axis  is considered to be 1 (the dimension that is equal to 'N')


    //Gather knn's indices from input array.
    unsigned indxS1, indxS2, indxD;
    unsigned
        B = inputTn->getShape()[0],
        N = inputTn->getShape()[1],
        K = indices->getShape()[2],
        D = inputTn->getShape()[2];
    TensorF* point_cloud_neighbors = new TensorF({B, N, K, D});
    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            for(int k=0;k<K;k++){
                indxS1 = b*N*K + n*K + k;
                for(int d=0;d<D;d++)
                {
                    indxD = b*N*K*D + n*K*D + k*D + d;
                    indxS2 = b*N*D +
                             indices->_buff[indxS1]*D +
                             d;
                    point_cloud_neighbors->_buff[indxD] = inputTn->_buff[indxS2];
                }
            }
        }
    }

    return point_cloud_neighbors;
}

TensorF* CpuImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});
    unsigned B,N,K,D,OverridedK,ch_out;
    unsigned indxS1,indxS2,indxD;

    B = inputTn->getShape()[0];
    N = inputTn->getShape()[1];
    K = inputTn->getShape()[2];
    D = inputTn->getShape()[3];
    OverridedK = (overrideDim2==-1)? K : overrideDim2;
    ch_out = weights->getShape().back();
    TensorF* rsltTn = new TensorF({B,N,OverridedK,ch_out});

    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            for(int k=0;k<OverridedK;k++){
                indxS1 = b*N*OverridedK*D + n*OverridedK*D + k*D + 0;
                for(int ch=0;ch<ch_out;ch++){
                    float sum=0;
                    for(int d=0;d<D;d++){
                        indxS2 = d*ch_out + ch;
                        sum += inputTn->_buff[indxS1+d] * weights->_buff[indxS2];
                    }
                    indxD=b*N*OverridedK*ch_out+ n*OverridedK*ch_out+ k*ch_out+ ch;
                    rsltTn->_buff[indxD] = sum + biases->_buff[ch];
                }
            }
        }
    }
    return rsltTn;
}

TensorF* CpuImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});
    unsigned dim = inputTn->getLength();
    TensorF* tmp = new TensorF(inputTn->getShape());

    for(unsigned i=0;i<dim;i++){
        tmp->_buff[i] = (inputTn->_buff[i]>0)?inputTn->_buff[i]:0;
    }
    return tmp;
}

TensorF* CpuImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNxD   ----> BxNxKxD        2               3
    // BxN     ----> BxNxK          2               2
    // BxN     ----> BxKxN          1               2

    PrintInfo("Tile","tileAxis",tileAxis,"tileCount",tileCount,"",0,inputTn->getShape(),{},{});
    unsigned indxS1,indxD;
    if(inputTn->getRank()==3 && tileAxis==2) {
        unsigned B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[1];
        D = inputTn->getShape()[2];
        K = (unsigned)tileCount;

        //tile ing input of shape BxNxD into BxNxKxD.
        TensorF* rsltTn = new TensorF({B, N, K, D});

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                indxS1 = b * N * D + n * D + 0; //beginning of dim2 of input
                for (int k = 0; k < K; k++) {
                    indxD = b * N * K * D + n * K * D + k * D + 0;
                    std::copy(inputTn->_buff + indxS1,
                              inputTn->_buff + indxS1 + D,
                              rsltTn->_buff + indxD);
                }
            }
        }
        return rsltTn;
    }

    if(inputTn->getRank()==2 && tileAxis==2) { //BxN = BxNx1   ------->  BxNxK  (PAGE 221 of my design notebook)
        unsigned B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[1];
        K = (unsigned)tileCount;

        //tile ing input of shape BxN or BxNx1 into BxNxK.
        TensorF* rsltTn = new TensorF({B, N, K});

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                indxS1 = b*N + n;
                for(int k=0;k<K;k++){
                    indxD = b*N*K + n*K + k;
                    rsltTn->_buff[indxD] = inputTn->_buff[indxS1];
                }
            }
        }
        return rsltTn;
    }

    if(inputTn->getRank()==2 && tileAxis==1) { //BxN = Bx1xN   ------->  BxKxN  (PAGE 221 of my design notebook)
        unsigned B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[1];
        K = (unsigned)tileCount;

        //tile ing input of shape BxN or Bx1xN into BxKxN.
        TensorF* rsltTn = new TensorF({B, K, N});

        for (int b = 0; b < B; b++) {
            for(int k=0;k<K;k++){
                for (int n = 0; n < N; n++) {
                    indxD  = b*K*N + k*N + n;
                    indxS1 = b*1*N + n;
                    rsltTn->_buff[indxD] = inputTn->_buff[indxS1];
                }
            }
        }
        return rsltTn;
    }
}

/*
template<typename T> int CpuImplementation::DumpMatrix(
        string npy_fname,
        int rank,
        T* mat,
        int dim0,
        int dim1,
        int dim2,
        int dim3,
        int dim4,
        string npy_dir){*/
void CpuImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorF* inputTn,
        string npy_dir){
#ifdef DUMP_ENABLED
    vector<unsigned> shape = inputTn->getShape();
    vector<unsigned long> shape_size_t(shape.begin(), shape.end());
    cnpy::npy_save<float>(npy_dir+npy_fname,inputTn->_buff, shape_size_t,"w");
#endif
}

void CpuImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorI* inputTn,
        string npy_dir){
#ifdef DUMP_ENABLED
        vector<unsigned> shape = inputTn->getShape();
        vector<unsigned long> shape_size_t(shape.begin(), shape.end());
        cnpy::npy_save<int>(npy_dir+npy_fname,inputTn->_buff ,shape_size_t,"w");
#endif
}

bool CpuImplementation::CompareTensors(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2) {
    float totalDiff=0;
    float currentDiff=0;
    if(inputTn1->getShape() == inputTn2->getShape()){
        unsigned _len = inputTn1->getLength();
        for(unsigned i =0 ; i<_len;i++){
            currentDiff = inputTn1->_buff[i] - inputTn2->_buff[i];
            cout<< "CurrentDiff "<<i<<" :"<<currentDiff<<endl;
            cout<< "Gold "<<i<<" :"<<inputTn1->_buff[i]<<endl;
            cout<< "UDT  "<<i<<" :"<<inputTn2->_buff[i]<<endl;
            //totalDiff += (currentDiff>=0)?currentDiff:-1*currentDiff;
            totalDiff += currentDiff;
        }
        if(totalDiff!=0)cout<< "totalDiff: "<<totalDiff<<endl;
        if((totalDiff> 0 &&totalDiff > 50.0f) || (totalDiff<= 0 &&totalDiff < -50.0f) ) {
            return false;
        }
        return true;
    }
    return false;
}

bool CpuImplementation::CompareTensorsInteger(WorkScheduler scheduler, TensorI *inputTn1, TensorI *inputTn2) {
    float totalDiff=0;
    int currentDiff=0;
    if(inputTn1->getShape() == inputTn2->getShape()){
        unsigned _len = inputTn1->getLength();
        for(unsigned i =0 ; i<_len;i++){
            currentDiff = inputTn1->_buff[i] - inputTn2->_buff[i];
            //totalDiff += (currentDiff>=0)?currentDiff:-1*currentDiff;
            totalDiff += currentDiff;
        }
        if(totalDiff!=0)cout<< "totalDiff: "<<totalDiff<<endl;
        if((totalDiff!=0) ) {
            return false;
        }
        return true;
    }
    return false;
}

TensorF* CpuImplementation::PadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimPadded){
    PrintInfo("PadLastDim","lastDimPadded",lastDimPadded,"",0,"",0,inputTn->getShape(),{},{});
    if(lastDimPadded<=inputTn->getShape()[inputTn->getRank()-1]){cout<<"PadLastDim: ERROR_BAD_PARAM"<<endl;return nullptr;}
    
    unsigned dim0, dim1, lcm, _gcd, argcnt;
    std::vector<unsigned> shape = inputTn->getShape();
    const unsigned rank = inputTn->getRank();

    if(rank!=1){
        dim0=1;
        for(int i=0; i<rank-1; i++){
            dim0*=shape[i];
        }
        dim1=shape[rank-1];
    }else{
        dim0 = 1;
        dim1 = shape[0];
    }

    if(shape[rank-1]<CONFIG_M_AXI_WIDTH){
        //sub-vector padding
        _gcd = __gcd(dim1, CONFIG_M_AXI_WIDTH);
        lcm = (dim1*CONFIG_M_AXI_WIDTH)/(_gcd);
    }else{
        lcm=0;
    }
    
    shape[rank-1] = lastDimPadded;
    TensorF* rsltTn = new TensorF(shape);

    for(unsigned d0=0; d0<dim0; d0++){
        for(unsigned d1=0; d1<lastDimPadded; d1++){
            rsltTn->_buff[d0*lastDimPadded+d1] = (d1<dim1) ? inputTn->_buff[d0*dim1+d1] : 0;
        }
    }

    return rsltTn;
}

TensorF* CpuImplementation::UnpadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned lastDimUnpadded){
    PrintInfo("UnpadLastDim","lastDimUnpadded",lastDimUnpadded,"",0,"",0,inputTn->getShape(),{},{});
    if(lastDimUnpadded>=inputTn->getShape()[inputTn->getRank()-1]){cout<<"UnpadLastDim: ERROR_BAD_PARAM"<<endl;return nullptr;}

    unsigned dim0, dim1, argcnt;
    std::vector<unsigned> shape = inputTn->getShape();
    const unsigned rank = inputTn->getRank();

    if(rank!=1){
        dim0=1;
        for(int i=0; i<rank-1; i++){
            dim0*=shape[i];
        }
        dim1=shape[rank-1];
    }else{
        dim0 = 1;
        dim1 = shape[0];
    }
    
    shape[rank-1] = lastDimUnpadded;
    TensorF* rsltTn = new TensorF(shape);

    for(unsigned d0=0; d0<dim0; d0++){
        for(unsigned d1=0; d1<lastDimUnpadded; d1++){
            rsltTn->_buff[d0*lastDimUnpadded+d1] = inputTn->_buff[d0*dim1+d1];
        }
    }

    return rsltTn;
}