//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_XILINXIMPLEMENTATION_H

#define DEEPPOINTV1_XILINXIMPLEMENTATION_H

#include <PlatformImplementation.h>
#include <ocl_imp/xilinx/xcl2.hpp>
#include <TensorF.h>
#include <TensorI.h>
#include <ocl_imp/OclTensorF.h>
#include <ocl_imp/OclTensorI.h>

#define XILINX_BOTTLENCK_BLOCKSIZE 1024

#define REPORT_EXECUTION_DURATION
//#undef REPORT_EXECUTION_DURATION

struct OclKernelObject{
    string fileName;
    string containerName;
    const char *kernelName_ndrange, *kernelName_task;
    cl::Kernel *kernel_ndrange,*kernel_task;
    bool use_ndrange_kernel;

    OclKernelObject(
    		string dir,
			string fname,
			string containerName,
			string kernelName_ndrange,
			string kernelName_task,
			bool use_ndrange_kernel){
        string *fPath = new string();
        fPath->append(dir);
        fPath->append(fname);
        fileName = *fPath;

        string *_kernelName = new string();
        _kernelName->append(kernelName_ndrange);
        this->kernelName_ndrange = _kernelName->c_str();

        string *_kernelName2 = new string();
        _kernelName2->append(kernelName_task);
        this->kernelName_task = _kernelName2->c_str();

        this->containerName = containerName;

        this->use_ndrange_kernel = use_ndrange_kernel;
    }
};

class XilinxImplementation: public PlatformImplementation {
public:
	XilinxImplementation(int aa); ///TODO: Constructor should handle platform initialization procedure!

    TensorF* Transpose(WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    TensorF* Square(WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode);
    TensorF* Sqrt(WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Concat2(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(WorkScheduler scheduler, TensorF* inputTn, int reductionDim);
    TensorI* TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2=-1);
    TensorF* ReLU(WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir);
    void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir);
    bool     CompareTensors(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    const char * getErrorString(cl_int error);

    cl::Context*         getContext();
    cl::CommandQueue*    getQueue();
    //void                ReadKernelSource(OclKernelObject *object);
    //void                GetPaddedWorkSize(int dims, size_t * inBlockSize, size_t * inWorkSize, size_t * outPaddedWorkSize);

    std::vector<OclKernelObject*> oclKernels;

private:
    int a;
    void PrintInfo(string opName, const string &setting1, int val1, const string &setting2, int val2,
                   const string &setting3, float val3, vector<unsigned int> shape1, vector<unsigned int> shape2, vector<bool> comb={});
    uint64_t get_duration_ns (const cl::Event &event);
    void ReportDuration(const std::string &name, const bool &isNDRange, const cl::Event &event);

    const std::string KERNEL_DIR = REPO_DIR "src/kernels";

    cl::Device          device;
    std::string			device_name;
    cl::Context         *context;
    cl::CommandQueue    *queue;
    cl_int              err;
};


#endif //DEEPPOINTV1_OCLIMPLEMENTATION_H
