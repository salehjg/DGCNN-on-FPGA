//
// Created by saleh on 8/22/18.
//

#pragma once

#include <PlatformImplementation.h>
#include <ocl_imp/xilinx/xcl.h>
#include <TensorF.h>
#include <TensorI.h>
#include <ocl_imp/OclTensorF.h>
#include <ocl_imp/OclTensorI.h>
#include <cnpy.h>
#include <ocl_imp/xilinx/AxiHelper.h>

#define XILINX_BOTTLENCK_BLOCKSIZE 1024

//#define REPORT_EXECUTION_DURATION
#undef REPORT_EXECUTION_DURATION

#define DUMP_ENABLED
//#undef DUMP_ENABLED

enum class RUN_MODE{
    SwEmu,
    HwEmu,
    Hw,
    Unknown
};

struct OclKernelObject{
    string fileName;
    string containerName;
    const char *kernelName_ndrange, *kernelName_task;
    cl_kernel kernel_ndrange,kernel_task;
    bool use_ndrange_kernel;
    bool disabled;

    OclKernelObject(
            string dir,
            string fname,
            string containerName,
            string kernelName_ndrange,
            string kernelName_task,
            bool use_ndrange_kernel,
            bool isDisabled=false){
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

        this->disabled = isDisabled;
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
    bool     CompareTensorsInteger(WorkScheduler scheduler, TensorI* inputTn1, TensorI* inputTn2);
    TensorF* PadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned int lastDimPadded);
    TensorF* UnpadLastDim(WorkScheduler scheduler, TensorF* inputTn, unsigned int lastDimUnpadded);

    const char * getErrorString(cl_int error);
    int SetModeEnvVar(const RUN_MODE mode);
    RUN_MODE GetModeEnvVar();

    cl_context          getContext();
    cl_command_queue    getQueue();
    cl_program          getProgram();
    void                GetPaddedWorkSize(int dims, size_t * inBlockSize, size_t * inWorkSize, size_t * outPaddedWorkSize);
    ~XilinxImplementation();

    std::vector<OclKernelObject*> oclKernels;

private:
    int a;
    void PrintInfo(string opName, const string &setting1, int val1, const string &setting2, int val2,
                   const string &setting3, float val3, vector<unsigned int> shape1, vector<unsigned int> shape2, vector<bool> comb={});
    cl_ulong get_duration_ns (const cl_event &event);
    void ReportDuration(const std::string &name, const bool &isNDRange, const cl_event &event);
    TensorF* _ReduceSum4D_Task(
            TensorF* inputTn,
            bool overaxis0,
            bool overaxis1,
            bool overaxis2,
            bool overaxis3,
            int pow_y);

    TensorF* _ReduceSum4D(WorkScheduler scheduler,
            TensorF* inputTn,
            bool over_axis0,
            bool over_axis1,
            bool over_axis2,
            bool over_axis3,
            int pow_y);

    TensorF* _ReluSqrtSquare(WorkScheduler scheduler, TensorF* inputTn, bool runRelu, bool runSqrt, bool runSquare);

    const std::string KERNEL_DIR = REPO_DIR "src/kernels";

    std::string deviceName;
    cl_platform_id *_platforms;        // OpenCL platform
    cl_device_id *_devices;           // device ID
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id deviceId;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    char *binary_content;             // program binary content
    cl_int err;
};
