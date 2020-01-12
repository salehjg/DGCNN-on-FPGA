//
// Created by saleh on 12/23/19.
//

#ifndef SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H
#define SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H

template <typename DType, int VectorizationDepth>
struct VectorizedArray{
    DType vec[VectorizationDepth];
};

#define DO_PRAGMA(x) _Pragma ( #x )

// Be careful with parenthesis in a preprocessor, especially when the expression has math ops!
#define FlatIdx_to_VecIdx(vecDepth, flatIdx) (((unsigned long)flatIdx)/((unsigned long)vecDepth))
#define FlatIdx_to_VecSubIdx(vecDepth, flatIdx) (((unsigned long)flatIdx)%((unsigned long)vecDepth))

// Input and output tensor vectorization depth for each of the kernels:(words of 4-bytes)
// All instances of OclTensors without specified AXI_WIDTH, will be padded to be dividable to CONFIG_M_AXI_WIDTH.
#define CONFIG_M_AXI_WIDTH	16

// AXI bus width for inputTn(only) of task_gather(tensors are still padded to be devidable to CONFIG_M_AXI_WIDTH).
#define CONFIG_GATHER_INPUTTN_M_AXI_WIDTH 4

// AXI bus width for indicesSplitedTn(only) of task_topk(kValue should be devidable to CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH).
#define CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH 10

#endif //SDACCEL_CMAKE_SIMPLE_VECTORIZATIONHELPER_H
