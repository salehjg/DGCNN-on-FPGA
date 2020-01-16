//
// Created by saleh on 12/23/19.
//

#ifndef AxiHelper_H
#define AxiHelper_H

template <typename DType, int Depth>
struct PackedArray{
    DType vec[Depth];
};

#define DO_PRAGMA(x) _Pragma ( #x )

// Be careful with parenthesis in a preprocessor, especially when the expression has math ops!
#define FlatIdx_to_VecIdx(vecDepth, flatIdx) (((unsigned long)flatIdx)/((unsigned long)vecDepth))
#define FlatIdx_to_VecSubIdx(vecDepth, flatIdx) (((unsigned long)flatIdx)%((unsigned long)vecDepth))

// * Input and output tensor vectorization depth for each of the kernels:(words of 4-bytes)
// * All instances of OclTensors without specified AXI_WIDTH, will be padded to be dividable to CONFIG_M_AXI_WIDTH.
// * Should be a power of two.
#define CONFIG_M_AXI_WIDTH	16

// * AXI bus width for inputTn(only) of task_gather(tensors are still padded to be devidable to CONFIG_M_AXI_WIDTH).
// * This will not cause any problem if the respective tensor is padded for 16-words wide m_axi bus, because 
//   len % 16 is always greater than len % 4, so there won't be any out of bound memory accesses.
// * Should be a power of two.
#define CONFIG_GATHER_INPUTTN_M_AXI_WIDTH 4

// * AXI bus width for indicesSplitedTn(only) of task_topk(kValue should be devidable to CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH).
// * This will not cause any problem if the respective tensor is padded for 16-words wide m_axi bus, because 
//   len % 16 is always greater than len % 4, so there won't be any out of bound memory accesses.
// * Should be a power of two.
#define CONFIG_TOPK_OUTPUTTN_M_AXI_WIDTH 4

#endif //AxiHelper_H
