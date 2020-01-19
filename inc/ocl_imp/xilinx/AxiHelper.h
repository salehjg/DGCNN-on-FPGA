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

#endif //AxiHelper_H
