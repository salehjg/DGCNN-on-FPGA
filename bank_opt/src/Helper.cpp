#include "Helper.h"

enum class PLATFORMS{
    DEFAULT,
    CPU
};

enum class MAT_OPS{
    ADD,
    SUB,
    MUL_ELEMENTWISE,
    DIV_ELEMENTWISE
};

template<typename T>
T DivCeil(T a, T b){
    return ((T)(a-1)/(T)b)+1;
}

template<typename T>
T MakeDivisible(T value, T by){
    return (value%by==0)?
            value:
            value+(by-value%by);
}