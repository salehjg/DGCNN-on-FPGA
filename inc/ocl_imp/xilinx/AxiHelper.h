#pragma once

#define DO_PRAGMA(x) _Pragma ( #x )

template<typename T>
inline T DivCeil(T a, T b){
#pragma HLS INLINE
    return ((T)(a-1)/(T)b)+1;
}

template<typename T>
inline T MakeDivisible(T value, T by){
#pragma HLS INLINE
    return (value%by==0)?
            value:
            value+(by-value%by);
}
