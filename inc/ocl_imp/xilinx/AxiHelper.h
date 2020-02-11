//
// Created by saleh on 12/23/19.
//

#ifndef AxiHelper_H
#define AxiHelper_H

#define DO_PRAGMA(x) _Pragma ( #x )

template<typename T>
T DivCeil(T a, T b){
#pragma HLS INLINE
	return ((T)(a-1)/(T)b)+1;
}

#endif //AxiHelper_H
