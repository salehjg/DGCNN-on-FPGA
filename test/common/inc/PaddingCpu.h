#pragma once
#include <vector>

template<typename T>
void PadTensor(
	const std::vector<T> &inputTn,
	std::vector<T> &outputTn, 
	unsigned dim0, 
	unsigned dim1, 
	unsigned dim1Padded){

    for(unsigned d0=0; d0<dim0; d0++){
        for(unsigned d1=0; d1<dim1Padded; d1++){
            outputTn[d0*dim1Padded+d1] = (d1<dim1) ? inputTn[d0*dim1+d1] : 0;
        }
    }
}