#pragma once

#include "TensorF.h"
#include "ocl_imp/OclTensorI.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"
#include "ocl_imp/xilinx/xcl.h"
#include <exception>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

extern int LaunchDataMover( 
    cl_program program,
    cl_command_queue queue,
    cl_context context,
    cl_mem srcBuff, 
    cl_mem dstBuff, 
    const unsigned srcBank, 
    const unsigned dstBank, 
    const unsigned len,
    const unsigned vectorWords);
