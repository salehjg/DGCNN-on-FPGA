#pragma once

#include "TensorF.h"
#include "ocl_imp/OclTensorI.h"
#include "ocl_imp/xilinx/AxiHelper.h"
#include "xilinx/config.h"
#include "ocl_imp/xilinx/xcl2.hpp"

using namespace std;

extern int LaunchDataMover(
        cl::Program *program,
        cl::CommandQueue *queue,
        cl::Context *context,
        cl::Buffer &srcBuff,
        cl::Buffer &dstBuff,
        const unsigned srcBank,
        const unsigned dstBank,
        const unsigned len,
        const unsigned vectorWords);
