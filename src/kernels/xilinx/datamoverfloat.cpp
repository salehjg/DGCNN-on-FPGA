#include <cassert>
#include <stdio.h> 
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace ConfigTaskDataMoverFloat;

/**
 * @brief      Data mover task kernel that
 *             copies data of length 'len':
 *                  reverseSwitch=0 --> from srcBuff to dstBuff, so, from bankA to bankB
 *                  reverseSwitch=1 --> from dstBuff to srcBuff, so, from bankB to bankA
 *
 * @param      srcBuff        srcBuff should ALWAYS be on bankA
 * @param      dstBuff        dstBuff should ALWAYS be on bankB
 * @param[in]  reverseSwitch  The reverse switch
 * @param[in]  len            len is number of (VecDepth*sizeof(DType)-bytes)words
 */
void _task_datamover(
        MemoryPackF_t *srcBuff,
        MemoryPackF_t *dstBuff,
        int reverseSwitch,
        const unsigned len){
#pragma HLS inline

    MemoryPackF_t buff;
    
    if(reverseSwitch==0){
        for(unsigned i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            buff = srcBuff[i];
            dstBuff[i] = buff;
        }
    }else{
        for(unsigned i=0;i<len;i++){
        #pragma HLS PIPELINE II=1
            buff = dstBuff[i];
            srcBuff[i] = buff;
        }
    }
}

extern "C" {

// The --sp option decides which bank is bankA and which is bankB
// Currently, bankA is bank1 & bankB is bank2 and no SLR assignment is done.
void task_datamover_mod1_float(
        MemoryPackF_t *srcBuff,
        MemoryPackF_t *dstBuff,
        int reverseSwitch,
        const unsigned len){
#pragma HLS INTERFACE m_axi     port=srcBuff            offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=dstBuff            offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=srcBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=dstBuff            bundle=control
#pragma HLS INTERFACE s_axilite port=reverseSwitch      bundle=control
#pragma HLS INTERFACE s_axilite port=len                bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    _task_datamover(srcBuff, dstBuff, reverseSwitch, len);
}

}
