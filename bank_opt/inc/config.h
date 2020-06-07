// main repository commit hash: 
// 75c575be95f81b442d57dfc91e67e77758ba6105


// config repository commit hash: 
// 537009ce085c5cb5a8becd05aad1d2d9f58dcbbe


#pragma once
#include <cstddef>
#include <cstdint>

using CONFIG_DTYPE = float;
constexpr unsigned CONFIG_DTYPE_SIZE = 4;
constexpr unsigned CONFIG_M_AXI_WIDTH = 16;

namespace ConfigTaskConv2
{
    constexpr int BankIndex_inputTn = 1;
    constexpr int BankIndex_weightTn = 1;
    constexpr int BankIndex_biasTn = 1;
    constexpr int BankIndex_outputTn = 1;

    constexpr int kMemoryWidthBytesN = 64;
    constexpr int kMemoryWidthBytesK = 64;
    constexpr int kMemoryWidthBytesM = 64;
    constexpr unsigned long kOuterTileSizeN = 256;
    constexpr unsigned long kOuterTileSizeM = 256;
    constexpr unsigned long kInnerTileSizeN = 16;
    constexpr int kComputeTileSizeM = 16; 
    constexpr int kComputeTileSizeN = 1;
    constexpr int kTransposeWidthBytes = 64;
    constexpr float kFrequency = 190; 

    #if 64 != 64 
        #define MM_CONVERT_B
    #endif
    // When A is not transposed, the data width must be converted if a PE buffers
    // more than one row of A (currently unsupported). When A is transposed, the
    // data width must be converted if the memory bus is wider than the number of
    // rows buffered per PE.
    #if (defined(MM_TRANSPOSED_A) && \
         (4 != 64))
        #define MM_CONVERT_A
    #endif
}

namespace ConfigTaskTopK
{
    constexpr int BankIndex_inputTn = 2;
    constexpr int BankIndex_indicesSplitedTn = 2;
    constexpr unsigned MaxSliceLen = 1024;
    constexpr unsigned PipeDepth = 16;
    constexpr unsigned UnitCount = 12;
}

namespace ConfigTaskMatOps
{
    constexpr int BankIndex_inputTn1 = 1;
    constexpr int BankIndex_inputTn2 = 1;
    constexpr int BankIndex_outputTn = 1;
}

namespace ConfigTaskReduce
{
    constexpr int BankIndex_inputTn  = 1; 
    constexpr int BankIndex_outputTn = 1;
    namespace Sum4D{
        constexpr int MaxPowY = 2;
        constexpr unsigned MaxSliceLen = 1024;
        constexpr unsigned PipeDepth = 8;
    }
    namespace Max3D{
        constexpr unsigned MaxSliceLen = 1024;
    }
    namespace Sum3D{
        constexpr unsigned MaxSliceLen = 64;
    }
}

namespace ConfigTaskMatMul
{
    constexpr int BankIndex_inputTn1 = 2;
    constexpr int BankIndex_inputTn2 = 2;
    constexpr int BankIndex_outputTn = 2;
    constexpr unsigned MaxM = 1024;
    constexpr unsigned RowTileSizeD = 4;
}

namespace ConfigTaskTile
{
    constexpr int BankIndex_inputTn = 2;
    constexpr int BankIndex_outputTn = 2;
    constexpr unsigned MaxSliceLen = 1024; 
}

namespace ConfigTaskGather
{
    constexpr int BankIndex_inputTn = 1; 
    constexpr int BankIndex_indicesTn = 1; 
    constexpr int BankIndex_outputTn = 1;
    constexpr unsigned PipeDepth = 4;
}

namespace ConfigTaskConcat 
{    
    constexpr int BankIndex_inputTn1 = 2;
    constexpr int BankIndex_inputTn2 = 2;
    constexpr int BankIndex_outputTn = 2;
}

namespace ConfigTaskTranspose
{
    constexpr int BankIndex_inputTn = 2;
    constexpr int BankIndex_outputTn = 2;
    constexpr unsigned TileWidth = 16;
    constexpr unsigned TileHeight = 64;
}

namespace ConfigTaskReluSqrtSquare
{
    constexpr int BankIndex_inputTn = 2;
    constexpr int BankIndex_outputTn = 2;
    constexpr unsigned ModeRelu = 0;
    constexpr unsigned ModeSqrt = 1;
    constexpr unsigned ModeSquare = 2;
}

namespace ConfigTaskPadUnpad
{
    constexpr int BankIndex_inputTn = 1;
    constexpr int BankIndex_outputTn = 1;
}
