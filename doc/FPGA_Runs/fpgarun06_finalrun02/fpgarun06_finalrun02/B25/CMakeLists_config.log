#-------------------------------------------------------------------------------
# CFG0: General Configurations

set(CFG0_DATA_TYPE "float" CACHE STRING "Data-type(float)")
set(CFG0_M_AXI_WIDTH "16" CACHE STRING "General Preferred AXI Width(words of four-bytes)")

check_type_size(${CFG0_DATA_TYPE} DATA_WIDTH_${CFG0_DATA_TYPE})
if(NOT DATA_WIDTH_${CFG0_DATA_TYPE}) # Non-primitive data type
    if(CFG0_DATA_TYPE STREQUAL "half")
        set(DATA_WIDTH_${CFG0_DATA_TYPE} 2)
    elseif(CFG0_DATA_TYPE STREQUAL "uint8_t")
        set(DATA_WIDTH_${CFG0_DATA_TYPE} 1)
    else()
        message(FATAL_ERROR "Could not get size of data type ${CFG0_DATA_TYPE}.") 
    endif()
endif()
set(CFG0_DATA_TYPE_SIZE "${DATA_WIDTH_${CFG0_DATA_TYPE}}")

#-------------------------------------------------------------------------------
# CFG1: conv2_1x1 (gemm_hls)
set(CFG1_Conv2_DDRBANK_inputTn "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG1_Conv2_DDRBANK_weightTn "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG1_Conv2_DDRBANK_biasTn "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG1_Conv2_DDRBANK_outputTn "0" CACHE STRING "DDR bank index for the given parameter")

set(CFG1_CONV2_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(CFG1_CONV2_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(CFG1_CONV2_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(CFG1_CONV2_MEMORY_TILE_SIZE_N 256 CACHE STRING "Tile size of outer memory tile in N(i-dimension)(x_b*x_t).")
set(CFG1_CONV2_MEMORY_TILE_SIZE_M 128 CACHE STRING "Tile size of outer memory tile in M(j-dimension)(y_b*y_t).")
set(CFG1_CONV2_PARALLELISM_N 16 CACHE STRING "Number of parallel compute in N(i-dimension)(InnerTileSizeN)(x_p).")
set(CFG1_CONV2_PARALLELISM_M 8 CACHE STRING "Number of parallel compute in M(j-dimension)(ComputeTileSizeM)(y_c).")
set(CFG1_CONV2_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(CFG1_CONV2_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(CFG1_CONV2_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(CFG1_CONV2_ADD_RESOURCE OFF CACHE STRING "")
set(CFG1_CONV2_MULT_RESOURCE OFF CACHE STRING "") 

math(EXPR CFG1_CONV2_KERNEL_WIDTH_N "${CFG0_DATA_TYPE_SIZE} * ${CFG1_CONV2_PARALLELISM_N}")
math(EXPR CFG1_CONV2_KERNEL_WIDTH_M "${CFG0_DATA_TYPE_SIZE} * ${CFG1_CONV2_PARALLELISM_M}")
math(EXPR CFG1_CONV2_INNER_TILES "(${CFG1_CONV2_MEMORY_TILE_SIZE_N} / ${CFG1_CONV2_PARALLELISM_N}) * (${CFG1_CONV2_MEMORY_TILE_SIZE_M} / ${CFG1_CONV2_PARALLELISM_M})")
if(NOT CFG1_CONV2_TRANSPOSED_A AND CFG1_CONV2_MEMORY_TILE_SIZE_N GREATER CFG1_CONV2_INNER_TILES)
    message(WARNING "In-memory transposition for A cannot keep up with the instantiated number of compute units. The number of inner tiles (currently ${CFG1_CONV2_INNER_TILES}) must be greater than or equal to the outer tile size in N (currently ${CFG1_CONV2_MEMORY_TILE_SIZE_N}).")
endif()
math(EXPR CFG1_CONV2_MEMORY_WIDTH_M "${CFG1_CONV2_MEMORY_BUS_WIDTH_M} / ${CFG0_DATA_TYPE_SIZE}")
math(EXPR CFG1_CONV2_REM "${CFG1_CONV2_MEMORY_WIDTH_M} % ${CFG1_CONV2_PARALLELISM_M}")
if(CFG1_CONV2_REM GREATER 0)
    message(FATAL_ERROR "Bus width in M (${CFG1_CONV2_MEMORY_WIDTH_M}) must be a multiple of the parallelism in M (${CFG1_CONV2_PARALLELISM_M}).")
endif()
math(EXPR CFG1_CONV2_REM "${CFG1_CONV2_MEMORY_TILE_SIZE_M} % ${CFG1_CONV2_MEMORY_WIDTH_M}")
if(CFG1_CONV2_REM GREATER 0)
    message(FATAL_ERROR "Outer memory tile size must be divisible by element width of memory bus.")
endif()

#-------------------------------------------------------------------------------
# CFG2: TopK

# General Options:
set(CFG2_TopK_DDRBANK_inputTn "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG2_TopK_DDRBANK_indicesSplitedTn "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG2_TopK_UnitCount "1" CACHE STRING "Number of PEs for Topk")
set(CFG2_TopK_MaxK "20" CACHE STRING "The maximum possible value for K")
set(CFG2_TopK_MaxSliceLen "1024" CACHE STRING "Maximum length of the input slices")

# SelectionSort and MergeSort Based Kernel's Options:
#set(CFG2_TopK_PipeDepth "8" CACHE STRING "FIFO depth, recommended to be greater or equal to the number of vectors per input silce")


#-------------------------------------------------------------------------------
# CFG3: MatOps
set(CFG3_MatOps_DDRBANK_inputTn1 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG3_MatOps_DDRBANK_inputTn2 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG3_MatOps_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG3_MatOps_PipeDepth  "1" CACHE STRING "FIFO depth")

#-------------------------------------------------------------------------------
# CFG4: Reduce
set(CFG4_Reduce_DDRBANK_inputTn  "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG4_Reduce_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG4_ReduceSum4D_MaxPowY "2" CACHE STRING "Maximum value for kernel parameter pow_y")
set(CFG4_ReduceSum4D_MaxSliceLen "1024" CACHE STRING "Maximum length of input slices")
#set(CFG4_ReduceSum4D_PipeDepth "8" CACHE STRING "FIFO depth, 8 means 8*M_AXI_WIDTH")
set(CFG4_ReduceMax_MaxSliceLen "1024" CACHE STRING "Maximum length of the input slices")
set(CFG4_ReduceSum_MaxSliceLen "64" CACHE STRING "Maximum length of the input slices")
set(CFG4_ReduceSum_PipeDepth "1" CACHE STRING "FIFO depth")

#-------------------------------------------------------------------------------
# CFG5: MatMul
set(CFG5_MatMul_DDRBANK_inputTn1 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG5_MatMul_DDRBANK_inputTn2 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG5_MatMul_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG5_MatMul_MaxM "1024" CACHE STRING "Maximum value for M of matB of shape BxKxM")
set(CFG5_MatMul_RowTileSizeD "4" CACHE STRING "Row-tile Size(D)")

#-------------------------------------------------------------------------------
# CFG6: Tile
set(CFG6_Tile_DDRBANK_inputTn  "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG6_Tile_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG6_Tile_MaxSliceLen "1024" CACHE STRING "Maximum length of input slices")

#-------------------------------------------------------------------------------
# CFG7: Gather
set(CFG7_Gather_DDRBANK_inputTn   "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG7_Gather_DDRBANK_indicesTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG7_Gather_DDRBANK_outputTn  "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG7_Gather_PipeDepth  "1" CACHE STRING "FIFO depth")

#-------------------------------------------------------------------------------
# CFG8: Concat
set(CFG8_Concat_DDRBANK_inputTn1 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG8_Concat_DDRBANK_inputTn2 "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG8_Concat_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG8_Concat_PipeDepth  "1" CACHE STRING "FIFO depth")

#-------------------------------------------------------------------------------
# CFG9: Transpose
set(CFG9_Transpose_DDRBANK_inputTn  "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG9_Transpose_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG9_Transpose_PipeDepth1 "32" CACHE STRING "FIFO depth for transposition operation, must be divisible by M_AXI_WIDTH.")
set(CFG9_Transpose_PipeDepth2 "32" CACHE STRING "FIFO depth for write operation. Suggested to be greater than PipeDepth1/M_AXI_WIDTH.")

#-------------------------------------------------------------------------------
# CFG10: ReluSqrtSquare
set(CFG10_ReluSqrtSquare_DDRBANK_inputTn  "1" CACHE STRING "DDR bank index for the given parameter")
set(CFG10_ReluSqrtSquare_DDRBANK_outputTn "1" CACHE STRING "DDR bank index for the given parameter")

#-------------------------------------------------------------------------------
# CFG11: Pad/Unpad
set(CFG11_PadUnpad_DDRBANK_inputTn  "0" CACHE STRING "DDR bank index for the given parameter")
set(CFG11_PadUnpad_DDRBANK_outputTn "0" CACHE STRING "DDR bank index for the given parameter")

#-------------------------------------------------------------------------------
# CFG12: DataMover
set(CFG12_DataMover_PipeDepth  "1" CACHE STRING "FIFO depth")

execute_process(
    COMMAND
        git --git-dir=${PROJECT_SOURCE_DIR}/config/.git rev-parse HEAD
    RESULT_VARIABLE
        HASH_RESULT
    OUTPUT_VARIABLE
        CONFIG_HEAD_HASH)

execute_process(
    COMMAND
        git --git-dir=${PROJECT_SOURCE_DIR}/.git rev-parse HEAD
    RESULT_VARIABLE
        HASH_RESULT
    OUTPUT_VARIABLE
        MAIN_HEAD_HASH)

configure_file(${PROJECT_SOURCE_DIR}/config/raw/xilinx/config.h.in ${PROJECT_SOURCE_DIR}/config/output/xilinx/config.h)
configure_file(${PROJECT_SOURCE_DIR}/config/output/xilinx/config.h ${CMAKE_BINARY_DIR}/config.h.log COPYONLY)
include(${CMAKE_SOURCE_DIR}/config/cmake_inc/report_memory_banks.cmake)
include(${CMAKE_SOURCE_DIR}/config/cmake_inc/misc_sanity_checks.cmake)
