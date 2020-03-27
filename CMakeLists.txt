###############################################################################################################
# Targets:
#           synthesis       : Synthesizes all of the kernels
#           ---------------------------------------------------------------------------------------------------
#           compile_hw      : Compiles all of the kernels to generate *.xo files. (system build equivalent)
#           link_hw         : Links all of the *.xo files to generate a xclbin file. (system link equivalent)
#           interactive_hw  : Experimental...
#           ---------------------------------------------------------------------------------------------------
#           compile_hwemu   : Compiles all of the kernels to generate *.xo files. (hw-emu build equivalent)
#           link_hwemu      : Links all of the *.xo files to generate a xclbin file. (hw-emu link equivalent)
#           ---------------------------------------------------------------------------------------------------
###############################################################################################################


cmake_minimum_required(VERSION 3.0)
project(DeepPointV1FPGA)
set(HostExecutableName DeepPointV1FPGA)
if(NOT DEFINED ENV{AWS_PLATFORM})
    message(FATAL_ERROR "You must set env variable AWS_PLATFORM to the path of the SDAccel platform file(*.xpfm)")
endif()

#==========================================================================================================================================



# $ENV{AWS_PLATFORM}   OR    xilinx_kcu1500_dynamic_5_0
set(DSA_NAME "$ENV{AWS_PLATFORM}" CACHE STRING "Known SDAccel platform name or xpfm file path") 
set(SYNTH_PART_NAME "xcvu9p-flga2104-2-e" CACHE STRING "Part name for synthesis only.")
set(SYNTH_FLAGS "-DKERNEL_LOGS -DHLSLIB_SYNTHESIS -DHLSLIB_XILINX -std=c++11 -I${CMAKE_SOURCE_DIR}/inc/ocl_imp/xilinx -I${CMAKE_SOURCE_DIR}/config/output -I${CMAKE_SOURCE_DIR}/hlslib/include" CACHE STRING "CFlags for synthesis only.")
set(ENABLE_PROFILING ON CACHE BOOL "Collect profiling information")
#set(ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design")
set(OPTIMIZATION_OPTION "0" CACHE STRING "0:default, 1:reduce power, 2:increase kernel speed, 3:highest level of optimization, s:optimize for size, quick:quick compilation")
set(XOCC_FLAGS "" CACHE STRING "Extra flags for xocc")
set(KERNEL_CLOCK "-1" CACHE STRING "MHz, For XOCC Compile and Link Procedures. All of the kernels will use this clock frequency. Set to -1 to ignore frequency override.")
set(KERNEL_CLOCK_UNCERTAINTY "0.9" CACHE STRING "Clock Uncertainty")

set(PASTEBIN_0_1 "0" CACHE STRING "Enable or disable PasteBin agent for uploading autobuild log files to PasteBin.") 
set(PASTEBIN_USERNAME "" CACHE STRING "PasteBin Account's Username") 
set(PASTEBIN_PASSWORD "" CACHE STRING "PasteBin Account's Password") 
set(PASTEBIN_API_KEY  "" CACHE STRING "PasteBin Account's API Key") 

set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_SOURCE_DIR}/hlslib/cmake
    )

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)
include(CheckTypeSize)


#==========================================================================================================================================



include_directories(config/output inc hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
add_definitions(-DREPO_DIR=\"${CMAKE_SOURCE_DIR}\")
add_definitions(-DBIN_DIR=\"${CMAKE_BINARY_DIR}\")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#add_definitions("-DHLSLIB_STREAM_TIMEOUT=16")
if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
  add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
  add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()



#==========================================================================================================================================



set(hw_link_command "")
set(hwemu_link_command "")
set(swemu_link_command "")
set(hw_link "")
set(hwemu_link "")
set(swemu_link "")



#==========================================================================================================================================



function(floatexpr expr output)
    set(FLOAT_EXPR_SRC "
    #include <stdio.h>
    int main(int argc, char *argv[]){printf(\"%f\", ${expr})\; return 0\;}
    ")
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFloatExpr.c ${FLOAT_EXPR_SRC})

    try_run(RESULT COMPRES ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFloatExpr.c
            RUN_OUTPUT_VARIABLE __output)

    set(${output} ${__output} PARENT_SCOPE)
endfunction()



#==========================================================================================================================================



# Creates TCL scripts for the given kernel.
function(synthesis_target 
    KERNEL_FILE 
    KERNEL_NAME)

    set(HLSLIB_PART_NAME "${SYNTH_PART_NAME}")
    set(HLSLIB_PROJECT_NAME "${KERNEL_NAME}")
    set(HLSLIB_SRC_SYNTHESIS "${KERNEL_FILE}")
    set(HLSLIB_ENTRY_FUNCTION "${KERNEL_NAME}")
    set(HLSLIB_TARGET_CLOCK_UNCERTAINTY "${KERNEL_CLOCK_UNCERTAINTY}")
    set(HLSLIB_SYNTHESIS_FLAGS ${SYNTH_FLAGS})

    if(KERNEL_CLOCK GREATER 0)
        set(CLK "${KERNEL_CLOCK}")
    else()
        set(CLK "200")
    endif()
    set(HLSLIB_TARGET_CLOCK "${CLK}")

    configure_file(scripts/Synthesis.tcl.in Synthesize${KERNEL_NAME}.tcl)

endfunction()



#==========================================================================================================================================



# Generates and appends commands for compiling and linking the given kernel.
function(sdaccel_target 
    KERNEL_FILE
    KERNEL_FNAME_WITHOUT_EXTENSION 
    KERNEL_NAME  
    IS_PROFILED
    IS_MULTI_DDR_BANK
    MULTI_DDR_SP_TAG
    ADDITIONAL_FLAGS)
    set(KERNEL_FLAGS "${SYNTH_FLAGS}")
    #string(REPLACE " " ";" XOCC_FLAGS "${XOCC_FLAGS}")
    #set(SYNTH_FLAGS "") #should be ';' seperated
    set(XOCC_COMMAND
        -s
        ${XOCC_FLAGS}
        -I${CMAKE_SOURCE_DIR}/inc
        -I${CMAKE_SOURCE_DIR}/hlslib/include
        -I${CMAKE_BINARY_DIR}
        --kernel ${KERNEL_NAME} 
        --platform ${DSA_NAME}
        --xp prop:kernel.${KERNEL_NAME}.kernel_flags="${KERNEL_FLAGS}"
        --xp misc:solution_name="${KERNEL_NAME}_solution" 
        --max_memory_ports all
    )

    if(ENABLE_PROFILING)
        set(XOCC_COMMAND ${XOCC_COMMAND}
            --profile_kernel "data:all:all:all"
            --profile_kernel "stall:all:all"
            --profile_kernel "exec:all:all")
    endif()

    #if(ENABLE_DEBUGGING)
    #  set(XOCC_COMMAND ${XOCC_COMMAND} -g
    #    --dk chipscope:${KERNEL_NAME}_1:M_AXI_GMEM0
    #    --dk chipscope:${KERNEL_NAME}_1:M_AXI_GMEM1
    #    --dk chipscope:${KERNEL_NAME}_1:M_AXI_GMEM2)
    #endif()
    if(KERNEL_CLOCK GREATER 0)
        if((${SDAccel_MAJOR_VERSION} EQUAL 2017) AND (${SDAccel_MINOR_VERSION} EQUAL 4))
            set(XOCC_COMMAND ${XOCC_COMMAND}
                --xp prop:solution.hls_pre_tcl=KernelFreqSDx20174.tcl
            )
        else()
            set(XOCC_COMMAND ${XOCC_COMMAND}
                --kernel_frequency ${KERNEL_CLOCK}
            )
        endif()
    endif()
    string (REPLACE ";" " " XOCC_COMMAND "${XOCC_COMMAND}")

    set(hw_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -c -t hw ${XOCC_COMMAND} -R2 ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw.xo")
    set(hwemu_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -c -t hw_emu ${XOCC_COMMAND} -Restimate ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw_emu.xo") 
    set(swemu_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -c -t sw_emu ${XOCC_COMMAND} ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_sw_emu.xo") 
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "${hw_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "${hwemu_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "${swemu_compile}\n")

    #---------------------------------------------------------------------------------------------------------
    # Generating linker commands
    set(hw_link "${hw_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw.xo" PARENT_SCOPE) 
    set(hwemu_link "${hwemu_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw_emu.xo" PARENT_SCOPE) 
    set(swemu_link "${swemu_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_sw_emu.xo" PARENT_SCOPE) 

    set(hw_link_command "${hw_link_command} --nk ${KERNEL_NAME}:1")
    if(IS_PROFILED)
        set(hw_link_command "${hw_link_command} --profile_kernel data:${KERNEL_NAME}:all:all")
    endif()
    if(IS_MULTI_DDR_BANK)
        set(hw_link_command "${hw_link_command} ${MULTI_DDR_SP_TAG}")
    endif()

    # Identical commands for hw-emu link process
    set(hwemu_link_command "${hw_link_command}")
    set(swemu_link_command "${hw_link_command}")

    # Set link commands globally
    set(hw_link_command         "${hw_link_command}"        PARENT_SCOPE)
    set(hwemu_link_command      "${hwemu_link_command}"     PARENT_SCOPE)
    set(swemu_link_command      "${swemu_link_command}"     PARENT_SCOPE)
    #---------------------------------------------------------------------------------------------------------
endfunction()




#==========================================================================================================================================




configure_file(scripts/SynthesisScriptLauncherBatch.sh.in SynthesisScriptLauncherBatch.sh)
add_custom_target(synthesis COMMAND sh SynthesisScriptLauncherBatch.sh) 

# Creating required tcl scripts to run hls synthesis
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/concat.cpp"             "task_concat"              )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/conv2_1x1_direct.cpp"   "task_conv2_1x1_direct"    )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverfloat.cpp"     "task_datamover_mod1_float")
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverinteger.cpp"   "task_datamover_mod1_int"  )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/padding.cpp"            "task_pad_last_dim"        )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/unpadding.cpp"          "task_unpad_last_dim"      )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/gather.cpp"             "task_gather"              )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matmul.cpp"             "task_matmul"              )
###synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matops.cpp"             "task_matops"              )
###synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducemax.cpp"          "task_reducemax"           )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum.cpp"          "task_reducesum"           )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp"        "task_reducesum4d"         )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/relu.cpp"               "task_relu"                )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/sqrt.cpp"               "task_sqrt"                )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/square.cpp"             "task_square"              )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/tile.cpp"               "task_tile"                )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk_selectionsort.cpp"               "task_topk"                )
#synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp"          "task_transpose"           )








#==========================================================================================================================================







#Setting frequency override option.
if(KERNEL_CLOCK GREATER 0)
    if((${SDAccel_MAJOR_VERSION} EQUAL 2017) AND (${SDAccel_MINOR_VERSION} EQUAL 4))
        floatexpr("1000.0 / ${KERNEL_CLOCK}" KERNEL_PERIOD)
        message(WARNING "Bug workaround for SDx2017.4 : https://forums.xilinx.com/t5/SDAccel/How-to-set-the-target-frequency-for-the-kernel-in-SDACCL-2018-2/m-p/916088 https://forums.xilinx.com/t5/SDAccel/sdx2017-4-XOCC-kernel-compiler-cannot-set-kernel-frequency/td-p/894185")
        message("Calculated period for all of the kernels: ${KERNEL_PERIOD} nanoseconds")
        configure_file(scripts/KernelFreqSDx20174.tcl.in KernelFreqSDx20174.tcl) 
        set(hw_link_command "${hw_link_command} --xp prop:solution.hls_pre_tcl=${CMAKE_BINARY_DIR}/KernelFreqSDx20174.tcl")
    else()
        message("Overriding default kernel clock frequency normally with the kernel_frequency option...")
        set(hw_link_command "${hw_link_command} --kernel_frequency ${KERNEL_CLOCK}")
    endif()
    set(KERNEL_CLOCK_RESOLVED ${KERNEL_CLOCK})
else()
    set(KERNEL_CLOCK_RESOLVED 200)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/config)

file(WRITE "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")





#==========================================================================================================================================

set(SP_TAG_CONV2D           
"--sp task_conv2_1x1_direct_1.a:bank${CFG1_Conv2_DDRBANK_inputTn}\
 --sp task_conv2_1x1_direct_1.b:bank${CFG1_Conv2_DDRBANK_weightTn}\
 --sp task_conv2_1x1_direct_1.e:bank${CFG1_Conv2_DDRBANK_biasTn}\
 --sp task_conv2_1x1_direct_1.c:bank${CFG1_Conv2_DDRBANK_outputTn}")

set(SP_TAG_TOPK             
"--sp task_topk_1.inputTn:bank${CFG2_TopK_DDRBANK_inputTn}\
 --sp task_topk_1.indicesSplitedTn:bank${CFG2_TopK_DDRBANK_indicesSplitedTn}")

set(SP_TAG_MATOPS           
"--sp task_matops_1.inputTn1:bank${CFG3_MatOps_DDRBANK_inputTn1}\
 --sp task_matops_1.inputTn2:bank${CFG3_MatOps_DDRBANK_inputTn2}\
 --sp task_matops_1.outputTn:bank${CFG3_MatOps_DDRBANK_outputTn}")

set(SP_TAG_REDUCESUM4D      
"--sp task_reducesum4d_1.inputTn:bank${CFG4_ReduceSum4D_DDRBANK_inputTn}\
 --sp task_reducesum4d_1.outputTn:bank${CFG4_ReduceSum4D_DDRBANK_outputTn}")

set(SP_TAG_MATMUL           
"--sp task_matmul_1.inputTn1:bank${CFG5_MatMul_DDRBANK_inputTn1}\
 --sp task_matmul_1.inputTn2:bank${CFG5_MatMul_DDRBANK_inputTn2}\
 --sp task_matmul_1.outputTn:bank${CFG5_MatMul_DDRBANK_outputTn}")

set(SP_TAG_TILE             
"--sp task_tile_1.inputTn:bank${CFG6_Tile_DDRBANK_inputTn}\
 --sp task_tile_1.outputTn:bank${CFG6_Tile_DDRBANK_outputTn}")

set(SP_TAG_GATHER           
"--sp task_gather_1.inputTn:bank${CFG7_Gather_DDRBANK_inputTn}\
 --sp task_gather_1.indicesTn:bank${CFG7_Gather_DDRBANK_indicesTn}\
 --sp task_gather_1.outputTn:bank${CFG7_Gather_DDRBANK_outputTn}")

set(SP_TAG_RELU             
"--sp task_relu_1.inputTn:bank${CFG8_Relu_DDRBANK_inputTn}\
 --sp task_relu_1.outputTn:bank${CFG8_Relu_DDRBANK_outputTn}")

set(SP_TAG_CONCAT
"--sp task_concat_1.inputTn1:bank${CFG9_Concat_DDRBANK_inputTn1}\
 --sp task_concat_1.inputTn2:bank${CFG9_Concat_DDRBANK_inputTn2}\
 --sp task_concat_1.outputTn:bank${CFG9_Concat_DDRBANK_outputTn}")

set(SP_TAG_REDUCEMAX        
"--sp task_reducemax_1.inputTn:bank${CFG10_ReduceMax_DDRBANK_inputTn}\
 --sp task_reducemax_1.outputTn:bank${CFG10_ReduceMax_DDRBANK_outputTn}")

set(SP_TAG_TRANSPOSE        
"--sp task_transpose_1.inputTn:bank${CFG11_Transpose_DDRBANK_inputTn}\
 --sp task_transpose_1.outputTn:bank${CFG11_Transpose_DDRBANK_outputTn}")

set(SP_TAG_SQUARE           
"--sp task_square_1.inputTn:bank${CFG12_Square_DDRBANK_inputTn}\
 --sp task_square_1.outputTn:bank${CFG12_Square_DDRBANK_outputTn}")

set(SP_TAG_REDUCESUM        
"--sp task_reducesum_1.inputTn:bank${CFG13_ReduceSum_DDRBANK_inputTn}\
 --sp task_reducesum_1.outputTn:bank${CFG13_ReduceSum_DDRBANK_outputTn}")

set(SP_TAG_SQRT             
"--sp task_sqrt_1.inputTn:bank${CFG14_Sqrt_DDRBANK_inputTn}\
 --sp task_sqrt_1.outputTn:bank${CFG14_Sqrt_DDRBANK_outputTn}")

set(SP_TAG_DATAMOVER_FLOAT  
"--sp task_datamover_mod1_float_1.srcBuff:bank${CFG15_DataMoverF_DDRBANK_srcBuff}\
 --sp task_datamover_mod1_float_1.dstBuff:bank${CFG15_DataMoverF_DDRBANK_dstBuff}")

set(SP_TAG_DATAMOVER_INT    
"--sp task_datamover_mod1_int_1.srcBuff:bank${CFG16_DataMoverI_DDRBANK_srcBuff}\
 --sp task_datamover_mod1_int_1.dstBuff:bank${CFG16_DataMoverI_DDRBANK_dstBuff}")

set(SP_TAG_PADDING          
"--sp task_pad_last_dim_1.inputTn:bank${CFG17_Padding_DDRBANK_inputTn}\
 --sp task_pad_last_dim_1.outputTn:bank${CFG17_Padding_DDRBANK_outputTn}")

set(SP_TAG_UNPADDING        
"--sp task_unpad_last_dim_1.inputTn:bank${CFG18_Unpadding_DDRBANK_inputTn}\
 --sp task_unpad_last_dim_1.outputTn:bank${CFG18_Unpadding_DDRBANK_outputTn}")

###sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/conv2_1x1_direct.cpp"    "conv2_1x1_direct"  "task_conv2_1x1_direct"     FALSE TRUE ${SP_TAG_CONV2D} "" )

###sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk_selectionsort.cpp"      "topk"              "task_topk"                 FALSE TRUE ${SP_TAG_TOPK} "" )
###sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk_insertionsort.cpp"      "topk"              "task_topk"                 FALSE TRUE ${SP_TAG_TOPK} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk_mergesort.cpp"      "topk"              "task_topk"                 FALSE TRUE ${SP_TAG_TOPK} "" )

#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matops.cpp"              "matops"            "task_matops"               FALSE TRUE ${SP_TAG_MATOPS} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp"         "reducesum4d"       "task_reducesum4d"          FALSE TRUE ${SP_TAG_REDUCESUM4D} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matmul.cpp"              "matmul"            "task_matmul"               FALSE TRUE ${SP_TAG_MATMUL} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/tile.cpp"                "tile"              "task_tile"                 FALSE TRUE ${SP_TAG_TILE} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/gather.cpp"              "gather"            "task_gather"               FALSE TRUE ${SP_TAG_GATHER} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/relu.cpp"                "relu"              "task_relu"                 FALSE TRUE ${SP_TAG_RELU} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/concat.cpp"              "concat"            "task_concat"               FALSE TRUE ${SP_TAG_CONCAT} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducemax.cpp"           "reducemax"         "task_reducemax"            FALSE TRUE ${SP_TAG_REDUCEMAX} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp"           "transpose"         "task_transpose"            FALSE TRUE ${SP_TAG_TRANSPOSE} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/square.cpp"              "square"            "task_square"               FALSE TRUE ${SP_TAG_SQUARE} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum.cpp"           "reducesum"         "task_reducesum"            FALSE TRUE ${SP_TAG_REDUCESUM} "" )
#sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/sqrt.cpp"                "sqrt"              "task_sqrt"                 FALSE TRUE ${SP_TAG_SQRT} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverfloat.cpp"      "datamoverfloat"    "task_datamover_mod1_float" FALSE TRUE ${SP_TAG_DATAMOVER_FLOAT} "")
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverinteger.cpp"    "datamoverinteger"  "task_datamover_mod1_int"   FALSE TRUE ${SP_TAG_DATAMOVER_INT} "")
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/padding.cpp"             "padding"           "task_pad_last_dim"         FALSE TRUE ${SP_TAG_PADDING} "")
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/unpadding.cpp"           "unpadding"         "task_unpad_last_dim"       FALSE TRUE ${SP_TAG_UNPADDING} "")

#==========================================================================================================================================



# Generating bash script to compile the kernels to get *.xo files on a cmake custom target
add_custom_target(compile_hw COMMAND sh Compile_Hw_Batch.sh) 
add_custom_target(compile_hwemu COMMAND sh Compile_HwEmu_Batch.sh) 
add_custom_target(compile_swemu COMMAND sh Compile_SwEmu_Batch.sh) 

# Time to link *.xo files that have been generated from the kernels, to generate *.xclbin
file(WRITE "${CMAKE_BINARY_DIR}/Link_Hw_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -l -t hw --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin"
)

file(WRITE "${CMAKE_BINARY_DIR}/Link_HwEmu_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -l -t hw_emu --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hwemu_link_command} ${hwemu_link} -o ${CMAKE_PROJECT_NAME}_hw_emu.xclbin"
)

file(WRITE "${CMAKE_BINARY_DIR}/Link_SwEmu_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -l -t sw_emu --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${swemu_link_command} ${swemu_link} -o ${CMAKE_PROJECT_NAME}_sw_emu.xclbin"
)
add_custom_target(link_hw COMMAND sh Link_Hw_Batch.sh) 
add_custom_target(link_hwemu COMMAND sh Link_HwEmu_Batch.sh ) 
add_custom_target(link_swemu COMMAND sh Link_SwEmu_Batch.sh ) 


file(WRITE "${CMAKE_BINARY_DIR}/Interactrive_Hw_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -g -l -t hw --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin --interactive synth"
)
add_custom_target(interactive_hw COMMAND sh Interactrive_Hw_Batch.sh) 


#==========================================================================================================================================


set(HOSTSIDE_SRC
    #src/ClassifierMultiPlatform.cpp
    src/cnpy.cpp
    #src/MainExecutable.cpp
    #src/ModelArchTop05.cpp
    src/PlatformImplementation.cpp
    src/PlatformSelector.cpp
    src/TensorF.cpp
    src/TensorI.cpp
    src/WeightsLoader.cpp
    src/WorkScheduler.cpp
    src/cpu_imp/CpuImplementation.cpp
    src/ocl_imp/OclTensorF.cpp
    src/ocl_imp/OclTensorI.cpp
    src/ocl_imp/xilinx/xcl.cpp
    src/ocl_imp/xilinx/XilinxImplementation.cpp
    #src/ocl_imp/xilinx/XilinxImpUnitTests.cpp
)

set(HOSTSIDE_MAIN_SRC
    ${HOSTSIDE_SRC}
    src/ClassifierMultiPlatform.cpp
    src/MainExecutable.cpp
    src/ModelArchTop05.cpp
    src/ocl_imp/xilinx/XilinxImpUnitTests.cpp
)

add_executable(${HostExecutableName} ${HOSTSIDE_MAIN_SRC})
target_link_libraries(${HostExecutableName} ${SDAccel_LIBRARIES} ${SDAccel_FLOATING_POINT_LIBRARY} z)



set(xcl_path_swemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_sw_emu.xclbin")
set(xcl_path_hwemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw_emu.xclbin")
set(xcl_path_hw "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw.xclbin")
set(HostExecutablePath "${CMAKE_BINARY_DIR}/${HostExecutableName}")
set(DataDirectory "${CMAKE_SOURCE_DIR}/data")
configure_file(${CMAKE_SOURCE_DIR}/scripts/LaunchDeepPointV1FPGA.sh.in LaunchDeepPointV1FPGA.sh @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_swemu.sh ${CMAKE_BINARY_DIR}/autobuild_swemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hwemu.sh ${CMAKE_BINARY_DIR}/autobuild_hwemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hw.sh ${CMAKE_BINARY_DIR}/autobuild_hw.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PasteBinAgentForAutoBuilds.py ${CMAKE_BINARY_DIR}/PasteBinAgentForAutoBuilds.py @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/sdaccel.ini ${CMAKE_BINARY_DIR}/sdaccel.ini COPYONLY)
add_subdirectory("test")