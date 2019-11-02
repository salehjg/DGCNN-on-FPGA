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


#==========================================================================================================================================



# $ENV{AWS_PLATFORM}   OR    xilinx_kcu1500_dynamic_5_0
set(DSA_NAME "$ENV{AWS_PLATFORM}" CACHE STRING "Known SDAccel platform name or xpfm file path") 
set(ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information")
set(ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design")
set(OPTIMIZATION_OPTION "3" CACHE STRING "0:default, 1:reduce power, 2:increase kernel speed, 3:highest level of optimization, s:optimize for size, quick:quick compilation")
set(XOCC_FLAGS "" CACHE STRING "Extra flags for xocc")
set(KERNEL_CLOCK "110" CACHE STRING "MHz, For XOCC Compile and Link Procedures. All of the kernels will use this clock frequency. Set to -1 to ignore frequency override.")
set(KERNEL_CLOCK_UNCERTAINTY "0.9" CACHE STRING "Clock Uncertainty")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)



#==========================================================================================================================================



include_directories(inc hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions("-DHLSLIB_STREAM_TIMEOUT=16")
if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
  add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
  add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()



#==========================================================================================================================================



SET(hw_link_command "")
SET(hwemu_link_command "")
SET(swemu_link_command "")
SET(hw_link "")
SET(hwemu_link "")
SET(swemu_link "")
set(SYNTHESIS_FLAGS "-DHLSLIB_SYNTHESIS -DHLSLIB_XILINX -std=c++11 -O3")



#==========================================================================================================================================



function(floatexpr expr output)
    SET(FLOAT_EXPR_SRC "
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

    set(HLSLIB_PROJECT_NAME "${KERNEL_NAME}")
    set(HLSLIB_SRC_SYNTHESIS "${KERNEL_FILE}")
    set(HLSLIB_ENTRY_FUNCTION "${KERNEL_NAME}")
    set(HLSLIB_TARGET_CLOCK "${KERNEL_CLOCK}")
    set(HLSLIB_TARGET_CLOCK_UNCERTAINTY "${KERNEL_CLOCK_UNCERTAINTY}")
    set(HLSLIB_SYNTHESIS_FLAGS ${SYNTHESIS_FLAGS})
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
    set(HLSLIB_SYNTHESIS_FLAGS "${SYNTHESIS_FLAGS} ${ADDITIONAL_FLAGS}")
    #string(REPLACE " " ";" XOCC_FLAGS "${XOCC_FLAGS}")
    set(SYNTHESIS_FLAGS "") #should be ';' seperated
    set(XOCC_COMMAND
        -s
        ${XOCC_FLAGS}
        -I${CMAKE_SOURCE_DIR}/inc
        -I${CMAKE_SOURCE_DIR}/hlslib/include
        -I${CMAKE_BINARY_DIR}
        --kernel ${KERNEL_NAME} 
        --platform ${DSA_NAME}
        --xp prop:kernel.${KERNEL_NAME}.kernel_flags="${SYNTHESIS_FLAGS}"
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

    set(hw_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c -t hw ${XOCC_COMMAND} -R2 ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw.xo")
    set(hwemu_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c -t hw_emu ${XOCC_COMMAND} -Restimate ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw_emu.xo") 
    set(swemu_compile "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -c -t sw_emu ${XOCC_COMMAND} ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_sw_emu.xo") 
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
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/concat.cpp"             "task_concat"              )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/conv2_1x1_direct.cpp"   "task_conv2_1x1_direct"    )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverfloat.cpp"     "task_datamover_mod1_float")
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverinteger.cpp"   "task_datamover_mod1_int"  )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/gather.cpp"             "task_gather"              )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matmul.cpp"             "task_matmul"              )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matops.cpp"             "task_matops"              )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducemax.cpp"          "task_reducemax"           )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum.cpp"          "task_reducesum"           )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp"        "task_reducesum4d"         )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/relu.cpp"               "task_relu"                )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/sqrt.cpp"               "task_sqrt"                )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/square.cpp"             "task_square"              )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/tile.cpp"               "task_tile"                )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk.cpp"               "task_topk"                )
synthesis_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp"          "task_transpose"           )








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
endif()





file(WRITE "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "")





#==========================================================================================================================================

if(
    ((${SDAccel_MAJOR_VERSION} EQUAL 2017) AND (${SDAccel_MINOR_VERSION} EQUAL 4)) OR
    ((${SDAccel_MAJOR_VERSION} EQUAL 2018) AND (${SDAccel_MINOR_VERSION} EQUAL 2)) 
)
    # V2018.2 and before: old sp tag style
    message(FATAL_ERROR "OLD --SP STYLE IS NO LONGER SUPPORTED")
    #SET(SP_TAG_DATAMOVER_FLOAT  "--sp task_datamover_mod1_float_1.m_axi_gmem0:bank1 --sp task_datamover_mod1_float_1.m_axi_gmem1:bank2")
    #SET(SP_TAG_DATAMOVER_INT    "--sp task_datamover_mod1_int_1.m_axi_gmemi0:bank1 --sp task_datamover_mod1_int_1.m_axi_gmemi1:bank2")
    #SET(SP_TAG_CONV2D           "--sp task_conv2_1x1_direct_1.m_axi_gmem1:bank2 --sp task_conv2_1x1_direct_1.m_axi_gmem2:bank2")
    #SET(SP_TAG_TILE             "--sp task_tile_1.m_axi_gmem1:bank2")
    #SET(SP_TAG_CONCAT           "--sp task_concat_1.m_axi_gmem1:bank2 --sp task_concat_1.m_axi_gmem2:bank2")
    #SET(SP_TAG_GATHER           "--sp task_gather_1.m_axi_gmem1:bank2 --sp task_gather_1.m_axi_gmem2:bank2")
    #SET(SP_TAG_MATMUL           "--sp task_matmul_1.m_axi_gmem1:bank2 --sp task_matmul_1.m_axi_gmem2:bank2")
    #SET(SP_TAG_MATOPS           "--sp task_matops_1.m_axi_gmem1:bank2 --sp task_matops_1.m_axi_gmem2:bank2")
    #SET(SP_TAG_REDUCEMAX        "--sp task_reducemax_1.m_axi_gmem1:bank1 --sp task_reducemax_1.m_axi_gmem2:bank1")
    #SET(SP_TAG_REDUCESUM        "--sp task_reducesum_1.m_axi_gmem1:bank1 --sp task_reducesum_1.m_axi_gmem2:bank1")
    #SET(SP_TAG_REDUCESUM4D      "--sp task_reducesum4d_1.m_axi_gmem1:bank1 --sp task_reducesum4d_1.m_axi_gmem2:bank1")
    #SET(SP_TAG_RELU             "--sp task_relu_1.m_axi_gmem1:bank1")
    #SET(SP_TAG_SQRT             "--sp task_sqrt_1.m_axi_gmem1:bank1")
    #SET(SP_TAG_SQUARE           "--sp task_square_1.m_axi_gmem1:bank1")
    #SET(SP_TAG_TOPK             "--sp task_topk_1.m_axi_gmem1:bank1 --sp task_topk_1.m_axi_gmem2:bank1")
    #SET(SP_TAG_TRANSPOSE        "--sp task_transpose_1.m_axi_gmem1:bank1")
else()
    # V2018.3 and on: new sp tag style
    SET(SP_TAG_DATAMOVER_FLOAT  "--sp task_datamover_mod1_float_1.srcBuff:DDR[1] --sp task_datamover_mod1_float_1.dstBuff:DDR[2]")
    SET(SP_TAG_DATAMOVER_INT    "--sp task_datamover_mod1_int_1.srcBuff:DDR[1] --sp task_datamover_mod1_int_1.dstBuff:DDR[2]")

    SET(SP_TAG_CONV2D           "--sp task_conv2_1x1_direct_1.inputTn:DDR[2] --sp task_conv2_1x1_direct_1.weightTn:DDR[2] --sp task_conv2_1x1_direct_1.biasTn:DDR[2] --sp task_conv2_1x1_direct_1.outputTn:DDR[2]")
    SET(SP_TAG_TILE             "--sp task_tile_1.inputTn:DDR[2] --sp task_tile_1.outputTn:DDR[2]")
    SET(SP_TAG_CONCAT           "--sp task_concat_1.inputTn1:DDR[2] --sp task_concat_1.inputTn2:DDR[2] --sp task_concat_1.outputTn:DDR[2]")
    SET(SP_TAG_GATHER           "--sp task_gather_1.inputTn:DDR[2] --sp task_gather_1.indicesTn:DDR[2] --sp task_gather_1.outputTn:DDR[2]")
    SET(SP_TAG_MATMUL           "--sp task_matmul_1.inputTn1:DDR[2] --sp task_matmul_1.inputTn2:DDR[2] --sp task_matmul_1.outputTn:DDR[2]")
    SET(SP_TAG_MATOPS           "--sp task_matops_1.inputTn1:DDR[2] --sp task_matops_1.inputTn2:DDR[2] --sp task_matops_1.outputTn:DDR[2]")

    SET(SP_TAG_REDUCEMAX        "--sp task_reducemax_1.inputTn:DDR[1] --sp task_reducemax_1.outputTn:DDR[1]")
    SET(SP_TAG_REDUCESUM        "--sp task_reducesum_1.inputTn:DDR[1] --sp task_reducesum_1.outputTn:DDR[1]")
    SET(SP_TAG_REDUCESUM4D      "--sp task_reducesum4d_1.inputTn:DDR[1] --sp task_reducesum4d_1.outputTn:DDR[1]")
    SET(SP_TAG_RELU             "--sp task_relu_1.inputTn:DDR[1] --sp task_relu_1.outputTn:DDR[1]")
    SET(SP_TAG_SQRT             "--sp task_sqrt_1.inputTn:DDR[1] --sp task_sqrt_1.outputTn:DDR[1]")
    SET(SP_TAG_SQUARE           "--sp task_square_1.inputTn:DDR[1] --sp task_square_1.outputTn:DDR[1]")
    SET(SP_TAG_TOPK             "--sp task_topk_1.inputTn:DDR[1] --sp task_topk_1.indicesSplitedTn:DDR[1]")
    SET(SP_TAG_TRANSPOSE        "--sp task_transpose_1.inputTn:DDR[1] --sp task_transpose_1.outputTn:DDR[1]")
endif()






sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/concat.cpp"              "concat"            "task_concat"               FALSE TRUE ${SP_TAG_CONCAT} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/conv2_1x1_direct.cpp"    "conv2_1x1_direct"  "task_conv2_1x1_direct"     FALSE TRUE ${SP_TAG_CONV2D} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverfloat.cpp"      "datamoverfloat"    "task_datamover_mod1_float" FALSE TRUE ${SP_TAG_DATAMOVER_FLOAT} "")
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamoverinteger.cpp"    "datamoverinteger"  "task_datamover_mod1_int"   FALSE TRUE ${SP_TAG_DATAMOVER_INT} "")
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/gather.cpp"              "gather"            "task_gather"               FALSE TRUE ${SP_TAG_GATHER} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matmul.cpp"              "matmul"            "task_matmul"               FALSE TRUE ${SP_TAG_MATMUL} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matops.cpp"              "matops"            "task_matops"               FALSE TRUE ${SP_TAG_MATOPS} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducemax.cpp"           "reducemax"         "task_reducemax"            FALSE TRUE ${SP_TAG_REDUCEMAX} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum.cpp"           "reducesum"         "task_reducesum"            FALSE TRUE ${SP_TAG_REDUCESUM} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp"         "reducesum4d"       "task_reducesum4d"          FALSE TRUE ${SP_TAG_REDUCESUM4D} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/relu.cpp"                "relu"              "task_relu"                 FALSE TRUE ${SP_TAG_RELU} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/sqrt.cpp"                "sqrt"              "task_sqrt"                 FALSE TRUE ${SP_TAG_SQRT} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/square.cpp"              "square"            "task_square"               FALSE TRUE ${SP_TAG_SQUARE} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/tile.cpp"                "tile"              "task_tile"                 FALSE TRUE ${SP_TAG_TILE} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk.cpp"                "topk"              "task_topk"                 FALSE TRUE ${SP_TAG_TOPK} "" )
sdaccel_target("${CMAKE_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp"           "transpose"         "task_transpose"            FALSE TRUE ${SP_TAG_TRANSPOSE} "" )




#==========================================================================================================================================



# Generating bash script to compile the kernels to get *.xo files on a cmake custom target

#foreach(i ${hw_compile})
#    file(APPEND "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "${i}\n")
#endforeach(i)


#foreach(i ${hwemu_compile})
#    file(APPEND "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "${i}\n")
#endforeach(i)


#foreach(i ${swemu_compile})
#    file(APPEND "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "${i}\n")
#endforeach(i)

add_custom_target(compile_hw COMMAND sh Compile_Hw_Batch.sh) 
add_custom_target(compile_hwemu COMMAND sh Compile_HwEmu_Batch.sh) 
add_custom_target(compile_swemu COMMAND sh Compile_SwEmu_Batch.sh) 

# Time to link *.xo files that have been generated from the kernels, to generate *.xclbin
file(WRITE "${CMAKE_BINARY_DIR}/Link_Hw_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t hw --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin"
)

file(WRITE "${CMAKE_BINARY_DIR}/Link_HwEmu_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t hw_emu --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hwemu_link_command} ${hwemu_link} -o ${CMAKE_PROJECT_NAME}_hw_emu.xclbin"
)

file(WRITE "${CMAKE_BINARY_DIR}/Link_SwEmu_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t sw_emu --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${swemu_link_command} ${swemu_link} -o ${CMAKE_PROJECT_NAME}_sw_emu.xclbin"
)
add_custom_target(link_hw COMMAND sh Link_Hw_Batch.sh) 
add_custom_target(link_hwemu COMMAND sh Link_HwEmu_Batch.sh ) 
add_custom_target(link_swemu COMMAND sh Link_SwEmu_Batch.sh ) 


file(WRITE "${CMAKE_BINARY_DIR}/Interactrive_Hw_Batch.sh" 
    "XILINX_PATH=${CMAKE_BINARY_DIR} ${SDAccel_XOCC} -l -t hw --platform ${DSA_NAME} --optimize ${OPTIMIZATION_OPTION} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin --interactive synth"
)
add_custom_target(interactive_hw COMMAND sh Interactrive_Hw_Batch.sh) 



#==========================================================================================================================================




# Host code to launch kernel
set(HOSTSIDE_SRC
    src/ClassifierMultiPlatform.cpp
    src/cnpy.cpp
    src/MainExecutable.cpp
    src/ModelArchTop05.cpp
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
    src/ocl_imp/xilinx/XilinxImpUnitTests.cpp
)

add_executable(${HostExecutableName} ${HOSTSIDE_SRC})
target_link_libraries(${HostExecutableName} ${SDAccel_LIBRARIES} ${SDAccel_FLOATING_POINT_LIBRARY} z)

set(xcl_path_swemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_sw_emu.xclbin")
set(xcl_path_hwemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw_emu.xclbin")
set(xcl_path_hw "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw.xclbin")
set(HostExecutablePath "${CMAKE_BINARY_DIR}/${HostExecutableName}")
set(DataDirectory "${CMAKE_SOURCE_DIR}/data")
configure_file(scripts/LaunchDeepPointV1FPGA.sh.in LaunchDeepPointV1FPGA.sh @ONLY)



#-t hw --platform xilinx_kcu1500_dynamic_5_0 --save-temps -Os --report system -l --nk task_conv2_1x1_direct:1 --profile_kernel data:task_conv2_1x1_direct:all:all --nk task_matmul:1 --nk task_matops:1 --nk task_reducesum4d:1 --nk task_relu:1 --nk task_sqrt:1 --nk task_tile:1 --nk task_topk:1 --profile_kernel data:task_topk:all:all --nk task_transpose:1 --nk task_datamover_mod1_float:1 --nk task_datamover_mod1_int:1 --nk task_reducemax:1 --messageDb binary_container_1.mdb --xp misc:solution_name=_xocc_link_binary_container_1 --sp task_datamover_mod1_float_1.m_axi_gmem0:bank1 --sp task_datamover_mod1_float_1.m_axi_gmem1:bank2 --sp task_datamover_mod1_int_1.m_axi_gmemi0:bank1 --sp task_datamover_mod1_int_1.m_axi_gmemi1:bank2 --sp task_tile_1.m_axi_gmem1:bank2 --sp task_conv2_1x1_direct_1.m_axi_gmem1:bank2 --sp task_conv2_1x1_direct_1.m_axi_gmem2:bank2 --remote_ip_cache /home/saleh/02_workspace/00_SDx2017.4/ip_cache -obinary_container_1.xclbin binary_container_1/task_conv2_1x1_direct.xo binary_container_1/task_matmul.xo binary_container_1/task_matops.xo binary_container_1/task_reducesum4d.xo binary_container_1/task_relu.xo binary_container_1/task_sqrt.xo binary_container_1/task_tile.xo binary_container_1/task_topk.xo binary_container_1/task_transpose.xo binary_container_1/task_datamover_mod1_float.xo binary_container_1/task_datamover_mod1_int.xo binary_container_1/task_reducemax.xo
# --interactive synth
