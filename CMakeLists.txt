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
    message(FATAL_ERROR "You must set env variable AWS_PLATFORM to the path of the Vitis platform file(*.xpfm)")
endif()

#==========================================================================================================================================

#-DKERNEL_LOGS -DHLSLIB_SYNTHESIS

# $ENV{AWS_PLATFORM}   OR    xilinx_kcu1500_dynamic_5_0
set(BinaryContainersName "binarycontainer1" CACHE STRING "Binary Container's Name")
set(Platform "$ENV{AWS_PLATFORM}" CACHE STRING "Known Vitis platform name or xpfm file path") 
set(PartName "xcvu9p-flga2104-2-e" CACHE STRING "Part name for synthesis only.")
set(KernelFlags "-DKERNEL_LOGS -DHLSLIB_SYNTHESIS -DHLSLIB_XILINX -std=c++11 -I${CMAKE_SOURCE_DIR}/inc/ocl_imp/xilinx -I${CMAKE_SOURCE_DIR}/config/output -I${CMAKE_SOURCE_DIR}/hlslib/include" CACHE STRING "CFlags for the kernels")
set(Profiling ON CACHE BOOL "Profiling ON/OFF")
set(OptimizationLevel "0" CACHE STRING "0:default, 1:reduce power, 2:increase kernel speed, 3:highest level of optimization, s:optimize for size, quick:quick compilation")
set(KernelFrequency "200" CACHE STRING "MHz")
set(IpCachePath "${CMAKE_BINARY_DIR}/ip_cache" CACHE STRING "IP Cache Path(with enough space for a hw build)")

set(PASTEBIN_0_1 "0" CACHE STRING "Enable or disable PasteBin agent for uploading autobuild log files to PasteBin.") 
set(PASTEBIN_USERNAME "" CACHE STRING "PasteBin Account's Username") 
set(PASTEBIN_PASSWORD "" CACHE STRING "PasteBin Account's Password") 
set(PASTEBIN_API_KEY  "" CACHE STRING "PasteBin Account's API Key") 

set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_SOURCE_DIR}/hlslib/cmake
    )

find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)
include(CheckTypeSize)


#==========================================================================================================================================

if(((${Vitis_MAJOR_VERSION} LESS 2018) AND (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
  message(FATAL_ERROR "Legacy SDx is not supported on this branch.")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHLSLIB_LEGACY_SDX=0")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/config)
include_directories(config/output inc hlslib/include ${CMAKE_BINARY_DIR} ${Vitis_INCLUDE_DIRS})
add_definitions(-DREPO_DIR=\"${CMAKE_SOURCE_DIR}\")
add_definitions(-DBIN_DIR=\"${CMAKE_BINARY_DIR}\")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(hw_link_command "")
set(hwemu_link_command "")
set(swemu_link_command "")
set(hw_link "")
set(hwemu_link "")
set(swemu_link "")

#Creating IpCacheFolder
file(MAKE_DIRECTORY ${IpCachePath})

# Create common-config.ini
file(WRITE "${CMAKE_BINARY_DIR}/common-config.ini" "platform=${Platform}\nsave-temps=1\n")

file(WRITE "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")

include(ProcessorCount)
ProcessorCount(_CpuCount)
if(NOT ${_CpuCount} EQUAL 0)
    set(CpuCount ${_CpuCount})
else()
    set(CpuCount 4)
endif()
message("Detected CPU Core Count: ${CpuCount}")

if(${Profiling})
    message("Profiling is enabled.")
    set(true_false_profiling "true")
else()
    message("Profiling is disabled.")
    set(true_false_profiling "false")
endif()

configure_file(${CMAKE_SOURCE_DIR}/scripts/xrt.ini.in ${CMAKE_BINARY_DIR}/xrt.ini)

set(SlrUtilizationReportFile ${CMAKE_BINARY_DIR}/${BinaryContainersName}.build/reports/SlrUtilizationReportOutput.txt)
configure_file(${CMAKE_SOURCE_DIR}/scripts/SlrUtilizationReport.tcl.in ${CMAKE_BINARY_DIR}/SlrUtilizationReport.tcl)

#==========================================================================================================================================


function(CreateLinkIni
    nkTagsForKernels)
    # Create <binary container>-link.ini
    file(WRITE "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "debug=1\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "messageDb=${CMAKE_BINARY_DIR}/${BinaryContainersName}.mdb\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "temp_dir=${CMAKE_BINARY_DIR}/${BinaryContainersName}.build\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "report_dir=${CMAKE_BINARY_DIR}/${BinaryContainersName}.build/reports\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "log_dir=${CMAKE_BINARY_DIR}/${BinaryContainersName}.build/logs\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "remote_ip_cache=${IpCachePath}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "[advanced]\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "misc=solution_name=link\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "[connectivity]\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini" "${${nkTagsForKernels}}\n")
endfunction()

# Generates and appends commands for compiling and linking the given kernel.
function(CreateKernel 
    ContainersName
    Path
    TopFunctionsName  
    IsProfiled
    ProfileData
    ProfileStall
    ProfileExec
    SpTag
    OutNkTagsAppended)

    # Creating compile config file for this kernel
    file(WRITE "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "kernel=${TopFunctionsName}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "debug=1\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "messageDb=${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}.mdb\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "temp_dir=${CMAKE_BINARY_DIR}/${ContainersName}.build\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "report_dir=${CMAKE_BINARY_DIR}/${ContainersName}.build/reports\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "log_dir=${CMAKE_BINARY_DIR}/${ContainersName}.build/logs\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "[advanced]\n")
    file(APPEND "${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini" "misc=solution_name=${TopFunctionsName}\n")
    set(NkTag "nk=${TopFunctionsName}:1:${TopFunctionsName}_1")
    set(${OutNkTagsAppended} "${${OutNkTagsAppended}}\n${NkTag}" PARENT_SCOPE)

    set(ProfilingOptionsCompile "")
    set(ProfilingOptionsLink "")

    if(${IsProfiled})
        if(${ProfileData})
            set(ProfilingOptionsLink "${ProfilingOptionsLink} --profile_kernel data:${TopFunctionsName}:all:all:all")
        endif()
        if(${ProfileStall})
            set(ProfilingOptionsCompile "${ProfilingOptionsCompile} --profile_kernel stall:${TopFunctionsName}:all:all")
            set(ProfilingOptionsLink "${ProfilingOptionsLink} --profile_kernel stall:${TopFunctionsName}:all:all")
        endif()
        if(${ProfileExec})
            set(ProfilingOptionsLink "${ProfilingOptionsLink} --profile_kernel exec:${TopFunctionsName}:all:all")
        endif()
    endif()


    set(hw_compile 
        ${Vitis_COMPILER} 
        --target hw
        --compile
        --debug
        --jobs ${CpuCount}
        --kernel_frequency ${KernelFrequency}
        --optimize ${OptimizationLevel}
        --config ${CMAKE_BINARY_DIR}/common-config.ini 
        --config ${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini
        --advanced.prop kernel.${TopFunctionsName}.kernel_flags=\"${KernelFlags}\"
        -o\"${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_hw.xo\" 
        \"${Path}\"
    )

    set(hwemu_compile 
        ${Vitis_COMPILER} 
        --target hw_emu
        --compile
        --debug
        --jobs ${CpuCount}
        --kernel_frequency ${KernelFrequency}
        --optimize ${OptimizationLevel}
        --config ${CMAKE_BINARY_DIR}/common-config.ini 
        --config ${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini
        --advanced.prop kernel.${TopFunctionsName}.kernel_flags=\"${KernelFlags}\"
        -o\"${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_hw_emu.xo\" 
        \"${Path}\"
    )

    set(swemu_compile 
        ${Vitis_COMPILER} 
        --target sw_emu
        --compile
        --debug
        --jobs ${CpuCount}
        --kernel_frequency ${KernelFrequency}
        --optimize ${OptimizationLevel}
        --config ${CMAKE_BINARY_DIR}/common-config.ini 
        --config ${CMAKE_BINARY_DIR}/${ContainersName}-${TopFunctionsName}-compile.ini
        --advanced.prop kernel.${TopFunctionsName}.kernel_flags=\"${KernelFlags}\"
        -o\"${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_sw_emu.xo\" 
        \"${Path}\"
    )

    string (REPLACE ";" " " hw_compile "${hw_compile}")
    string (REPLACE ";" " " hwemu_compile "${hwemu_compile}")
    string (REPLACE ";" " " swemu_compile "${swemu_compile}")

    file(APPEND "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "${hw_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "${hwemu_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "${swemu_compile}\n")

    set(hw_link "${hw_link} ${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_hw.xo" PARENT_SCOPE) 
    set(hwemu_link "${hwemu_link} ${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_hw_emu.xo" PARENT_SCOPE) 
    set(swemu_link "${swemu_link} ${CMAKE_BINARY_DIR}/${ContainersName}.build/${TopFunctionsName}_sw_emu.xo" PARENT_SCOPE) 

    if(${IsProfiled} AND (${ProfileData} OR ${ProfileStall} OR ${ProfileExec}))
        set(hw_link_command "${hw_link_command} ${SpTag} ${ProfilingOptionsLink}")
    else()
        set(hw_link_command "${hw_link_command} ${SpTag}")
    endif()
    

    # Identical commands for hw-emu link process
    set(hwemu_link_command "${hw_link_command}")
    set(swemu_link_command "${hw_link_command}")

    # Set link commands globally
    set(hw_link_command "${hw_link_command}" PARENT_SCOPE)
    set(hwemu_link_command "${hwemu_link_command}" PARENT_SCOPE)
    set(swemu_link_command "${swemu_link_command}" PARENT_SCOPE)

endfunction()


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

set(SP_TAG_CONCAT
"--sp task_concat_1.inputTn1:bank${CFG8_Concat_DDRBANK_inputTn1}\
 --sp task_concat_1.inputTn2:bank${CFG8_Concat_DDRBANK_inputTn2}\
 --sp task_concat_1.outputTn:bank${CFG8_Concat_DDRBANK_outputTn}")

set(SP_TAG_REDUCEMAX        
"--sp task_reducemax_1.inputTn:bank${CFG9_ReduceMax_DDRBANK_inputTn}\
 --sp task_reducemax_1.outputTn:bank${CFG9_ReduceMax_DDRBANK_outputTn}")

set(SP_TAG_TRANSPOSE        
"--sp task_transpose_1.inputTn:bank${CFG10_Transpose_DDRBANK_inputTn}\
 --sp task_transpose_1.outputTn:bank${CFG10_Transpose_DDRBANK_outputTn}")

set(SP_TAG_REDUCESUM        
"--sp task_reducesum_1.inputTn:bank${CFG11_ReduceSum_DDRBANK_inputTn}\
 --sp task_reducesum_1.outputTn:bank${CFG11_ReduceSum_DDRBANK_outputTn}")

set(SP_TAG_RELUSQRTSQUARE             
"--sp task_relu_sqrt_square_1.inputTn:bank${CFG12_ReluSqrtSquare_DDRBANK_inputTn}\
 --sp task_relu_sqrt_square_1.outputTn:bank${CFG12_ReluSqrtSquare_DDRBANK_outputTn}")

set(SP_TAG_DATAMOVER   
"--sp task_datamover_1.dataBank0:bank0\
 --sp task_datamover_1.dataBank1:bank1\
 --sp task_datamover_1.dataBank2:bank2\
 --sp task_datamover_1.dataBank3:bank3")

set(SP_TAG_PADDING          
"--sp task_pad_last_dim_1.inputTn:bank${CFG13_Padding_DDRBANK_inputTn}\
 --sp task_pad_last_dim_1.outputTn:bank${CFG13_Padding_DDRBANK_outputTn}")

set(SP_TAG_UNPADDING        
"--sp task_unpad_last_dim_1.inputTn:bank${CFG14_Unpadding_DDRBANK_inputTn}\
 --sp task_unpad_last_dim_1.outputTn:bank${CFG14_Unpadding_DDRBANK_outputTn}")



set(GlobalNkTags "")

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/conv2_1x1_direct.cpp"
    "task_conv2_1x1_direct"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_CONV2D}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/topk_mergesort.cpp"
    "task_topk"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_TOPK}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matops.cpp"
    "task_matops"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_MATOPS}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp"
    "task_reducesum4d"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_REDUCESUM4D}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/matmul.cpp"
    "task_matmul"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_MATMUL}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/tile.cpp"
    "task_tile"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_TILE}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/gather.cpp"
    "task_gather"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_GATHER}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/concat.cpp"
    "task_concat"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_CONCAT}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducemax.cpp"
    "task_reducemax"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_REDUCEMAX}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp"
    "task_transpose"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_TRANSPOSE}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/reducesum.cpp"
    "task_reducesum"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_REDUCESUM}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/relu_sqrt_square.cpp"
    "task_relu_sqrt_square"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_RELUSQRTSQUARE}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/datamover.cpp"
    "task_datamover"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_DATAMOVER}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/padding.cpp"
    "task_pad_last_dim"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_PADDING}"
    GlobalNkTags)

CreateKernel(
    "${BinaryContainersName}"
    "${CMAKE_SOURCE_DIR}/src/kernels/xilinx/unpadding.cpp"
    "task_unpad_last_dim"
    ${Profiling}
    FALSE # Data Profiling
    FALSE # Stall Profiling
    TRUE # Exec Profiling
    "${SP_TAG_UNPADDING}"
    GlobalNkTags)

CreateLinkIni(GlobalNkTags)


#==========================================================================================================================================


add_custom_target(compile_hw COMMAND sh Compile_Hw_Batch.sh) 
add_custom_target(compile_hwemu COMMAND sh Compile_HwEmu_Batch.sh) 
add_custom_target(compile_swemu COMMAND sh Compile_SwEmu_Batch.sh) 


set( _link_hw_
    ${Vitis_COMPILER}
    --target hw --link
    --debug
    --optimize ${OptimizationLevel}
    --jobs ${CpuCount}
    --kernel_frequency ${KernelFrequency}
    --config ${CMAKE_BINARY_DIR}/common-config.ini
    --config ${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini
    --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.TCL.PRE=${CMAKE_BINARY_DIR}/SlrUtilizationReport.tcl
    -o\"${CMAKE_BINARY_DIR}/${BinaryContainersName}_hw.xclbin\"
    ${hw_link_command}
    ${hw_link}
    )
string (REPLACE ";" " " _link_hw_ "${_link_hw_}")
file(WRITE "${CMAKE_BINARY_DIR}/Link_Hw_Batch.sh" "${_link_hw_}")

set( _link_hwemu_
    ${Vitis_COMPILER}
    --target hw_emu --link
    --debug
    --optimize ${OptimizationLevel}
    --jobs ${CpuCount}
    --kernel_frequency ${KernelFrequency}
    --config ${CMAKE_BINARY_DIR}/common-config.ini
    --config ${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini
    -o\"${CMAKE_BINARY_DIR}/${BinaryContainersName}_hw_emu.xclbin\"
    ${hwemu_link_command}
    ${hwemu_link}
    )
string (REPLACE ";" " " _link_hwemu_ "${_link_hwemu_}")
file(WRITE "${CMAKE_BINARY_DIR}/Link_HwEmu_Batch.sh" "${_link_hwemu_}")

set( _link_swemu_
    ${Vitis_COMPILER}
    --target sw_emu --link
    --debug
    --optimize ${OptimizationLevel}
    --jobs ${CpuCount}
    --kernel_frequency ${KernelFrequency}
    --config ${CMAKE_BINARY_DIR}/common-config.ini
    --config ${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini
    -o\"${CMAKE_BINARY_DIR}/${BinaryContainersName}_sw_emu.xclbin\"
    ${swemu_link_command}
    ${swemu_link}
    )
string (REPLACE ";" " " _link_swemu_ "${_link_swemu_}")
file(WRITE "${CMAKE_BINARY_DIR}/Link_SwEmu_Batch.sh" "${_link_swemu_}")

set( _link_interactive_imp_
    ${Vitis_COMPILER}
    --target hw --link
    --debug
    --optimize ${OptimizationLevel}
    --jobs ${CpuCount}
    --kernel_frequency ${KernelFrequency}
    --config ${CMAKE_BINARY_DIR}/common-config.ini
    --config ${CMAKE_BINARY_DIR}/${BinaryContainersName}-link.ini
    -o\"${CMAKE_BINARY_DIR}/${BinaryContainersName}_hw.xclbin\"
    ${hw_link_command}
    ${hw_link}
    --interactive impl
    )
string (REPLACE ";" " " _link_interactive_imp_ "${_link_interactive_imp_}")
file(WRITE "${CMAKE_BINARY_DIR}/Interactrive_Hw_Batch.sh" "${_link_interactive_imp_}")

add_custom_target(link_hw COMMAND sh Link_Hw_Batch.sh) 
add_custom_target(link_hwemu COMMAND sh Link_HwEmu_Batch.sh ) 
add_custom_target(link_swemu COMMAND sh Link_SwEmu_Batch.sh ) 
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
    src/ocl_imp/DataMover.cpp
    src/ocl_imp/xilinx/xcl.cpp
    src/ocl_imp/xilinx/XilinxImplementation.cpp
    #src/ocl_imp/xilinx/XilinxImpUnitTests.cpp
)

set(HOSTSIDE_MAIN_SRC
    ${HOSTSIDE_SRC}
    src/ClassifierMultiPlatform.cpp
    src/MainExecutable.cpp
    src/ModelArchTop04.cpp
    src/ocl_imp/xilinx/XilinxImpUnitTests.cpp
)

add_executable(${HostExecutableName} ${HOSTSIDE_MAIN_SRC})
target_link_libraries(${HostExecutableName} ${Vitis_LIBRARIES} ${Vitis_FLOATING_POINT_LIBRARY} z)



set(xcl_path_swemu "${CMAKE_BINARY_DIR}/${BinaryContainersName}_sw_emu.xclbin")
set(xcl_path_hwemu "${CMAKE_BINARY_DIR}/${BinaryContainersName}_hw_emu.xclbin")
set(xcl_path_hw "${CMAKE_BINARY_DIR}/${BinaryContainersName}_hw.xclbin")
set(HostExecutablePath "${CMAKE_BINARY_DIR}/${HostExecutableName}")
set(DataDirectory "${CMAKE_SOURCE_DIR}/data")
configure_file(${CMAKE_SOURCE_DIR}/scripts/LaunchDeepPointV1FPGA.sh.in LaunchDeepPointV1FPGA.sh @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_swemu.sh ${CMAKE_BINARY_DIR}/autobuild_swemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hwemu.sh ${CMAKE_BINARY_DIR}/autobuild_hwemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hw.sh ${CMAKE_BINARY_DIR}/autobuild_hw.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PasteBinAgentForAutoBuilds.py ${CMAKE_BINARY_DIR}/PasteBinAgentForAutoBuilds.py @ONLY)

include(CTest)
add_subdirectory("test")