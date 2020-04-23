find_package(Threads REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/inc/ocl_imp/xilinx 
	${PROJECT_SOURCE_DIR}/hlslib/include 
	inc 
	${PROJECT_SOURCE_DIR}/test/common/inc)

add_executable(CpuTestConcat
    src/CpuTestConcat.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/concat.cpp)

target_link_libraries(CpuTestConcat
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${Vitis_FLOATING_POINT_LIBRARY} 
    ${Vitis_LIBRARIES})

add_test(NAME CpuTestConcat COMMAND CpuTestConcat)