find_package(Threads REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/inc/ocl_imp/xilinx 
	${PROJECT_SOURCE_DIR}/hlslib/include 
	inc 
	${PROJECT_SOURCE_DIR}/test/common/inc)

add_executable(CpuTestReduceSum4D
    src/CpuTestReduceSum4D.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/reducesum4d.cpp)

target_link_libraries(CpuTestReduceSum4D
    ${SDAccel_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${Vitis_FLOATING_POINT_LIBRARY} 
    ${Vitis_LIBRARIES})

add_test(NAME CpuTestReduceSum4D COMMAND CpuTestReduceSum4D)