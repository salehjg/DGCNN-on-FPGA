find_package(Threads REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/inc/ocl_imp/xilinx 
	${PROJECT_SOURCE_DIR}/hlslib/include 
	inc 
	${PROJECT_SOURCE_DIR}/test/common/inc)

add_executable(CpuTestTranspose
    src/CpuTestTranspose.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/transpose.cpp)

target_link_libraries(CpuTestTranspose
    ${SDAccel_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${SDAccel_FLOATING_POINT_LIBRARY} 
    ${SDAccel_LIBRARIES})

add_test(NAME CpuTestTranspose COMMAND CpuTestTranspose)