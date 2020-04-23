find_package(Threads REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/inc/ocl_imp/xilinx 
	${PROJECT_SOURCE_DIR}/hlslib/include 
	inc 
	${PROJECT_SOURCE_DIR}/test/common/inc)

add_executable(CpuTestTopkSelectionSort
    src/CpuTestTopk.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_selectionsort.cpp)

target_link_libraries(CpuTestTopkSelectionSort
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${Vitis_FLOATING_POINT_LIBRARY} 
    ${Vitis_LIBRARIES})

#add_test(NAME CpuTestTopkSelectionSort COMMAND CpuTestTopkSelectionSort)

add_executable(CpuTestTopkInsertionSort
    src/CpuTestTopk.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_insertionsort.cpp)

target_link_libraries(CpuTestTopkInsertionSort 
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${Vitis_FLOATING_POINT_LIBRARY} 
    ${Vitis_LIBRARIES})

#add_test(NAME CpuTestTopkInsertionSort COMMAND CpuTestTopkInsertionSort)

add_executable(CpuTestTopkMergeSort
    src/CpuTestTopk.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_mergesort.cpp)

target_link_libraries(CpuTestTopkMergeSort
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${Vitis_FLOATING_POINT_LIBRARY} 
    ${Vitis_LIBRARIES})

add_test(NAME CpuTestTopkMergeSort COMMAND CpuTestTopkMergeSort)