find_package(Threads REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/inc/ocl_imp/xilinx 
	${PROJECT_SOURCE_DIR}/hlslib/include 
	inc 
	${PROJECT_SOURCE_DIR}/test/common/inc)

#add_executable(CpuTestTopkSelectionSort
#    src/CpuTestTopk.cpp 
#    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_selectionsort.cpp)
#
#target_link_libraries(CpuTestTopkSelectionSort
#    ${SDAccel_LIBRARIES} 
#    ${CMAKE_THREAD_LIBS_INIT} 
#    ${SDAccel_FLOATING_POINT_LIBRARY} 
#    ${SDAccel_LIBRARIES})
#
#add_test(NAME CpuTestTopkSelectionSort COMMAND CpuTestTopkSelectionSort)

#add_executable(CpuTestTopkInsertionSort
#    src/CpuTestTopk.cpp 
#    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_insertionsort.cpp)
#
#target_link_libraries(CpuTestTopkInsertionSort 
#    ${SDAccel_LIBRARIES} 
#    ${CMAKE_THREAD_LIBS_INIT} 
#    ${SDAccel_FLOATING_POINT_LIBRARY} 
#    ${SDAccel_LIBRARIES})
#
#add_test(NAME CpuTestTopkInsertionSort COMMAND CpuTestTopkInsertionSort)

add_executable(CpuTestTopkMergeSort
    src/CpuTestTopk.cpp 
    ${PROJECT_SOURCE_DIR}/src/kernels/xilinx/topk_mergesortdf_pe.cpp)

target_link_libraries(CpuTestTopkMergeSort
    ${SDAccel_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${SDAccel_FLOATING_POINT_LIBRARY} 
    ${SDAccel_LIBRARIES})

add_test(NAME CpuTestTopkMergeSort COMMAND CpuTestTopkMergeSort)