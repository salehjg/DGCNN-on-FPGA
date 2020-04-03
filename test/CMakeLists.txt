# To enable kernel stdout for all of the unittests below(add_subdirectory)
add_definitions("-DKERNEL_LOGS -DHLSLIB_XILINX")

add_subdirectory("conv2")
add_subdirectory("topk")
add_subdirectory("matops")
add_subdirectory("reducesum4d")
add_subdirectory("reducemax")
add_subdirectory("reducesum")
add_subdirectory("matmul")
add_subdirectory("padding")
add_subdirectory("unpadding")

