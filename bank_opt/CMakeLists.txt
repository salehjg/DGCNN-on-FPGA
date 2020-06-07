find_package(Threads REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/bank_opt/inc
)

add_executable(MemBankOptimizer
    ${PROJECT_SOURCE_DIR}/bank_opt/src/MemBankOptimizer.cpp
    ${PROJECT_SOURCE_DIR}/bank_opt/src/Helper.cpp
    ${PROJECT_SOURCE_DIR}/bank_opt/src/FakeTensorF.cpp
    ${PROJECT_SOURCE_DIR}/bank_opt/src/FakeTensorI.cpp
    ${PROJECT_SOURCE_DIR}/bank_opt/src/Layers.cpp
    ${PROJECT_SOURCE_DIR}/bank_opt/src/FakeModelArchTop.cpp
)

target_link_libraries(MemBankOptimizer
    ${CMAKE_THREAD_LIBS_INIT} 
)