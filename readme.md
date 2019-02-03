# DeepPoint-V1 Project
This repository contains the code base for pure C++, OpenCL and CUDA
implementations of official DGCNN model.

## Platforms
Currenlty, only CPU, CUDA and OCL are supported. Each ModelArch class
supports an specific platform.

Name | Supported Platform | Implementation
---  | ---                | ---
ModelArch01 | CPU                   | CPU
ModelArch02 | CPU, CUDA             | CPU - CUDA Mix
ModelArch03 | CUDA                  | Pure CUDA **Without** Tensor Deletion
ModelArch04 | CUDA                  | Pure CUDA **With** Tensor Deletion
ModelArch05 | OCL                   | Pure OCL **With** Tensor Deletion

## UnitTests for Platforms
DeepPoint uses UnitTest++ framework for managing and running unit tests.
Currently, Platforms CUDA and OCL have their dedicated unit tests for
each kernel. The pupose of unit tests are to compare results of CPU platform
against the platform under test.

- CudaTestAll.cpp
- OclTestAll.cpp

## Credit
Used repositories are listed below:
* [cnpy](https://github.com/rogersce/cnpy)
* [PointNet](https://github.com/charlesq34/pointnet)
* [PointNet++](https://github.com/charlesq34/pointnet2)

