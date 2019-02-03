# DeepPoint-V1 Project
This repository contains the code base for Xilinx SDAccel FPGA implementation of official DGCNN model.

## Platforms
Currenlty, only Xilinx FPGAs are supported.

Name | Supported Platform | Implementation
---  | ---                | ---
ModelArch01 | CPU                   | CPU 

## UnitTests for Platforms
Unlike parent project, this repository does not use UnitTest++ framework for managing unittests.
The reason is that declaring multiple executables in Eclipse SDAceel managed build environment is not straight forward. This repository has only one main executable(MainExecutable.cpp) that handles both classifier and unit tests while you can enable or disable each phase in the source code.

## Credit
Used repositories are listed below:
* [cnpy](https://github.com/rogersce/cnpy)
* [PointNet](https://github.com/charlesq34/pointnet)
* [PointNet++](https://github.com/charlesq34/pointnet2)

