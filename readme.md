# DeepPoint-V1 Project
This repository contains the code base for Xilinx SDAccel FPGA implementation of official DGCNN model.

# Build System
As easy as it is to use SDx GUI, it is recommended to use provided cmake scripts to run synthesis and build the binaries for both FPGA and the host.

# How to...
## 1. Build Host
```
mkdir build
cd build
cmake ..
make DeepPointV1FPGA
```

## 2. Compile FPGA Kernels
Considering that step one is already done and current directory is `build`. This step generates `*.xo` files needed for the linking process. 
For HW-Emulation:
```
make compile_hwemu
```
For HW(system build for real FPGA):
```
make compile_hw
```

## 3. Link FPGA Kernels
Considering that steps one and two are already done and current directory is `build`. This step generates requested `*.xclbin` file needed for the host program.
For HW-Emulation:
```
make link_hwemu
```
For HW(system build for real FPGA):
```
make link_hw
```

## 4. Launch The Host Program
Considering that steps one, two and three are already done, current directory is `build` and the default shell is `bash`. This command is the unified solution to launch the host program in `sw-emu`, `hw-emu` or `hw` modes.
```
sh LaunchDeepPointV1FPGA.sh
```

## Extra. Setting Kernel Clock Frequencies 
Refer to `CMakeLists.txt` script to change kernel clock frequency.

## Extra. Setting Kernel Arguments' DDR Banks 
Refer to `CMakeLists.txt` script to assign DDR bank for each kernel's arguments.

## Extra. Run Synthesis Only
Considering that step one is already done and current directory is `build`, the following command would only run hls synthesis for all of the kernels.
```
make synthesis
```

# Platforms
Refer to the table below.

Name | Supported Platform | Implementation
---  | ---                | ---
ModelArch01 | CPU                   | CPU 
ModelArch05 | CPU, FPGA             | Xilinx SDAccel Platform 

# UnitTests for Platforms
Unlike parent project, this repository does not use UnitTest++ framework for managing unittests.
The reason is that declaring multiple executables in Eclipse SDAceel managed build environment is not straight forward. This repository has only one main executable(MainExecutable.cpp) that handles both classifier and unit tests while you can enable or disable each phase in the source code.

# Credit
Used repositories are listed below:
* [cnpy](https://github.com/rogersce/cnpy)
* [PointNet](https://github.com/charlesq34/pointnet)
* [PointNet++](https://github.com/charlesq34/pointnet2)
* [hlslib](https://github.com/definelicht/hlslib)
* [dgcnn](https://github.com/WangYueFt/dgcnn)

 
## Current Linker Options(Temp.):
```--sp task_datamover_mod1_float_1.m_axi_gmem0:bank1 --sp task_datamover_mod1_float_1.m_axi_gmem1:bank2 --sp task_datamover_mod1_int_1.m_axi_gmemi0:bank1 --sp task_datamover_mod1_int_1.m_axi_gmemi1:bank2 --sp task_concat_1.m_axi_gmem1:bank2 --sp task_concat_1.m_axi_gmem2:bank2 --sp task_tile_1.m_axi_gmem1:bank2 --sp task_conv2_1x1_direct_1.m_axi_gmem1:bank2 --sp task_conv2_1x1_direct_1.m_axi_gmem2:bank2```

