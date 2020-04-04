# DeepPoint-V1 Project
This repository contains the code base for Xilinx SDAccel FPGA implementation of official DGCNN model.

# Build System
As easy as it is to use SDx GUI, it is recommended to use provided cmake scripts to run synthesis and build the binaries for both the selected FPGA platform and the host.

# Dependencies
This project relies on these software/libraries(These should be installed on the OS):
```
Xilinx SDAccel 2019.1(Tested), 2018.3 2018.2 2017.4(Not Tested)
Xilinx XRT
python2.7(Symlinked as `python`)
PasteBin(Library for Python2.7, pip available)
CMake3 (>3.0, Do **not** use default CMake package available on AWS-F1)
Bash (>4.0, Dash and others are not tested)
```

# Configuration
To make it easier to explore the design space and try different configurations, all of the parameters that affect the output performance of the task kernels are gathered in a separate submodule repository at directory `config`. 

# How to...
## 1. Building The Host Program
```
mkdir build
cd build
cmake ..
make DeepPointV1FPGA
```
In order to automate the building modes, the PasteBin agent is developed to automatically upload log files generated by XOCC during compilation and linking processes to PasteBin. Just make sure that it is enabled in the main CMakeLists.txt and username, password, and API key of your PasteBin account are set.
## 2. Compiling FPGA Kernels
Considering that step one is already done and current directory is `build`. This step generates `*.xo` files needed for the linking process.  
For SW-Emulation:
```
make compile_swemu
```
For HW-Emulation:
```
make compile_hwemu
```
For HW(system build for real FPGA):
```
make compile_hw
```

## 3. Linking FPGA Kernels
Considering that steps one and two are already done and current directory is `build`. This step generates requested `*.xclbin` file needed for the host program.  
For SW-Emulation:
```
make link_swemu
```
For HW-Emulation:
```
make link_hwemu
```
For HW(system build for real FPGA):
```
make link_hw
```

## 4. Automated Build
The autobuild scripts are intended to make building process on an AWS instance easier. They compile and link the project consecutively and take log of each step in a text file. Finally, after finishing up, the instance would be powered off with the `sudo poweroff` command. (considering that the root user has no password)  
For SW-Emulation:
```
bash autobuild_swemu
```
For HW-Emulation:
```
bash autobuild_hwemu
```
For HW(system build for real FPGA):
```
bash autobuild_hw
```


## 5. Launching The Host Program
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

Name | Supported Platform | Implementation | Notes
---  | ---                | --- | ---
ModelArch01 | CPU                   | CPU | Review's needed
ModelArch05 | CPU, FPGA             | Xilinx SDAccel Platform | Review's needed
ModelArch04 | CPU             | CPU | Compatible with new `Tile` layer

# UnitTests for Platforms
This repository has only one main executable(MainExecutable.cpp) that handles both the classifier and the unit tests while you can enable or disable each phase in the source code.

# UnitTests for Kernels
In order to make design and debugging of the kernels much more easier, separate unit tests are developed(`test` directory). These tests are isolated from OpenCL platform and therefore could be debugged as normal CPU codes.

# Debugging Host-side in CLion
In order to debug the host-side program in any modes(`sw_emu`, `hw_emu`, or `system`), one could use CLion or any other IDE.

Remember to run `scripts/debug_script.sh` before debugging session. `XilinxImplementation` is configured to select `sw_emu` in case variable `XCL_EMULATION_MODE` was not set beforehand.  

# Credit
Used repositories are listed below:
* [cnpy](https://github.com/rogersce/cnpy)
* [PointNet](https://github.com/charlesq34/pointnet)
* [PointNet++](https://github.com/charlesq34/pointnet2)
* [hlslib](https://github.com/definelicht/hlslib)
* [dgcnn](https://github.com/WangYueFt/dgcnn)

