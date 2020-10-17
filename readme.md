<p align="center"><img width="50%" src="https://gitlab.com/salehjg/deeppoint-v1-fpga/-/raw/optimizing01_area_f1/cover_rc.png" /></p>
# DeepPoint-V1 Project
This repository contains the code base for Xilinx SDAccel FPGA implementation of [Dynamic Graph CNN](https://github.com/WangYueFt/dgcnn) model.

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

# 1. Building The Host Program
```
mkdir build
cd build
cmake ..
make DeepPointV1FPGA
```
In order to automate the building modes, the PasteBin agent is developed to automatically upload log files generated by XOCC during compilation and linking processes to PasteBin. Just make sure that it is enabled in the main CMakeLists.txt and username, password, and API key of your PasteBin account are set.  
The linking process requires a large amount of free memory(~30GB of ram for 8 parallel jobs) and close to 5GBs of disk space which takes almost 9 hours to finish with a `i7-6700HQ` machine.

# 2. Compiling FPGA Kernels
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

# 3. Linking FPGA Kernels
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

# 4. Automated Build
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


# 5. Launching The Host Program
Considering that steps one, two and three are already done, current directory is `build` and the default shell is `bash`. This command is the unified solution to launch the host program in `sw-emu`, `hw-emu` or `hw` modes.
```
sh LaunchDeepPointV1FPGA.sh
```
The launcher script forwards its arguments to the host program.

## Platforms
Refer to the table below.

Name | Supported Platform | Implementation | Notes
---  | ---                | --- | ---
ModelArch01 | CPU         | CPU | CPU Only
ModelArch02 | CPU, FPGA   | Xilinx SDAccel Platform | FPGA Only

## OpenCL UnitTests
To run the OCL unittests:
```
sh LaunchDeepPointV1FPGA.sh -t
```

## Kernel-specific UnitTests
In order to make debugging of the kernels easier, separate unit tests are developed(`test` directory). These tests are isolated from OpenCL platform and therefore could be debugged as normal CPU c++ codes.
```
make test
```

# 6. AWS F1 Deployment
Please refer to `AWS-F1-Wiki.md`.

# 7. Project Structure
## Branches
This repository contains multiple branches as described below:

Branch | AXI Width | DType | Tool | Notes
---  |  --- |  --- |  --- |  ---
master | 512-bits | float32 | SDx2019.1 | DEPRECATED
axi32 | 32-bits | float32 | SDx2019.1 | DEPRECATED
new_transpose | 512-bits | float32 | SDx2019.1 | DEPRECATED
optimizing01 | 512-bits | float32 | SDx2019.1 | DEPRECATED
optimizing01_area_f1 | 512-bits | float32 | SDx2019.1 | Up-to-date
vitis20192_axi512 | 512-bits | float32 | Vitis2019.2 | HW build fails with clock partitioning error

# 8. Useful Tips
## Debugging Host-side in CLion
In order to debug the host-side program in any modes(`sw_emu`, `hw_emu`, or `system`), CLion or any other C++ IDE could be used.

Remember to run `scripts/debug_script.sh` before starting debugging session. Note that class `XilinxImplementation` is configured to select `sw_emu` in the case that variable `XCL_EMULATION_MODE` was not set beforehand.  

## Launching Vivado HLS
It is possible to launch Vivado HLS GUI and optimize the kernel of choice. This could be done after running a `hw_emu` build:
```
cd _x/task_<KERNEL>_solution/task_<KERNEL>
vivado_hls -p task_<KERNEL>
```
* Please note that any changes to the source files will be reflected on the main repository files.

# 9. Publication
TBD.

# 10. References
These repositories are used in this project:
* [dgcnn](https://github.com/WangYueFt/dgcnn)
* [DeepPointV1-GPGPU](https://gitlab.com/salehjg/DeepPoint-V1-GPGPU)
* [hlslib](https://github.com/definelicht/hlslib)
* [gemm_hls](https://github.com/spcl/gemm_hls)
* [pp4fpgas](https://github.com/KastnerRG/pp4fpgas)
* [cnpy](https://github.com/rogersce/cnpy)
* [PointNet](https://github.com/charlesq34/pointnet)
* [PointNet++](https://github.com/charlesq34/pointnet2)
* [argparse](https://github.com/jamolnng/argparse)
* [spdlog](https://github.com/gabime/spdlog)



