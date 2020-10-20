# Memory Bank Optimizer
`bank_optimizer_v2.py` is created to find optimized memory banks for the kernels with the objective of minimizing the number of `DataMover` kernel launches needed to execute the computation graph of the selected `TopModelArch`.

# Instructions
1. Run the selected `TopModelArch` on FPGA for `hostlog_0trace.log` to be created.
2. Open `hostlog_0trace.log` and find the line with `[source XilinxImplementation.cpp][function DumpDataMoverLaunchLogs][line 173]`
3. Copy the content of the line and paste it in `bank_optimizer_v2.py`, method `get_objective` and line `objective =`
4. Assign the allowed banks per kernel like `banks_transpose=[1,2]` to allow banks one and two to be selected for kernel `Transpose`, or `banks_transpose=[1]` to force the kernel to use only the bank one.
5. Run the script.
6. Use the output to configure `config` submodule of the main `DeepPoint-V1-FPGA` repository and then rebuild the FPGA image.
