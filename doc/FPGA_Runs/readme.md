# FPGA Run-logs
This directory contains results of FPGA runs. 

# How To Analyze
`analyzer.py` is a python3 script, to use it make sure the requirements are met.

## Requirements
```
sudo pip3 install numpy xlsxwriter pytest pandas
```

## Run the script
```
python3 analyzer.py <run-dir>/report_host_kernel.log
```
