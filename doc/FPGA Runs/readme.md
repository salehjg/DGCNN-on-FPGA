# FPGA Run-logs
This directory contains results of FPGA runs. 

# How To Analyze
The `analyze.py` is a python3 script, to use it make sure requirements are met.

## Requirements
```
sudo pip3 install numpy xlsxwriter pytest pandas
```

## Run the script
```
python3 analyze.py <run-dir>/DeepPointV1FPGA_Host_Log.txt
```