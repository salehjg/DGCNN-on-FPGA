# Run on tcl.pre route(only for a HW build)
report_utilization -slr -file ${SlrUtilOutputFile}
report_utilization -pblocks pblock_pcie -pblocks pblock_SH -pblocks pblock_SH_SHIM -file ${PblockUtilOutputFile}
write_checkpoint -force design_pre_route.dcp
