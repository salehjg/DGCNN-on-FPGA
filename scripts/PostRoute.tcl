# Run on tcl.post route(only for a HW build)
write_checkpoint -force ${PostRouteCheckpointFile}
report_power -hier all -file ${PowerReportTextFile} -rpx ${PowerReportRpxFile}
