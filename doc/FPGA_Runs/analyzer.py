from typing import List

import parser as analyzer
import numpy as np
import pandas as pd
import sys
import datetime
import performance as perfhelper


class ReportGenerator:
    def __init__(self, layers: List[analyzer.Layer], path_to_config_h, design_freq_mhz):
        self.perf = perfhelper.PerformanceHelper(path_to_config_h, design_freq_mhz)
        self.layers = layers

    def get_sublayer_depth(self, layer: analyzer.Layer, depth, index):
        if depth == 0:
            return layer
        if depth > 1:
            return self.get_sublayer_depth(layer.sub_layers[index], depth - 1, index)
        else:
            return layer.sub_layers[index]

    def generate_xilinximplementaion_layerbased_host_device(self):
        # Targets:
        #       - Host-side overhead per layer
        #       - Multi-depth layer based analysis
        #       - Total device execution time per layer
        #
        # This method generates layer based report for host-side overhead
        # and total device-side execution-time per layer.
        # This includes higher level layers including mean and variance with multiple sub-layers.
        _layers = self.layers.copy()
        writer = pd.ExcelWriter("xilinximplementaion_layerbased_host_device.xlsx", engine='xlsxwriter')

        def accumulate_device_times(src_layer: analyzer.Layer):
            deviceonly_usec = 0
            if src_layer.is_fpga and src_layer.layer_name != 'LaunchDataMover':
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_times(src_layer.sub_layers[i])
            return deviceonly_usec

        def accumulate_device_datamover_times(src_layer: analyzer.Layer):
            deviceonly_usec = 0
            if src_layer.is_fpga and src_layer.layer_name == 'LaunchDataMover':
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_datamover_times(src_layer.sub_layers[i])
            return deviceonly_usec

        # ---------------------------------------------------
        # 1. Sort by layer name...
        sorted_by_layername = {}

        for layer in _layers:
            if not layer.is_fpga:
                continue
            is_already_added = False
            keys = sorted_by_layername.keys()
            for key in keys:
                if key == layer.layer_name:
                    is_already_added = True

            if is_already_added:
                sorted_by_layername[layer.layer_name].append(layer)
            else:
                sorted_by_layername[layer.layer_name] = [layer]

        layer_names = list(sorted_by_layername.keys())

        # ---------------------------------------------------
        # 2. Generate the per layer sheets...
        for name in layer_names:
            perlayer_dataframe = {}
            perlayer_dataframe['Shape1'] = []
            perlayer_dataframe['Shape2'] = []
            perlayer_dataframe['AccumulatedHostOnly(us)'] = []
            perlayer_dataframe['AccumulatedDeviceOnly(us)'] = []
            perlayer_dataframe['AccumulatedDataMoverOnly(us)'] = []
            perlayer_dataframe['Msg'] = []
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))

                total_device_usec = accumulate_device_times(layer)
                total_device_datamover_usec = accumulate_device_datamover_times(layer)
                total_usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                perlayer_dataframe['AccumulatedHostOnly(us)'].append(total_usec - total_device_usec - total_device_datamover_usec)
                perlayer_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
                perlayer_dataframe['AccumulatedDataMoverOnly(us)'].append(total_device_datamover_usec)
                perlayer_dataframe['Msg'].append(layer.msg)
                perlayer_dataframe['Total(us)'].append(total_usec)

            df = pd.DataFrame(perlayer_dataframe)
            df.to_excel(writer, sheet_name=name)

        # ---------------------------------------------------
        # 3. Generate the summary sheet...
        summary_dataframe = {}
        summary_dataframe['Name'] = []
        summary_dataframe['Launch Count'] = []
        summary_dataframe['AccumulatedHostOnly(us)'] = []
        summary_dataframe['AccumulatedDeviceOnly(us)'] = []
        summary_dataframe['Total(us)'] = []
        for name in layer_names:
            launch_cnt = 0
            total_device_usec = 0
            total_usec = 0
            total_host_usec = 0
            for layer in sorted_by_layername[name]:
                launch_cnt += 1
                __device_usec = accumulate_device_times(layer)
                __usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                __host_usec = __usec - __device_usec
                total_device_usec += accumulate_device_times(layer)
                total_usec += layer.host_elapsed / datetime.timedelta(microseconds=1)
                total_host_usec += __host_usec

            summary_dataframe['Name'].append(name)
            summary_dataframe['Launch Count'].append(launch_cnt)
            summary_dataframe['AccumulatedHostOnly(us)'].append(total_host_usec)
            summary_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
            summary_dataframe['Total(us)'].append(total_usec)

        df = pd.DataFrame(summary_dataframe)
        df.to_excel(writer, sheet_name="Summary")

        writer.save()

    def generate_xilinximplementaion_kernelbased_deviceonly(self):
        # Targets:
        #       - Device kernels only
        #
        # This method generates layer based report for total execution-time per kernel.
        _layers = self.layers.copy()
        writer = pd.ExcelWriter("xilinximplementaion_kernelbased_deviceonly.xlsx", engine='xlsxwriter')

        def get_all_kernels(src_layer: analyzer.Layer):
            kernels = []
            if src_layer.is_fpga and src_layer.device_elapsed_us != 0:
                kernels.append(src_layer)
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                kernels.extend(get_all_kernels(src_layer.sub_layers[i]))
            return kernels

        # ---------------------------------------------------
        # 0. Get all of the device layers (is_fpga=true)
        print("WARN01: Filtering out non kernel layers by the device_elapsed_us param.")
        _device_kernels = []
        for layer in _layers:
            _device_kernels.extend(get_all_kernels(layer))

        # ---------------------------------------------------
        # 1. Sort by layer name...
        sorted_by_layername = {}

        for layer in _device_kernels:
            assert layer.is_fpga
            is_already_added = False
            keys = sorted_by_layername.keys()
            for key in keys:
                if key == layer.layer_name:
                    is_already_added = True

            if is_already_added:
                sorted_by_layername[layer.layer_name].append(layer)
            else:
                sorted_by_layername[layer.layer_name] = [layer]

        layer_names = list(sorted_by_layername.keys())

        # ---------------------------------------------------
        # 2. Generate the per layer sheets...
        for name in layer_names:
            perlayer_dataframe = {}
            perlayer_dataframe['Shape1'] = []
            perlayer_dataframe['Shape2'] = []
            perlayer_dataframe['Msg'] = []
            perlayer_dataframe['DeviceOnly(us)'] = []
            perlayer_dataframe['Throughput Read(MB/s)'] = []
            perlayer_dataframe['Throughput Write(MB/s)'] = []
            perlayer_dataframe['n_OPS'] = []
            perlayer_dataframe['Expected(GFLOP/s)'] = []
            perlayer_dataframe['Actual(GFLOP/s)'] = []
            perlayer_dataframe['Performance_Message'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
                perlayer_dataframe['Msg'].append(str(layer.msg))
                perlayer_dataframe['DeviceOnly(us)'].append(layer.device_elapsed_us)

                perf_report = 0

                if layer.layer_name == 'Transpose':
                    perf_report = self.perf.calc_performance_transpose(layer.shape1, layer.device_elapsed_us)

                if layer.layer_name == 'LaunchDataMover':
                    perf_report = self.perf.calc_performance_datamover(layer.shape1, layer.device_elapsed_us)

                if layer.layer_name == 'MatMul':
                    perf_report = self.perf.calc_performance_matmul(layer.shape1, layer.shape2, layer.device_elapsed_us)

                if layer.layer_name == 'MatOps':
                    perf_report = self.perf.calc_performance_matops(layer.shape1, layer.device_elapsed_us)

                if layer.layer_name == '_ReluSqrtSquare':
                    perf_report = self.perf.calc_performance_relusqrtsquare(layer.shape1, layer.device_elapsed_us)

                if layer.layer_name == '_Reduce_Task':
                    perf_report = self.perf.calc_performance_reduce(layer.shape1, layer.msg, layer.device_elapsed_us)

                if layer.layer_name == 'Tile':
                    perf_report = self.perf.calc_performance_tile(layer.shape1, layer.msg, layer.device_elapsed_us)

                if layer.layer_name == 'TopK':
                    perf_report = self.perf.calc_performance_topk(layer.shape1, layer.device_elapsed_us)

                if layer.layer_name == 'Gather':
                    perf_report = self.perf.calc_performance_gather(layer.shape1, layer.shape2, layer.device_elapsed_us)

                if layer.layer_name == 'Concat2':
                    perf_report = self.perf.calc_performance_concat(layer.shape1, layer.shape2, layer.device_elapsed_us)

                if layer.layer_name == 'Conv2D':
                    perf_report = self.perf.calc_performance_conv(layer.shape1, layer.shape2, layer.device_elapsed_us)

                if layer.layer_name == '_PadUnpadLastDim':
                    perf_report = self.perf.calc_performance_padunpad(layer.shape1, layer.msg, layer.device_elapsed_us)

                perlayer_dataframe['Throughput Read(MB/s)'] .append(
                    round(perf_report['throughput_read_bytes_per_second']/1024/1024, 2)
                )
                perlayer_dataframe['Throughput Write(MB/s)'] .append(
                    round(perf_report['throughput_write_bytes_per_second']/1024/1024, 2)
                )
                perlayer_dataframe['n_OPS'] .append(
                    str(perf_report['n_ops'])
                )
                perlayer_dataframe['Expected(GFLOP/s)'] .append(
                    round(perf_report['flop_per_second_expected']/1e+09, 2)
                )
                perlayer_dataframe['Actual(GFLOP/s)'] .append(
                    round(perf_report['flop_per_second_actual']/1e+09, 2)
                )
                perlayer_dataframe['Performance_Message'] .append(
                    str(perf_report['message'])
                )

            df = pd.DataFrame(perlayer_dataframe)
            df.to_excel(writer, sheet_name=name)

        # ---------------------------------------------------
        # 3. Generate the summary sheet...
        summary_dataframe = {}
        summary_dataframe['Name'] = []
        summary_dataframe['Launch Count'] = []
        summary_dataframe['DeviceOnly(us)'] = []
        for name in layer_names:
            launch_cnt = 0
            total_device_usec = 0
            for layer in sorted_by_layername[name]:
                launch_cnt += 1
                total_device_usec += layer.device_elapsed_us
            summary_dataframe['Name'].append(name)
            summary_dataframe['Launch Count'].append(launch_cnt)
            summary_dataframe['DeviceOnly(us)'].append(total_device_usec)

        df = pd.DataFrame(summary_dataframe)
        df.to_excel(writer, sheet_name="Summary")

        writer.save()

    def generate_cpuimplementation_layerbased(self):
        # Targets:
        #       - Cpu layers only
        #       - Multi-depth layer based analysis
        #
        # This includes higher level layers including mean and variance with multiple sub-layers.
        _layers = self.layers.copy()
        writer = pd.ExcelWriter("cpuimplementation_layerbased.xlsx", engine='xlsxwriter')

        def accumulate_device_times(src_layer: analyzer.Layer):
            deviceonly_usec = 0
            if src_layer.is_cpu:
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_times(src_layer.sub_layers[i])
            return deviceonly_usec

        # ---------------------------------------------------
        # 1. Sort by layer name...
        sorted_by_layername = {}

        for layer in _layers:
            if not layer.is_cpu:
                continue
            is_already_added = False
            keys = sorted_by_layername.keys()
            for key in keys:
                if key == layer.layer_name:
                    is_already_added = True

            if is_already_added:
                sorted_by_layername[layer.layer_name].append(layer)
            else:
                sorted_by_layername[layer.layer_name] = [layer]

        layer_names = list(sorted_by_layername.keys())

        # ---------------------------------------------------
        # 2. Generate the per layer sheets...
        for name in layer_names:
            perlayer_dataframe = {}
            perlayer_dataframe['Shape1'] = []
            perlayer_dataframe['Shape2'] = []
            perlayer_dataframe['Msg'] = []
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
                perlayer_dataframe['Msg'].append(str(layer.msg))
                total_usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                perlayer_dataframe['Total(us)'].append(total_usec)

            df = pd.DataFrame(perlayer_dataframe)
            df.to_excel(writer, sheet_name=name)

        # ---------------------------------------------------
        # 3. Generate the summary sheet...
        summary_dataframe = {}
        summary_dataframe['Name'] = []
        summary_dataframe['Launch Count'] = []
        summary_dataframe['Total(us)'] = []
        for name in layer_names:
            launch_cnt = 0
            total_usec = 0
            for layer in sorted_by_layername[name]:
                launch_cnt += 1
                total_usec += layer.host_elapsed / datetime.timedelta(microseconds=1)
            summary_dataframe['Name'].append(name)
            summary_dataframe['Launch Count'].append(launch_cnt)
            summary_dataframe['Total(us)'].append(total_usec)

        df = pd.DataFrame(summary_dataframe)
        df.to_excel(writer, sheet_name="Summary")

        writer.save()

    def generate_anyimplementaion_layerbased_host_device(self):
        # Targets:
        #       - Host-side overhead per layer
        #       - Multi-depth layer based analysis
        #       - Total device execution time per layer
        #       - CpuImplementation layers
        #       - XilinxImplementation layers
        #
        # This method generates layer based report for host-side overhead
        # and total device-side execution-time per layer.
        # This includes higher level layers including mean and variance with multiple sub-layers.
        _layers = self.layers.copy()
        writer = pd.ExcelWriter("anyimplementaion_layerbased_host_device.xlsx", engine='xlsxwriter')

        def accumulate_device_times(src_layer: analyzer.Layer):
            deviceonly_usec = 0
            if src_layer.is_fpga and src_layer.layer_name != 'LaunchDataMover':
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_times(src_layer.sub_layers[i])
            return deviceonly_usec

        def accumulate_device_datamover_times(src_layer: analyzer.Layer):
            deviceonly_usec = 0
            if src_layer.is_fpga and src_layer.layer_name == 'LaunchDataMover':
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_datamover_times(src_layer.sub_layers[i])
            return deviceonly_usec

        # ---------------------------------------------------
        # 1. Sort by layer name...
        sorted_by_layername = {}

        for layer in _layers:
            is_already_added = False
            keys = sorted_by_layername.keys()
            for key in keys:
                if key == layer.layer_name:
                    is_already_added = True

            if is_already_added:
                sorted_by_layername[layer.layer_name].append(layer)
            else:
                sorted_by_layername[layer.layer_name] = [layer]

        layer_names = list(sorted_by_layername.keys())

        # ---------------------------------------------------
        # 2. Generate the per layer sheets...
        for name in layer_names:
            perlayer_dataframe = {}
            perlayer_dataframe['Shape1'] = []
            perlayer_dataframe['Shape2'] = []
            perlayer_dataframe['Msg'] = []
            perlayer_dataframe['AccumulatedHostOnly(us)'] = []
            perlayer_dataframe['AccumulatedDeviceOnly(us)'] = []
            perlayer_dataframe['AccumulatedDataMoverOnly(us)'] = []
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
                perlayer_dataframe['Msg'].append(str(layer.msg))

                total_device_usec = accumulate_device_times(layer)
                total_device_datamover_usec = accumulate_device_datamover_times(layer)
                total_usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                perlayer_dataframe['AccumulatedHostOnly(us)'].append(total_usec - total_device_usec - total_device_datamover_usec)
                perlayer_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
                perlayer_dataframe['AccumulatedDataMoverOnly(us)'].append(total_device_datamover_usec)
                perlayer_dataframe['Total(us)'].append(total_usec)

            df = pd.DataFrame(perlayer_dataframe)
            df.to_excel(writer, sheet_name=name)

        # ---------------------------------------------------
        # 3. Generate the summary sheet...
        summary_dataframe = {}
        summary_dataframe['Name'] = []
        summary_dataframe['Launch Count'] = []
        summary_dataframe['AccumulatedHostOnly(us)'] = []
        summary_dataframe['AccumulatedDeviceOnly(us)'] = []
        summary_dataframe['Total(us)'] = []
        for name in layer_names:
            launch_cnt = 0
            total_device_usec = 0
            total_usec = 0
            total_host_usec = 0
            for layer in sorted_by_layername[name]:
                launch_cnt += 1
                __device_usec = accumulate_device_times(layer)
                __usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                __host_usec = __usec - __device_usec
                total_device_usec += __device_usec
                total_usec += __usec
                total_host_usec += __host_usec

            summary_dataframe['Name'].append(name)
            summary_dataframe['Launch Count'].append(launch_cnt)
            summary_dataframe['AccumulatedHostOnly(us)'].append(total_host_usec)
            summary_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
            summary_dataframe['Total(us)'].append(total_usec)

        df = pd.DataFrame(summary_dataframe)
        df.to_excel(writer, sheet_name="Summary")

        writer.save()

    def generate_xilinximplementaion_datamover_deviceonly(self):
        # Targets:
        #       - Device kernel datamover only
        #
        # This method generates layer based report for total execution-time per kernel.
        _layers = self.layers.copy()
        writer = pd.ExcelWriter("xilinximplementaion_datamover_deviceonly.xlsx", engine='xlsxwriter')

        def get_all_kernels(src_layer: analyzer.Layer):
            kernels = []
            if src_layer.is_fpga and src_layer.device_elapsed_us != 0:
                kernels.append(src_layer)
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                kernels.extend(get_all_kernels(src_layer.sub_layers[i]))
            return kernels

        # ---------------------------------------------------
        # 0. Get all of the device layers (is_fpga=true)
        print("WARN01: Filtering out non kernel layers by the device_elapsed_us param.")
        _device_kernels = []
        for layer in _layers:
            _device_kernels.extend(get_all_kernels(layer))

        # ---------------------------------------------------
        # 1. Sort by layer name...
        sorted_by_layername = {}

        for layer in _device_kernels:
            assert layer.is_fpga
            is_already_added = False
            keys = sorted_by_layername.keys()
            for key in keys:
                if key == layer.layer_name:
                    is_already_added = True

            if layer.layer_name != "LaunchDataMover":
                continue

            if is_already_added:
                sorted_by_layername[layer.layer_name].append(layer)
            else:
                sorted_by_layername[layer.layer_name] = [layer]

        layer_names = list(sorted_by_layername.keys())

        # ---------------------------------------------------
        # 2. Generate the per layer sheets...
        for name in layer_names:
            perlayer_dataframe = {}
            perlayer_dataframe['Shape1'] = []
            perlayer_dataframe['Shape2'] = []
            perlayer_dataframe['Msg'] = []
            perlayer_dataframe['DeviceOnly(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
                perlayer_dataframe['Msg'].append(str(layer.msg))
                perlayer_dataframe['DeviceOnly(us)'].append(layer.device_elapsed_us)

            df = pd.DataFrame(perlayer_dataframe)
            df.to_excel(writer, sheet_name=name)

        # ---------------------------------------------------
        # 3. Generate the summary sheet...
        summary_dataframe = {}
        summary_dataframe['Name'] = []
        summary_dataframe['Launch Count'] = []
        summary_dataframe['DeviceOnly(us)'] = []
        for name in layer_names:
            launch_cnt = 0
            total_device_usec = 0
            for layer in sorted_by_layername[name]:
                launch_cnt += 1
                total_device_usec += layer.device_elapsed_us
            summary_dataframe['Name'].append(name)
            summary_dataframe['Launch Count'].append(launch_cnt)
            summary_dataframe['DeviceOnly(us)'].append(total_device_usec)

        df = pd.DataFrame(summary_dataframe)
        df.to_excel(writer, sheet_name="Summary")

        writer.save()


def main():
    if len(sys.argv) != 4:
        print("DeepPointV1-FPGA Log Analyzer")
        print("Error, python3 analyze.py <arg1> <arg2> <arg3>")
        print("\t<arg1>: Path to report_host_kernel.log   OR   report_kernel.log")
        print("\t<arg2>: Path to config.h.log")
        print("\t<arg3>: Design frequency in MHz")
        sys.exit(1)

    layers = analyzer.analyze(sys.argv[1])
    obj = ReportGenerator(layers, sys.argv[2], int(sys.argv[3]))
    obj.generate_cpuimplementation_layerbased()
    obj.generate_xilinximplementaion_layerbased_host_device()
    obj.generate_xilinximplementaion_kernelbased_deviceonly()
    obj.generate_anyimplementaion_layerbased_host_device()
    obj.generate_xilinximplementaion_datamover_deviceonly()


main()
