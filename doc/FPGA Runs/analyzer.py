from typing import List

import parser as analyzer
import numpy as np
import pandas as pd
import sys
import datetime


class ReportGenerator:
    def __init__(self, layers: List[analyzer.Layer]):
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
            if src_layer.is_fpga:
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_times(src_layer.sub_layers[i])
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
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))

                total_device_usec = accumulate_device_times(layer)
                total_usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                perlayer_dataframe['AccumulatedHostOnly(us)'].append(total_usec - total_device_usec)
                perlayer_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
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
                total_device_usec += accumulate_device_times(layer)
                total_usec += layer.host_elapsed / datetime.timedelta(microseconds=1)
                total_host_usec += total_usec - total_device_usec
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
            perlayer_dataframe['DeviceOnly(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
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
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
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
            if src_layer.is_fpga:
                deviceonly_usec += src_layer.device_elapsed_us
            subs = len(src_layer.sub_layers)
            for i in range(subs):
                deviceonly_usec += accumulate_device_times(src_layer.sub_layers[i])
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
            perlayer_dataframe['AccumulatedHostOnly(us)'] = []
            perlayer_dataframe['AccumulatedDeviceOnly(us)'] = []
            perlayer_dataframe['Total(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))

                total_device_usec = accumulate_device_times(layer)
                total_usec = layer.host_elapsed / datetime.timedelta(microseconds=1)
                perlayer_dataframe['AccumulatedHostOnly(us)'].append(total_usec - total_device_usec)
                perlayer_dataframe['AccumulatedDeviceOnly(us)'].append(total_device_usec)
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
                total_device_usec += accumulate_device_times(layer)
                total_usec += layer.host_elapsed / datetime.timedelta(microseconds=1)
                total_host_usec += total_usec - total_device_usec
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
            perlayer_dataframe['DeviceOnly(us)'] = []

            for layer in sorted_by_layername[name]:
                perlayer_dataframe['Shape1'].append(str(layer.shape1))
                perlayer_dataframe['Shape2'].append(str(layer.shape2))
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
    if len(sys.argv) != 2:
        print("DeepPointV1-FPGA Log Analyzer")
        print("Error, python3 analyze.py  <path to report_host_kernel.log   OR   report_kernel.log>")
        sys.exit(1)
    layers = analyzer.analyze(sys.argv[1])
    obj = ReportGenerator(layers)
    obj.generate_cpuimplementation_layerbased()
    obj.generate_xilinximplementaion_layerbased_host_device()
    obj.generate_xilinximplementaion_kernelbased_deviceonly()
    obj.generate_anyimplementaion_layerbased_host_device()
    obj.generate_xilinximplementaion_datamover_deviceonly()


main()
