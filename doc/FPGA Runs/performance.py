import numpy as np
import pandas as pd
import sys
import datetime


def load_text_file(path):
    arr = []
    with open(path) as f:
        for line in f:
            arr.append(line)
    return arr


def decode_namespaces(str_lines):
    def strip_conf_line(cline):
        return cline.strip().replace(' = ', '=').replace(';', '')

    def fix_keys(key):
        ret = key.replace('\n', '')
        ret = ret.replace('\r', '')
        ret = ret.replace('\t', '')
        ret = ret.replace(' ', '')
        return ret

    configs = {}
    state = -1
    current_kernel = ''
    for line in str_lines:
        if line == '':
            continue

        if line.find('}') != -1:
            state = state - 1

        if state >= 0:
            line1 = strip_conf_line(line)
            i_equal = line1.find('=')
            i_space_last = -1
            for i in range(i_equal):
                j = line1.find(' ', i)
                if j != -1:
                    i_space_last = j
            param_name = line1[i_space_last + 1: i_equal]
            param_val = line1[i_equal + 1:]
            if param_name != '' and param_name[0] != '#' and param_name[0:2] != '//':
                if param_val.isdigit():
                    configs[current_kernel][param_name] = int(param_val)
                else:
                    configs[current_kernel][param_name] = param_val

        if line.find('namespace') != -1:
            current_kernel = fix_keys(line)[len('namespace'):]
            configs[current_kernel] = {}
            state = -1

        if line.find('{') != -1:
            state = state + 1

    return configs


class PerformanceHelper:
    def __init__(self, path_to_config_header, freq_mhz):
        lines = load_text_file(path_to_config_header)
        self.configs = decode_namespaces(lines)
        self.frequency_mhz = freq_mhz

    def create_report_dict(self,
                           throughput_read_bytes_per_second,
                           throughput_write_bytes_per_second,
                           n_ops,
                           flop_per_second_expected,
                           flop_per_second_actual,
                           message):
        return {
            'throughput_read_bytes_per_second': throughput_read_bytes_per_second,
            'throughput_write_bytes_per_second': throughput_write_bytes_per_second,
            'n_ops': n_ops,
            'flop_per_second_expected': flop_per_second_expected,
            'flop_per_second_actual': flop_per_second_actual,
            'message': str(message)
        }

    def calc_performance_conv(self, shape_input, _shape_weight, device_runtime_us):
        def outer_tiles_n(arg_size_n):
            return arg_size_n / self.configs['ConfigTaskConv2']['kOuterTileSizeN']

        def outer_tiles_m(arg_size_m):
            return arg_size_m / self.configs['ConfigTaskConv2']['kOuterTileSizeM']

        def inner_tiles_n():
            return self.configs['ConfigTaskConv2']['kOuterTileSizeN'] / self.configs['ConfigTaskConv2'][
                'kInnerTileSizeN']

        def inner_tiles_m():
            return self.configs['ConfigTaskConv2']['kOuterTileSizeM'] / self.configs['ConfigTaskConv2'][
                'kComputeTileSizeM']

        def compute_tiles_n():
            return self.configs['ConfigTaskConv2']['kInnerTileSizeN'] / self.configs['ConfigTaskConv2'][
                'kComputeTileSizeN']

        assert len(shape_input) == 4
        assert len(_shape_weight) == 4
        assert _shape_weight[0] == 1
        assert _shape_weight[1] == 1

        if _shape_weight[3] < self.configs['ConfigTaskConv2']['kOuterTileSizeM']:
            print("Shared_MLP(gemmHLS): Auto padding weight tensor in axis-3 from ", _shape_weight[3], " to kOuterTileSizeM = ", self.configs['ConfigTaskConv2']['kOuterTileSizeM'],"...")
            _shape_weight[3] = self.configs['ConfigTaskConv2']['kOuterTileSizeM']

        shape_weight = _shape_weight[2:]
        assert shape_input[3] == shape_weight[0]
        size_n = shape_input[0] * shape_input[1] * shape_input[2]
        size_k = shape_input[3]
        size_m = shape_weight[1]

        n_ops = 2 * size_n * size_k * size_m
        expected_runtime = outer_tiles_n(size_n) * outer_tiles_m(size_m) * (
                size_k * inner_tiles_n() * inner_tiles_m() +
                inner_tiles_n() * (
                        self.configs['ConfigTaskConv2']['kComputeTileSizeM'] * inner_tiles_m() +
                        compute_tiles_n() * self.configs['ConfigTaskConv2']['kComputeTileSizeN'] * inner_tiles_m())
        ) / (self.frequency_mhz * 1e+06)
        expected_performance = n_ops / expected_runtime
        actual_performance = n_ops / (device_runtime_us * 1e-06)
        return self.create_report_dict(0, 0, n_ops, expected_performance, actual_performance, '')

    def calc_performance_matmul(self, shape_input1, shape_input2, device_runtime_us):
        size_batch = size_n = size_k = size_m = -1
        if len(shape_input1) == 2:
            assert len(shape_input2) == 2
            assert shape_input1[1] == shape_input2[0]
            size_n = shape_input1[0]
            size_k = shape_input1[1]
            size_m = shape_input2[1]
            size_batch = 1

        if len(shape_input1) == 3:
            assert len(shape_input2) == 3
            assert shape_input1[2] == shape_input2[1]
            assert shape_input1[0] == shape_input2[0]
            size_n = shape_input1[1]
            size_k = shape_input1[2]
            size_m = shape_input2[2]
            size_batch = shape_input1[0]

        n_ops = 2 * size_batch * size_n * size_k * size_m

        actual_performance = n_ops / (device_runtime_us * 1e-06)
        return self.create_report_dict(0, 0, n_ops, 0, actual_performance, '')

    def calc_performance_topk(self, shape_input, device_runtime_us):
        assert len(shape_input) == 3
        size_batch = shape_input[0] * shape_input[1]
        size_slice = shape_input[2]
        count_processing_element = self.configs['ConfigTaskTopK']['UnitCount']
        size_of_dtype = 4  # float

        data_size_bytes = size_batch * size_slice * size_of_dtype
        throughput_per_pe_bytes_per_second = (data_size_bytes / count_processing_element) / (device_runtime_us * 1e-06)
        throughput_total_bytes_per_second = data_size_bytes / (device_runtime_us * 1e-06)

        return self.create_report_dict(throughput_total_bytes_per_second, 0, 0, 0, 0,
                                       ''.join(['bytes/(sec.1 PE): ', str(throughput_per_pe_bytes_per_second)]))

    def calc_performance_matops(self, shape_input1, device_runtime_us):
        size_of_dtype = 4
        data_size = size_of_dtype
        for s in shape_input1:
            data_size = data_size * s

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_concat(self, shape_input1, shape_input2, device_runtime_us):
        assert len(shape_input1) == 4
        assert len(shape_input2) == 4
        assert shape_input1[0] == shape_input2[0]
        assert shape_input1[1] == shape_input2[1]
        assert shape_input1[2] == shape_input2[2]

        size_of_dtype = 4
        data_size = size_of_dtype * shape_input1[0] * shape_input1[1] * shape_input1[2] * (
                shape_input1[3] + shape_input2[3])

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_gather(self, shape_input, shape_indices, device_runtime_us):
        assert len(shape_input) == 3
        assert len(shape_indices) == 3
        assert shape_input[0] == shape_indices[0]
        assert shape_input[1] == shape_indices[1]

        size_of_dtype = 4
        data_size = size_of_dtype * shape_input[0] * shape_input[1] * shape_indices[2] * shape_input[2]

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_relusqrtsquare(self, shape_input, device_runtime_us):
        size_of_dtype = 4
        data_size = size_of_dtype
        for s in shape_input:
            data_size = data_size * s

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_transpose(self, shape_input, device_runtime_us):
        size_of_dtype = 4
        data_size = size_of_dtype
        for s in shape_input:
            data_size = data_size * s

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_tile(self, shape_input, log_msg, device_runtime_us):
        # Tile:tileAxis=2,tileCount=20,Shape1=5x1024x64x,
        tile_axis = 0
        tile_count = 0

        msg = log_msg[len('Tile:') + 1:]
        fields = msg.split(',')
        for f in fields:
            if f.find('tileAxis=') != -1:
                tile_axis = int(f[len('tileAxis='):])
            if f.find('tileCount=') != -1:
                tile_count = int(f[len('tileCount='):])

        size_of_dtype = 4
        data_size = size_of_dtype * tile_count
        for s in shape_input:
            data_size = data_size * s

        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_datamover(self, _shape_input, device_runtime_us):
        size_of_dtype = 4
        data_size = size_of_dtype
        if type(_shape_input) == type(2222) and _shape_input == 0:
            shape_input = [0]
        else:
            shape_input = _shape_input
        for s in shape_input:
            data_size = data_size * s
        return self.create_report_dict(0, (data_size / (device_runtime_us * 1e-06)), 0, 0, 0, '')

    def calc_performance_reduce(
            self,
            shape_input,
            log_msg,
            device_runtime_us):
        """"
        =====================================================================
        _Reduce_Task:reduceSum=0,reduceMax:1.000000,Combination=0-0-1-0-,
        =====================================================================
        _Reduce_Task:reduceSum=1,reduceMax:0.000000,Combination=0-0-1-0-,
        =====================================================================
        _Reduce_Task:reduceSum=0,reduceMax:1.000000,Combination=0-1-0-0-,
        =====================================================================
        _Reduce_Task:reduceSum=1,reduceMax:0.000000,Combination=1-1-1-0-,
        =====================================================================
        """
        is_sum_rank3_axis2 = False
        is_sum_rank4_axes012 = False
        is_max_rank3_axis1 = False

        dim0 = 1
        dim1 = 1
        dim2 = 1
        dim3 = 1
        n_ops = 0

        dim0_out = 1
        dim1_out = 1
        dim2_out = 1

        # -------------------------------------------------------------------------------------------------------------
        if log_msg.find('reduceMax:1.000000') != -1 and log_msg.find('Combination=0-0-1-0-') != -1:
            # reduceMaxFFTF ---> reduceMaxFTF
            is_max_rank3_axis1 = True
            dim0 = shape_input[0] * shape_input[1]
            dim1 = shape_input[2]
            dim2 = shape_input[3]
            dim0_out = dim0
            dim1_out = dim2
            n_ops = dim0 * (dim1 - 1) * dim2

        if log_msg.find('reduceMax:1.000000') != -1 and log_msg.find('Combination=0-1-0-0-') != -1:
            # reduceMaxFTFF ---> reduceMaxFTF,     Dim2 should be equal 1
            assert shape_input[2] == 1
            is_max_rank3_axis1 = True
            dim0 = shape_input[0]
            dim1 = shape_input[1]
            dim2 = shape_input[3]
            dim0_out = dim0
            dim1_out = dim2
            n_ops = dim0 * (dim1 - 1) * dim2

        # -------------------------------------------------------------------------------------------------------------
        if log_msg.find('reduceSum=1') != -1 and log_msg.find('Combination=0-0-1-0-') != -1:
            is_sum_rank3_axis2 = True
            dim0 = shape_input[0]
            dim1 = shape_input[1]
            dim2 = shape_input[2]
            dim0_out = dim0
            dim1_out = dim1
            n_ops = (dim0 * dim1) * dim2

        # -------------------------------------------------------------------------------------------------------------
        if log_msg.find('reduceSum=1') != -1 and log_msg.find('Combination=1-1-1-0-') != -1:
            is_sum_rank4_axes012 = True
            dim0 = shape_input[0]
            dim1 = shape_input[1]
            dim2 = shape_input[2]
            dim3 = shape_input[3]
            dim0_out = dim3
            n_ops = dim3 * (dim0 * dim1 * dim2 - 1)

        # -------------------------------------------------------------------------------------------------------------
        size_of_dtype = 4
        data_read_size = size_of_dtype * dim0 * dim1 * dim2 * dim3
        data_write_size = size_of_dtype * dim0_out * dim1_out * dim2_out

        return self.create_report_dict(
            (data_read_size / (device_runtime_us * 1e-06)),
            (data_write_size / (device_runtime_us * 1e-06)),
            n_ops,
            0,
            (n_ops / (device_runtime_us * 1e-06)),
            ''.join(
                ['is_sum_rank3_axis2: ', str(is_sum_rank3_axis2), ', is_sum_rank4_axes012: ', str(is_sum_rank4_axes012),
                 ', is_max_rank3_axis1: ', str(is_max_rank3_axis1)])
        )

    def calc_performance_padunpad(
            self,
            shape_input,
            log_msg,
            device_runtime_us):

        # '_PadUnpadLastDim:lastDimUnpadded=0,Shape1=1x1x6x64x,'

        is_pad = False
        is_unpad = False
        last_dim_unpadded = 0
        last_dim_padded = 0
        size_of_dtype = 4
        data_size = size_of_dtype

        for s in shape_input:
            data_size = data_size * s

        data_size = data_size / shape_input[-1]

        msg = log_msg[len('_PadUnpadLastDim:') + 1:]
        fields = msg.split(',')
        for f in fields:
            if f.find('lastDimUnpadded=') != -1:
                last_dim_unpadded = int(f[len('lastDimUnpadded='):])
            if f.find('lastDimPadded=') != -1:
                last_dim_padded = int(f[len('lastDimPadded='):])

        if log_msg.find('lastDimUnpadded=0') != -1:
            # The kernel is used to PAD the input tensor
            is_pad = True
            data_size = data_size * shape_input[-1]

        if log_msg.find('lastDimPadded=0') != -1:
            # The kernel is used to UNPAD the input tensor
            is_unpad = True
            data_size = data_size * last_dim_unpadded

        if is_pad:
            return self.create_report_dict(
                (data_size / (device_runtime_us * 1e-06)),
                0,
                0,
                0,
                0,
                ''.join(['is_pad: ', str(is_pad), ', is_unpad: ', str(is_unpad), ', bytes: ', str(data_size)])
            )
        else:
            return self.create_report_dict(
                0,
                (data_size / (device_runtime_us * 1e-06)),
                0,
                0,
                0,
                ''.join(['is_pad: ', str(is_pad), ', is_unpad: ', str(is_unpad), ', bytes: ', str(data_size)])
            )


"""
performance_analyzer = PerformanceHelper(
    '/run/media/saleh/NVME_DATA/01_workspace2/00_SDx2019/'
    'DeepPoint-V1-FPGA/doc/FPGA Runs/fpgarun04_config01/'
    '00/config.h.log', 180)

flops = performance_analyzer.calc_performance_conv([5, 1024, 20, 256], [256, 256], 139427.03)
a = 1
"""
