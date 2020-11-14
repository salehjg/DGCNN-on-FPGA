import numpy as np
import pandas as pd
import sys
import datetime


# [18:55:13.708601][debug][source CpuImplementation.cpp][function PrintInfo][line 56] ## Tile: tileAxis=2, tileCount=1024, Shape1=5x1024x,

# MAIN ANALYZER CLASS
class LogLineDecoder:
    def __init__(self, log_line):
        self.log_line = log_line
        self.tags = []
        self.message = ""
        self.layer_name = ""
        self.shape1 = []
        self.shape2 = []
        self.isCpuImplementation = False
        self.isXilinxImplementation = False
        self.is_printinfo = False
        self.is_reportduration = False
        self.device_elapsed_us = 0
        self.device_elapsed_ms = 0
        self.device_elapsed_s = 0

        self.decode_recursive()

    def decode_recursive(self):
        first_available_tag_start = self.log_line.find('[')
        first_available_tag_stop = self.log_line.find(']')
        if first_available_tag_start < first_available_tag_stop:
            self.tags.append(self.log_line[first_available_tag_start + 1:first_available_tag_stop])
            self.log_line = self.log_line[first_available_tag_stop + 1:]
            self.decode_recursive()
        else:
            self.message = self.log_line.rstrip()[1:]
            if "source CpuImplementation" in self.tags[2]:
                self.isCpuImplementation = True
                self.analyzed = CpuImplementationLogAnalyzer(self)
            else:
                if "source XilinxImplementation" in self.tags[2] or "source DataMover" in self.tags[2]:
                    self.isXilinxImplementation = True
                    self.analyzed = XilinxImplementationLogAnalyzer(self)


# SUB ANALYZER CLASS
class CpuImplementationLogAnalyzer:
    def __init__(self, decoder_obj: LogLineDecoder):
        self.obj = decoder_obj
        assert len(self.obj.tags) == 5
        assert "source CpuImplementation" in self.obj.tags[2]
        self.process()

    def process(self):
        # duration = tag_time1 - tag_time2
        self.tag_time = datetime.datetime.strptime(self.obj.tags[0], '%H:%M:%S.%f').time()
        if "PrintInfo" in self.obj.tags[3]:
            # This is a PrintInfo log line
            self.process_printinfo()
            self.obj.is_printinfo = True
        else:
            str = self.obj.tags[3]
            self.obj.layer_name = str[9:]  # function<space> = 9 characters

    def process_printinfo(self):
        def decode():
            def decode_shape(str):
                shape = str.split("x")
                int_shape = []
                for i in shape:
                    if i is not "":
                        int_shape.append(int(i))
                return int_shape

            self.obj.layer_name = self.obj.message[:self.obj.message.find(':')]
            message = self.obj.message[self.obj.message.find(':') + 1:]
            props = message.split(",")
            for prop in props:
                if prop.find('Shape1') != -1:
                    self.obj.shape1 = decode_shape(prop[prop.find('=') + 1:])
                else:
                    if prop.find('Shape2') != -1:
                        self.obj.shape2 = decode_shape(prop[prop.find('=') + 1:])

        def reform_log_line(line):
            tmp = line.replace(' ', '')
            tmp = tmp.replace('\t', '')
            tmp = tmp.replace('\n', '')
            tmp = tmp.replace(",,", ',')
            tmp = tmp.replace(":,", ':')
            tmp = tmp[2:]
            return tmp

        self.obj.message = reform_log_line(self.obj.message)
        decode()


# SUB ANALYZER CLASS
class XilinxImplementationLogAnalyzer:
    def __init__(self, decoder_obj: LogLineDecoder):
        self.obj = decoder_obj
        assert len(self.obj.tags) == 5
        assert "source XilinxImplementation" in self.obj.tags[2] or "source DataMover" in self.obj.tags[2]
        self.process()

    def process(self):
        # duration = tag_time1 - tag_time2
        self.tag_time = datetime.datetime.strptime(self.obj.tags[0], '%H:%M:%S.%f').time()
        if "PrintInfo" in self.obj.tags[3]:
            # This is a PrintInfo log line
            self.process_printinfo()
            self.obj.is_printinfo = True
        else:
            if "ReportDuration" in self.obj.tags[3]:
                def reform_log_line(line):
                    tmp = line.replace(',', '')
                    tmp = tmp.replace(' ', '')
                    tmp = tmp.replace('\t', '')
                    tmp = tmp.replace('\n', '')
                    tmp = tmp[2:]
                    return tmp

                # ** _PadUnpadLastDim(task):: (us): 27750344.0, (ms): 27750.344, (s): 27.750345
                self.obj.is_reportduration = True
                reportduration = reform_log_line(self.obj.message)
                self.obj.layer_name = reportduration[0:reportduration.find("::") - 6]
                str_duration_us = reportduration[reportduration.find("(us):") + 5:reportduration.find("(ms):")]
                self.obj.device_elapsed_us = float(str_duration_us)
                self.obj.device_elapsed_ms = self.obj.device_elapsed_us / 1000.0
                self.obj.device_elapsed_s = self.obj.device_elapsed_us / 1000000.0
            else:
                str = self.obj.tags[3]
                self.obj.layer_name = str[9:]  # function<space> = 9 characters

    def process_printinfo(self):
        def decode():
            def decode_shape(str):
                shape = str.split("x")
                int_shape = []
                for i in shape:
                    if i is not "":
                        int_shape.append(int(i))
                return int_shape

            self.obj.layer_name = self.obj.message[:self.obj.message.find(':')]
            message = self.obj.message[self.obj.message.find(':') + 1:]
            props = message.split(",")
            for prop in props:
                if prop.find('Shape1') != -1:
                    self.obj.shape1 = decode_shape(prop[prop.find('=') + 1:])
                else:
                    if prop.find('Shape2') != -1:
                        self.obj.shape2 = decode_shape(prop[prop.find('=') + 1:])

        def reform_log_line(line):
            tmp = line.replace(' ', '')
            tmp = tmp.replace('\t', '')
            tmp = tmp.replace('\n', '')
            tmp = tmp.replace(",,", ',')
            tmp = tmp.replace(":,", ':')
            tmp = tmp[2:]
            return tmp

        self.obj.message = reform_log_line(self.obj.message)
        decode()


class Layer:
    def __init__(self, name, start_time, is_cpu, is_fpga):
        self.layer_name = name
        self.start_time = start_time
        self.stop_time = 0
        self.is_cpu = is_cpu
        self.is_fpga = is_fpga
        self.sub_layers = []
        self.host_elapsed = 0
        self.device_elapsed_us = 0
        self.device_elapsed_ms = 0
        self.device_elapsed_s = 0
        self.shape1 = 0
        self.shape2 = 0
        self.msg = ''

    def add_stop_time(self, stop_time):
        self.stop_time = stop_time
        self.host_elapsed = datetime.datetime.combine(datetime.date.min, self.stop_time) - datetime.datetime.combine(
            datetime.date.min, self.start_time)

    def add_sublayer(self, obj):
        self.sub_layers.append(obj)

    def add_shape1(self, shape):
        self.shape1 = shape

    def add_shape2(self, shape):
        self.shape2 = shape

    def add_msg(self, msg):
        self.msg = msg

    def add_fpga_elapsed(self, us, ms, s):
        self.device_elapsed_us = us
        self.device_elapsed_ms = ms
        self.device_elapsed_s = s


# PARSING LOGS
class AnalyzeHostAndKernelLog:
    def __init__(self, fname):
        self.fname = fname
        self.layers = []

    def get_sublayer_depth(self, layer: Layer, depth):
        if depth > 1:
            return self.get_sublayer_depth(layer.sub_layers[-1], depth - 1)
        else:
            return layer.sub_layers[-1]

    def process(self):
        depth = -1
        with open(self.fname) as f:
            content = f.readlines()
            for line in content:
                obj = LogLineDecoder(line)

                # debug
                if obj.layer_name == '_PadUnpadLastDim':
                    aaa = 100

                if obj.message == 'Started':
                    depth = depth + 1
                    if depth is 0:
                        self.layers.append(
                            Layer(obj.layer_name,
                                  obj.analyzed.tag_time,
                                  obj.isCpuImplementation,
                                  obj.isXilinxImplementation)
                        )
                    if depth is 1:
                        _sub = Layer(obj.layer_name,
                                     obj.analyzed.tag_time,
                                     obj.isCpuImplementation,
                                     obj.isXilinxImplementation)
                        self.layers[-1].add_sublayer(_sub)

                    if depth >= 2:
                        _sub = Layer(obj.layer_name,
                                     obj.analyzed.tag_time,
                                     obj.isCpuImplementation,
                                     obj.isXilinxImplementation)
                        self.get_sublayer_depth(self.layers[-1], depth - 1).add_sublayer(_sub)

                else:
                    if obj.message == 'Finished':
                        if depth is 0:
                            self.layers[-1].add_stop_time(obj.analyzed.tag_time)
                        if depth >= 1:
                            self.get_sublayer_depth(self.layers[-1], depth).add_stop_time(obj.analyzed.tag_time)
                        depth = depth - 1
                    else:
                        if obj.is_printinfo:
                            if depth is 0:
                                self.layers[-1].add_shape1(obj.shape1)
                                self.layers[-1].add_shape2(obj.shape2)
                                self.layers[-1].add_msg(obj.message)
                            if depth >= 1:
                                self.get_sublayer_depth(self.layers[-1], depth).add_shape1(obj.shape1)
                                self.get_sublayer_depth(self.layers[-1], depth).add_shape2(obj.shape2)
                                self.get_sublayer_depth(self.layers[-1], depth).add_msg(obj.message)

                        if obj.is_reportduration:
                            if depth is 0:
                                self.layers[-1].add_fpga_elapsed(obj.device_elapsed_us, obj.device_elapsed_ms,
                                                                 obj.device_elapsed_s)
                            if depth >= 1:
                                self.get_sublayer_depth(self.layers[-1], depth).add_fpga_elapsed(obj.device_elapsed_us,
                                                                                                 obj.device_elapsed_ms,
                                                                                                 obj.device_elapsed_s)

            print("Done.")


def analyze(input_log_path):
    # Launch sequence decoder class: AnalyzeHostAndKernelLog
    # Line decoder class: LogLineDecoder
    # Sub decoder class for cpu: CpuImplementationLogAnalyzer

    analyzer = AnalyzeHostAndKernelLog(input_log_path)
    analyzer.process()
    return analyzer.layers


print("Please use analyzer.py, terminating...")
