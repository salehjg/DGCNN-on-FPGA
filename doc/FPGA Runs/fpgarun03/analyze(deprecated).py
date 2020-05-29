import numpy as np
import pandas as pd
import sys


class KernelLog:
    def __init__(self, log_line):
        self.log_line = log_line
        self.duration_us = 0
        self.duration_ms = 0
        self.duration_s = 0
        self.decode_kernel_log(log_line)

    @staticmethod
    def reform_log_line(line):
        tmp = line.replace(' ', '')
        tmp = tmp.replace('\t', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp[2:]
        return tmp

    def decode_kernel_log(self, log_line):
        log_reformed = self.reform_log_line(log_line)
        self.kernel_name = log_reformed[0:log_reformed.find("::")]
        str_duration_us = log_reformed[log_reformed.find("(us):") + 5:log_reformed.find("(ms):")]
        self.duration_us = float(str_duration_us)
        self.duration_ms = self.duration_us / 1000.0
        self.duration_s = self.duration_us / 1000000.0


class LayerLog:
    def __init__(self, log_line):
        self.log_line = log_line
        self.Shape1 = ""
        self.Shape2 = ""
        self.decode_layer_log(log_line)

    def decode_layer_log(self, line):
        log_reformed = self.reform_log_line(line)
        self.layer_name = log_reformed[0:log_reformed.find(":")]
        log_reformed_name_removed = log_reformed[log_reformed.find(":") + 1:]
        self.layer_properties = log_reformed_name_removed
        self.decode_properties()

    def decode_properties(self):
        props = self.layer_properties.split(",")
        for prop in props:
            if prop.find('Shape1') != -1:
                self.Shape1 = prop[prop.find('=') + 1:]
            else:
                if prop.find('Shape2') != -1:
                    self.Shape2 = prop[prop.find('=') + 1:]

    def reform_log_line(self, line):
        tmp = line.replace(' ', '')
        tmp = tmp.replace('\t', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp.replace(",,", ',')
        tmp = tmp.replace(":,", ':')
        tmp = tmp[2:]
        return tmp

    def add_kernel_log_for_this_layer(self, line):
        self.kernel = KernelLog(line)


def analyze_host_log(filename):
    with open(filename) as f:
        content = f.readlines()
        layer_launches = []
        state = 1
        pending = []
        for line in content:
            if state == 1:
                if line.startswith("		**"):
                    # main layer log found
                    layer_launches.append(LayerLog(line))
                    state = 2
                else:
                    if line.startswith("	**"):
                        # This should be a pending kernel log
                        pending[-1].add_kernel_log_for_this_layer(line)
                        # assert len(pending) <= 10
                        layer_launches.append(pending.pop())
                        state = 1
            else:
                if state == 2:
                    if line.startswith("	**"):
                        # kernel launch log found
                        layer_launches[-1].add_kernel_log_for_this_layer(line)
                        state = 1
                    else:
                        if line.startswith("		**"):
                            # extra layer log
                            new_log_tmp = LayerLog(line)
                            last_name = layer_launches[-1].layer_name
                            new_name = new_log_tmp.layer_name
                            if new_name == last_name or new_name == "ReluSqrtSquare":
                                layer_launches.pop()
                                layer_launches.append(new_log_tmp)
                            else:
                                pending.append(layer_launches.pop())  # remove last item
                                layer_launches.append(new_log_tmp)

    return layer_launches, pending


def main():
    if len(sys.argv)!=3:
        print("Error, python3 analyze.py  <path to the host log txt>  <path to the output xlsx file>")
        sys.exit(1)
    print("Decoding ModelArch04's host log for an fpga-run(hw)...")
    decoded_launches, incomplete_launches = analyze_host_log(sys.argv[1])
    print("Decoded Kernel Launches: ", len(decoded_launches))
    print("Incomplete Kernel Launches: ", len(incomplete_launches))

    print("Sorting By Name...")
    layers = {}
    for lnch in decoded_launches:
        is_already_added = False
        keys = layers.keys()
        for key in keys:
            if key == lnch.layer_name:
                is_already_added = True

        if is_already_added:
            layers[lnch.layer_name].append(lnch)
        else:
            layers[lnch.layer_name] = [lnch]

    print("\n=====================================")
    layer_names = list(layers.keys())
    total_duration = 0
    for layer_name in layer_names:
        print(layer_name, ": Launches: ", len(layers[layer_name]))
        durations = []
        for lnch in layers[layer_name]:
            durations.append(lnch.kernel.duration_ms)
        print("Durations(ms): ", np.round(durations, decimals=2))
        print("Ave Durations(ms): ", np.round(np.mean(durations), decimals=2))
        total_duration += np.sum(durations)
        print("\n-------------------------------------")
    print("Total(ms): ", total_duration)

    print("\n=====================================")
    print("Exporting data into an excel file...")

    writer = pd.ExcelWriter(sys.argv[2], engine='xlsxwriter')

    for layer_name in layer_names:
        local_dataframe = {}
        index = 0
        local_dataframe['Options'] = []
        local_dataframe['Shape1'] = []
        local_dataframe['Shape2'] = []
        local_dataframe['Time(us)'] = []
        local_dataframe['Time(ms)'] = []
        local_dataframe['Time(s)'] = []
        for lnch in layers[layer_name]:
            local_dataframe['Options'].append(lnch.layer_properties)
            local_dataframe['Shape1'].append(lnch.Shape1)
            local_dataframe['Shape2'].append(lnch.Shape2)
            local_dataframe['Time(us)'].append(lnch.kernel.duration_us)
            local_dataframe['Time(ms)'].append(lnch.kernel.duration_ms)
            local_dataframe['Time(s)'].append(lnch.kernel.duration_s)
            index = index + 1

        df = pd.DataFrame(local_dataframe)
        df.to_excel(writer, sheet_name=layer_name)

    print("Creating the summary sheet...")
    summary_dataframe = {}
    summary_dataframe['Name'] = []
    summary_dataframe['Launches'] = []
    summary_dataframe['Time(ms)'] = []
    for layer_name in layer_names:
        summary_dataframe['Name'].append(layer_name)
        msec = []
        for lnch in layers[layer_name]:
            msec.append(lnch.kernel.duration_ms)
        summary_dataframe['Time(ms)'].append(np.sum(msec))
        summary_dataframe['Launches'].append(len(msec))

    df = pd.DataFrame(summary_dataframe)
    df.to_excel(writer, sheet_name="Summary")

    print("Creating the excel chart...")
    workbook = writer.book
    worksheet = writer.sheets["Summary"]

    # =======================================================================================
    # Summary Chart 1: Total Elapsed Time Per Kernel
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series(
        {'values': ''.join(['=', "Summary", '!$D$2:$D$', str(len(layer_names))]),
         'categories': ''.join(['=', 'Summary', '!$B$2:$B$', str(len(layer_names))])}
    )
    chart.set_title({'name': 'Total Elapsed Time Per Kernel',
                     'name_font': {'name': 'Calibri(Body)', 'size': 14}})
    worksheet.insert_chart('H2', chart)

    # =======================================================================================
    # Summary Chart 2: Launches Per Kernel
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series(
        {'values': ''.join(['=', "Summary", '!$C$2:$C$', str(len(layer_names))]),
         'categories': ''.join(['=', 'Summary', '!$B$2:$B$', str(len(layer_names))])}
    )
    chart.set_title({'name': 'Launches Per Kernel',
                     'name_font': {'name': 'Calibri(Body)', 'size': 14}})
    worksheet.insert_chart('H20', chart)

    # =======================================================================================
    writer.save()


main()
