import os
from collections import OrderedDict

import pydicom


def ct_series_sort(dcm_folder):
    series_set = set()
    ct_series = OrderedDict()
    for dcm_file in os.listdir(dcm_folder):
        layer = pydicom.dcmread(os.path.join(dcm_folder, dcm_file))
        if layer.SeriesNumber not in series_set:
            series_set.add(layer.SeriesNumber)
            ins = layer.InstanceNumber
            series_dict = {ins: dcm_file}
            ct_series[layer.SeriesNumber] = series_dict
        else:
            ins = layer.InstanceNumber
            ct_series[layer.SeriesNumber][ins] = dcm_file
    if len(series_set) == 0:
        raise ValueError("Series Number not found in %s" % dcm_folder)
    elif len(series_set) > 1:
        raise ValueError("Series should be 1, but %d series found in %s" % (len(series_set), dcm_folder))
    return ct_series[series_set.pop()]
