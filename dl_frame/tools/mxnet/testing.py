import argparse
import json
import logging
import os
import re
from time import strftime, gmtime
from test_factory import get_test_frame

import mxnet as mx

parser = argparse.ArgumentParser(description="InferScholar MXNET Testing Module.")
parser.add_argument('--job-dir',
                    type=str,
                    help="Job root directory.")
parser.add_argument('--data-format',
                    type=str,
                    help="dicom or normal")
parser.add_argument('--multi-slices',
                    type=bool)
parser.add_argument('--dcm-slices',
                    type=int)
parser.add_argument('--data-path',
                    type=str)
parser.add_argument('--data-root',
                    type=str,
                    default='')
parser.add_argument('--config-file',
                    type=str)
parser.add_argument('--model-file',
                    type=str)
parser.add_argument('--param-file',
                    type=str)
parser.add_argument('--result-type',
                    type=int)
parser.add_argument('--gpus',
                    type=str)

opt = parser.parse_args()


def _load_model(net, param_path, ctx):
    pass


def testing():
    # load config
    if not os.path.isfile(opt.config_file):
        raise ValueError("{} not found".format(opt.config_file))
    with open(opt.config_file, 'r') as cf:
        dl_config = json.load(cf)
    if not opt.job_dir:
        os.mkdir(opt.job_dir)
    job_dir = os.path.join(opt.job_dir, "test_{}".format(strftime(strftime("%Y%m%d%H%M%S", gmtime()))))
    os.mkdir(job_dir)

    # load custom model
    exec (open(opt.model_file).read(), globals())
    try:
        CustomModel
    except NameError:
        print("CustomModel is not defined.")
        exit(-1)

    gpus = filter(str.isdigit, re.split(',|_|-', opt.gpus))
    if len(gpus) > 0:
        gpus = [int(x) for x in gpus]
    # model
    task = get_test_frame(dl_config['dl_type'], CustomModel, job_dir, gpus)
    # dataloader
    if dl_config['dl_type'] == 'classification':
        pass
    elif dl_config['dl_type'] == 'object-detection':
        task.create_dataloader(opt.data_path, labels=dl_config['labels'], data_format=opt.data_format,
                               data_root=opt.data_root,
                               window_center=dl_config['window_center'],
                               window_width=dl_config['window_width'],
                               multi_slices=opt.multi_slices,
                               dcm_slices=opt.dcm_slices)
    task.net_config(opt.param_file)
    task.data_setup(resize_type=dl_config["resize_type"],
                    r_width=dl_config["width"],
                    r_height=dl_config["height"],
                    window_center=dl_config["window_center"],
                    window_width=dl_config["window_width"],
                    channels=dl_config['channels'])
    task.start_test(result_type=opt.result_type)


if __name__ == "__main__":
    testing()
