from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
from configparser import ConfigParser
from time import time, strftime, gmtime

from model_factory import get_model_frame


def load_config(ini_path):
    cp = ConfigParser()
    cp.read(ini_path)
    return cp


def load_mx_config():
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    config_dir = os.path.join(project_path, 'config')
    config_file = "mx_det_config.ini"
    config = load_config(os.path.join(config_dir, config_file))
    return config


parser = argparse.ArgumentParser(description="InferScholar MXNET Training Module.")
parser.add_argument('--directory',
                    type=str,
                    help="Job directory.")
parser.add_argument('--snap-prefix',
                    type=str)
parser.add_argument('--data-format',
                    type=str)
parser.add_argument('--multi-slices',
                    type=bool)
parser.add_argument('--dcm-slices',
                    type=int,
                    default=3)
parser.add_argument('--label-txt',
                    type=str)
parser.add_argument('--train-path',
                    type=str)
parser.add_argument('--t-root',
                    type=str,
                    default=None)
parser.add_argument('--val-path',
                    type=str,
                    default=None)
parser.add_argument('--v-root',
                    type=str,
                    default=None)
parser.add_argument('--model-path',
                    type=str)
parser.add_argument('--param-path',
                    type=str,
                    default=None)
parser.add_argument('--cast',
                    type=str,
                    default='float32')
parser.add_argument('--resize-type',
                    type=int,
                    default=1)
parser.add_argument('--width',
                    type=int)
parser.add_argument('--height',
                    type=int)
parser.add_argument('--interpolation',
                    type=int,
                    default=1)
parser.add_argument('--crop-width',
                    type=int,
                    default=0)
parser.add_argument('--crop-height',
                    type=int,
                    default=0)
parser.add_argument('--norm-mean',
                    type=str,
                    default='0.45,0.45,0.45')
parser.add_argument('--norm-std',
                    type=str,
                    default='0.45,0.45,0.45')
parser.add_argument('--window-center',
                    type=int)
parser.add_argument('--window-width',
                    type=int,
                    default=0)
parser.add_argument('--rand-flipping',
                    type=int)
parser.add_argument('--rrc-size',
                    type=str)
parser.add_argument('--rrc-scale',
                    type=str)
parser.add_argument('--rrc-ratio',
                    type=str)
parser.add_argument('--rand-brightness',
                    type=float)
parser.add_argument('--rand-contrast',
                    type=float)
parser.add_argument('--rand-saturation',
                    type=float)
parser.add_argument('--rand-hue',
                    type=float)
parser.add_argument('--rand-lighting',
                    type=float,
                    help="AlexNet-style PCA-based noise to an image")
parser.add_argument('--ml-type',
                    type=str)
parser.add_argument('--gpus',
                    type=str)
parser.add_argument('--shuffle',
                    type=bool)
parser.add_argument('--snap-interval',
                    type=int,
                    default=10)
parser.add_argument('--val-interval',
                    type=int,
                    default=10)
parser.add_argument('--epochs',
                    type=int,
                    default=0)
parser.add_argument('--batch-size',
                    type=int,
                    default=1)
parser.add_argument('--optimizer',
                    type=str,
                    default='adam')
parser.add_argument('--wd',
                    type=float,
                    default=.0)
parser.add_argument('--opt-param',
                    type=str,
                    default='')
parser.add_argument('--lr-mode',
                    type=str,
                    default='poly')
parser.add_argument('--base-lr',
                    type=float,
                    default=0.0001)
parser.add_argument('--lr-steps',
                    type=str,
                    default='')
parser.add_argument('--step-factor',
                    type=float)
parser.add_argument('--power',
                    type=float)
parser.add_argument('--target-lr',
                    type=float,
                    default=1e-12)
parser.add_argument('--warmup-lr',
                    type=float,
                    default=1e-12)
parser.add_argument('--warmup-epochs',
                    type=int,
                    default=10)
parser.add_argument('--warmup-mode',
                    type=str,
                    default='constant')

opt = parser.parse_args()


def training():
    ####################################################
    # ---------------load ML parameters--------------- #
    ####################################################

    # config = load_mx_config()
    ml_type = opt.ml_type
    # -job
    job_dir = opt.directory
    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)
    job_dir = os.path.join(job_dir, "train_{}".format(strftime(strftime("%Y%m%d%H%M%S", gmtime()))))
    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)
    snap_prefix = opt.snap_prefix
    # -data
    data_format = opt.data_format
    multi_slices = opt.multi_slices
    dcm_slices = opt.dcm_slices
    label_txt = opt.label_txt
    t_datapath = opt.train_path
    t_root = opt.t_root
    v_datapath = opt.val_path
    v_root = opt.v_root
    custom_model_path = opt.model_path
    param_path = opt.param_path
    # -hyper-param
    gpus = [int(i) for i in opt.gpus.split(',')]
    snap_interval = opt.snap_interval
    val_interval = opt.val_interval
    epoch_num = opt.epochs
    shuffle = opt.shuffle
    batch_size = opt.batch_size
    if len(gpus) > 0:
        batch_size *= len(gpus)
    optimizer = opt.optimizer
    opt_param = json.loads(opt.opt_param) if len(
        opt.opt_param) > 0 else dict()
    lr_mode = opt.lr_mode
    base_lr = opt.base_lr
    lr_steps = opt.lr_steps
    if len(lr_steps) > 0:
        lr_steps = [int(i) for i in lr_steps.split(',')]
    step_factor = opt.step_factor
    power = opt.power
    target_lr = opt.target_lr
    warmup_lr = opt.warmup_lr
    warmup_epochs = opt.warmup_epochs
    warmup_mode = opt.warmup_mode
    wd = opt.wd
    # -transform
    cast = opt.cast
    resize_type = opt.resize_type
    width = opt.width
    height = opt.height
    interpolation = opt.interpolation
    crop_width = opt.crop_width
    crop_height = opt.crop_height
    is_crop = crop_width and crop_height
    norm_mean = opt.norm_mean
    norm_mean = None if len(norm_mean) < 1 else norm_mean
    if norm_mean:
        norm_mean = tuple(float(i) for i in norm_mean.split(','))
    norm_std = opt.norm_std
    norm_std = None if len(norm_std) < 1 else norm_std
    if norm_std:
        norm_std = tuple(float(i) for i in norm_std.split(','))
    window_center = opt.window_center
    window_width = opt.window_width
    # -augmentation
    rand_flip = opt.rand_flipping
    rrc_size = opt.rrc_size
    if len(rrc_size) > 0:
        rrc_size = tuple(int(i) for i in rrc_size.split(','))
    else:
        rrc_size = (width, height)
    rrc_scale = opt.rrc_scale
    if len(rrc_scale) > 0:
        rrc_scale = tuple(float(i) for i in rrc_scale.split(','))
    else:
        rrc_scale = (1, 1)
    rrc_ratio = opt.rrc_ratio
    if len(rrc_ratio) > 0:
        rrc_ratio = tuple(float(i) for i in rrc_ratio.split(','))
    else:
        rrc_ratio = (1, 1)
    rand_brightness = opt.rand_brightness
    rand_contrast = opt.rand_contrast
    rand_saturation = opt.rand_saturation
    rand_hue = opt.rand_hue
    rand_lighting = opt.rand_lighting

    ####################################################
    # --------------------training-------------------- #
    ####################################################

    # load custom model
    exec (open(custom_model_path).read(), globals())
    try:
        CustomModel
    except NameError:
        print("CustomModel is not defined")
        exit(-1)

    # get model
    model = get_model_frame(ml_type, CustomModel, job_dir, gpus)
    t1 = time()
    if os.path.isfile(label_txt):
        with open(label_txt, 'r') as label_f:
            labels = [l.strip() for l in label_f.readlines()]
    else:
        raise ValueError('labels file can not be accessed')

    # dataloader
    if ml_type == 'classification':
        model.create_dataloader(t_datapath, v_datapath, data_format=data_format, labels=labels,
                                window_center=window_center,
                                window_width=window_width)
    elif ml_type == 'object-detection':
        model.create_dataloader(t_datapath, v_datapath, labels=labels, data_format=data_format,
                                t_root=t_root, v_root=v_root,
                                window_center=window_center,
                                window_width=window_width,
                                multi_slices=multi_slices,
                                dcm_slices=dcm_slices)
    elif ml_type == 'segmentation':
        pass
    else:
        raise ValueError("Unkown training type")
    print('create datasets costs={}'.format(time() - t1))
    t1 = time()
    model.net_config(param_path)
    print('net configuration costs={}'.format(time() - t1))
    t1 = time()
    model.data_setup(batch_size, shuffle,
                     resize_type=resize_type,
                     r_width=width,
                     r_height=height,
                     c_width=crop_width,
                     c_height=crop_height,
                     flip_type=rand_flip,
                     rand_brightness=rand_brightness,
                     rand_contrast=rand_contrast,
                     rand_saturation=rand_saturation,
                     rand_hue=rand_hue,
                     rrc_size=rrc_size,
                     rrc_scale=rrc_scale,
                     rrc_ratio=rrc_ratio,
                     window_center=window_center,
                     window_width=window_width,
                     channels=model.t_loader.channels)
    print('data_setup costs={}'.format(time() - t1))
    t1 = time()
    model.trainer_config(optimizer, lr_mode, base_lr, epoch_num, wd, opt_param,
                         step=lr_steps,
                         step_factor=step_factor,
                         target_lr=target_lr,
                         power=power,
                         warmup_epochs=warmup_epochs,
                         warmup_lr=warmup_lr,
                         warmup_mode=warmup_mode)
    print('trainer_config costs={}'.format(time() - t1))

    # save info
    info_path = os.path.join(job_dir, "dl_info.json")
    dl_info = {"dl_frame": "mxnet", 'dl_type': ml_type, 'resize_type': resize_type, 'width': width, 'height': height,
               'window_center': window_center, "window_width": window_width, "channels": model.t_loader.channels,
               "labels": labels}
    with open(info_path, 'w') as info_f:
        json.dump(dl_info, info_f, allow_nan=False)
        print("dl info file write to %s" % info_path)
    # do training
    model.start_train(epoch_num, os.path.join(job_dir, snap_prefix),
                      snap_interval, val_interval)


if __name__ == "__main__":
    training()
