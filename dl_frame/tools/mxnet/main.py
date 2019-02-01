from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def main():
    ####################################################
    # ---------------load ML parameters--------------- #
    ####################################################

    config = load_mx_config()
    ml_type = config['hyper-param']['ml_type']
    # -job
    job_dir = config['job']['directory']
    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)
    job_dir = os.path.join(job_dir, strftime(strftime("%Y%m%d%H%M%S", gmtime())))
    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)
    snap_prefix = config['job']['snap_prefix']
    # -data
    data_format = config['data']['data_format']
    multi_slices = config['data']['multi_slices']
    if multi_slices == 'True':
        multi_slices = True
    else:
        multi_slices = False
    dcm_slices = int(config['data']['dcm_slices'])
    accelerate = int(config['data']['accelerate'])
    label_txt = config['data']['label_txt']
    t_datapath = config['data']['train_path']
    t_root = config['data']['t_root']
    v_datapath = config['data']['val_path']
    v_root = config['data']['v_root']
    v_datapath = None if len(v_datapath) < 1 else v_datapath
    custom_model_path = config['data']['model_path']
    param_path = config['data']['param_path']
    param_path = None if len(param_path) < 1 else param_path
    # -hyper-param
    gpus = [int(i) for i in config['hyper-param']['gpus'].split(',')]
    # gpus = [i for i in mx.test_utils.list_gpus()]
    snap_interval = int(config['hyper-param']['snap_interval'])
    val_interval = int(config['hyper-param']['val_interval'])
    epoch_num = int(config['hyper-param']['epochs'])
    shuffle = True if config['hyper-param']['shuffle'] == 'True' else False
    batch_size = int(config['hyper-param']['batch_size'])
    if len(gpus) > 0:
        batch_size *= len(gpus)
    optimizer = config['hyper-param']['optimizer']
    opt_param = json.loads(config['hyper-param']['opt_param']) if len(
        config['hyper-param']['opt_param']) > 0 else dict()
    lr_mode = config['hyper-param']['lr_mode']
    base_lr = float(config['hyper-param']['base_lr'])
    lr_steps = config['hyper-param']['lr_steps']
    if len(lr_steps) > 0:
        lr_steps = [int(i) for i in lr_steps.split(',')]
    step_factor = config['hyper-param']['step_factor']
    step_factor = float(step_factor) if len(step_factor) > 0 else 0.1
    power = config['hyper-param']['power']
    power = float(power) if len(power) > 0 else 0.9
    target_lr = config['hyper-param']['target_lr']
    target_lr = float(target_lr) if len(target_lr) > 0 else 0.00001
    warmup_lr = float(config['hyper-param']['warmup_lr'])
    warmup_epochs = int(config['hyper-param']['warmup_epochs'])
    warmup_mode = config['hyper-param']['warmup_mode']
    wd = float(config['hyper-param']['wd'])
    # -transform
    cast = config['transformation']['cast']
    resize_type = int(config['transformation']['resize_type'])
    width = int(config['transformation']['width'])
    height = int(config['transformation']['height'])
    interpolation = int(config['transformation']['interpolation'])
    crop_width = config['transformation']['crop_width']
    crop_width = int(crop_width) if len(crop_width) > 0 and crop_width is not '0' else None
    crop_height = config['transformation']['crop_height']
    crop_height = int(crop_height) if len(crop_height) > 0 and crop_height is not '0' else None
    is_crop = crop_width and crop_height
    norm_mean = config['transformation']['norm_mean']
    norm_mean = None if len(norm_mean) < 1 else norm_mean
    if norm_mean:
        norm_mean = tuple(float(i) for i in norm_mean.split(','))
    norm_std = config['transformation']['norm_std']
    norm_std = None if len(norm_std) < 1 else norm_std
    if norm_std:
        norm_std = tuple(float(i) for i in norm_std.split(','))
    window_center = int(config['transformation']['window_center'])
    window_width = int(config['transformation']['window_width'])
    # -augmentation
    rand_flip = int(config['augmentation']['rand_flipping'])
    rrc_size = config['augmentation']['rrc_size']
    if len(rrc_size) > 0:
        rrc_size = tuple(int(i) for i in rrc_size.split(','))
    else:
        rrc_size = (width, height)
    rrc_scale = config['augmentation']['rrc_scale']
    if len(rrc_scale) > 0:
        rrc_scale = tuple(float(i) for i in rrc_scale.split(','))
    else:
        rrc_scale = (1, 1)
    rrc_ratio = config['augmentation']['rrc_ratio']
    if len(rrc_ratio) > 0:
        rrc_ratio = tuple(float(i) for i in rrc_ratio.split(','))
    else:
        rrc_ratio = (1, 1)
    rand_brightness = float(config['augmentation']['rand_brightness'])
    rand_contrast = float(config['augmentation']['rand_contrast'])
    rand_saturation = float(config['augmentation']['rand_saturation'])
    rand_hue = float(config['augmentation']['rand_hue'])
    rand_lighting = float(config['augmentation']['rand_lighting'])

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
                     channels=model.t_loader.channels,
                     accelerate=accelerate)
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
    model.start_train(epoch_num, os.path.join(job_dir, snap_prefix),
                      snap_interval, val_interval)


if __name__ == "__main__":
    main()
