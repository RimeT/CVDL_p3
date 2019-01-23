from __future__ import absolute_import

import mxnet as mx
from tqdm import tqdm
from gluoncv.data import batchify


class Model(object):

    def __init__(self, classes, ctx, logger):
        if ctx is None:
            ctx = [mx.gpu(i) for i in mx.test_utils.list_gpus()]
        if classes is None:
            self.classes = [str(i) for i in range(1000)]
        else:
            self.classes = classes
        self.ctx = ctx
        self.logger = logger

    def net_struct(self):
        raise NotImplementedError

    def class_loss(self):
        pass

    def val_eval_metric(self):
        pass

    @staticmethod
    def net_init():
        pass

    def custom_initialization(self, net):
        return False

    def print_train_stats(self, log_type, volume, week, epoch, epoch_num, batch_num, args_dict):
        curr_epoch = round(epoch_num * (epoch * week + batch_num) / (epoch_num * week), 2)
        log_str = ''
        if log_type == 1:
            log_str = self.format_train(curr_epoch, args_dict)
        elif log_type == 2:
            log_str = self.format_valid(curr_epoch, args_dict)
        self.logger.info(log_str)

    def format_train(self, epoch, args_dict):
        log_str = "Training (epoch " + str(epoch) + "): "
        params = []
        for arg in args_dict:
            arg_str = arg + " = " + "{:.6f}".format(args_dict[arg])
            params.append(arg_str)
        log_str = log_str + ", ".join(params)
        return log_str

    def format_valid(self, epoch, args_dict):
        log_str = "Validation (epoch " + str(epoch) + "): "
        params = []
        for arg in args_dict:
            arg_str = arg + " = " + "{:.6f}".format(args_dict[arg])
            params.append(arg_str)
        log_str = log_str + ", ".join(params)
        return log_str


class ClassificationModel(Model):

    def class_loss(self):
        raise NotImplementedError

    def mixup(self):
        raise NotImplementedError

    def label_smoothing(self):
        return False

    def train_eval_metric(self):
        pass

    def val_eval_metric(self):
        raise NotImplementedError


class DetectionModel(Model):

    def t_batchify_fn(self):
        raise NotImplementedError

    def v_batchify_fn(self):
        return batchify.Tuple(batchify.Stack(), batchify.Pad(pad_val=-1))

    def train_data_transform(self, **kwargs):
        raise NotImplementedError

    def val_data_transform(self):
        raise NotImplementedError

    def evaluations(self):
        raise NotImplementedError

    def train_batch(self, net, optimizer, epoch, batch_loader, lr_scheduler, batch_size, ctx, log_interval, print_log):
        raise NotImplementedError

    @staticmethod
    def dynamic_bar(bar, epoch, batch, **kwargs):
        if isinstance(bar, tqdm):
            params = []
            for arg in kwargs:
                arg_str = arg + "=" + "{:.6f}".format(kwargs[arg])
                params.append(arg_str)
            bar.set_description("Training: EPOCH[{}] batch[{}] {}".format(epoch, batch, ", ".join(params)))
