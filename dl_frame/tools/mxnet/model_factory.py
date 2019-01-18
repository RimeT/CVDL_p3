from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from time import time

import mxnet as mx
import numpy as np
from data.mx_data import LoaderFactory
from gluoncv.data import batchify
from gluoncv.utils import LRScheduler
from mxnet import autograd, nd
from mxnet.gluon import Trainer
from mxnet.gluon.data.vision import transforms as gdt
from mxnet.gluon.utils import split_and_load
from mxnet.initializer import Initializer
from tqdm import tqdm


def format_train(epoch, args_dict):
    log_str = "Training (epoch " + str(epoch) + "): "
    params = []
    for arg in args_dict:
        arg_str = arg + " = " + "{:.6f}".format(args_dict[arg])
        params.append(arg_str)
    log_str = log_str + ", ".join(params)
    return log_str


def format_valid(epoch, args_dict):
    log_str = "Validation (epoch " + str(epoch) + "): "
    params = []
    for arg in args_dict:
        arg_str = arg + " = " + "{:.6f}".format(args_dict[arg])
        params.append(arg_str)
    log_str = log_str + ", ".join(params)
    return log_str


def print_train_stats(logger, log_type, week, epoch, epoch_num, batch_num, args_dict):
    curr_epoch = round(epoch_num * (epoch * week + batch_num) / (epoch_num * week), 2)
    log_str = ''
    if log_type == 1:
        log_str = format_train(curr_epoch, args_dict)
    elif log_type == 2:
        log_str = format_valid(curr_epoch, args_dict)
    logger.info(log_str)


def print_snapshot_stats(logger, snap_pf, epoch):
    snapshot_path = str("%s-%04d.params" % (snap_pf, epoch))
    logger.info('Snapshotting to %s', snapshot_path)


class ModelFactory(object):
    def __init__(self, custom_model, job_dir, gpus) -> None:
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(job_dir, 'mxnet_training.log'))
        fh.setLevel(logging.DEBUG)
        # stream handler will send message to stdout
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)
        self.custom_model = custom_model
        self.custom_obj = None
        self.net = None
        self.class_loss = None
        self.eval_metric = None
        self.lr_scheduler = None
        self.optimizer = None
        self.job_dir = job_dir
        self.log_interval = 20
        self.ctx = [mx.gpu(i) for i in gpus]
        self.t_loader = None
        self.v_loader = None

    def create_dataloader(self, train_db, valid_db, data_format, **kwargs):
        raise NotImplementedError

    def data_setup(self, batch_size, shuffle, **kwargs):
        raise NotImplementedError

    def net_config(self, param_path):
        self.custom_obj = self.custom_model(self.t_loader.classes, self.ctx, self.logger)
        self.net = self.custom_obj.net_struct()
        custom_init = self.custom_obj.custom_initialization(self.net)
        if custom_init:
            # self.net.collect_params().reset_ctx(ctx=self.ctx)
            print('Custom initialing')
        else:
            if param_path:
                print('Loading net params...')
                self.net.collect_params().load(param_path, ctx=self.ctx)
                # self.net.collect_params().load(param_path)
            else:
                net_init = self.custom_model.net_init()
                if isinstance(net_init, Initializer):
                    print('Initializing net with custom function')
                    self.net.initialize(net_init, ctx=self.ctx)
                    # self.net.initialize(net_init, force_reinit=True)
                else:
                    print('Custom initialization access failed, net will be initialized with Xavier')
                    self.net.initialize(mx.init.Xavier(), ctx=self.ctx)
                    # self.net.initialize(mx.init.Xavier(), force_reinit=True)
        self.class_loss = self.custom_obj.class_loss()
        self.eval_metric = self.custom_obj.eval_metric()

    def trainer_config(self, optimizer, lr_mode, base_lr, epoch_num, wd, opt_param, **kwargs):
        niters = self.t_loader.niters
        assert (niters > 0, 'niters is 0')
        assert ('step' in kwargs)
        assert ('step_factor' in kwargs)
        assert ('target_lr' in kwargs)
        assert ('power' in kwargs)
        assert ('warmup_epochs' in kwargs)
        assert ('warmup_lr' in kwargs)
        assert ('warmup_mode' in kwargs)
        assert (isinstance(wd, float))
        assert (isinstance(opt_param, dict))
        self.lr_scheduler = LRScheduler(lr_mode, base_lr, niters, epoch_num,
                                        step=kwargs['step'],
                                        step_factor=kwargs['step_factor'],
                                        targetlr=kwargs['target_lr'],
                                        power=kwargs['power'],
                                        warmup_epochs=kwargs['warmup_epochs'],
                                        warmup_lr=kwargs['warmup_lr'],
                                        warmup_mode=kwargs['warmup_mode'])
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd': wd}
        for k in opt_param:
            optimizer_params[k] = opt_param[k]
        self.optimizer = Trainer(self.net.collect_params(), optimizer, optimizer_params)

    def start_train(self, epoch_num, snap_pf, snap_itv, valid_itv):
        raise NotImplementedError

    def validation(self, net):
        raise NotImplementedError

    @staticmethod
    def log_format(logger, epoch, batch, speed, lr, params):
        param_list = []
        for k in params:
            param_list.append("%s=%.6f" % (k, params[k]))
        param_str = " ".join(param_list)
        logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s\tlr=%f' % (
            epoch, batch, speed, param_str, lr))


class Classification(ModelFactory):

    def __init__(self, custom_model, job_dir, gpus) -> None:
        super(Classification, self).__init__(custom_model, job_dir, gpus)

    def create_dataloader(self, train_db, valid_db, **kwargs):
        self.t_loader = LoaderFactory.set_source('classification', train_db, **kwargs)
        if valid_db:
            self.v_loader = LoaderFactory.set_source('classification', valid_db, **kwargs)

    def data_setup(self, batch_size, shuffle, **kwargs):
        resizer = None
        if kwargs['resize_type'] == 1:
            resizer = gdt.Resize((kwargs['r_width'], kwargs['r_height']))
        if kwargs['c_width']:
            resizer = gdt.Compose([
                resizer,
                gdt.CenterCrop((kwargs['c_width'], kwargs['c_height']))
            ])
        random_flip = None
        if kwargs['flip_type'] == 1:
            random_flip = gdt.RandomFlipLeftRight()
        elif kwargs['flip_type'] == 2:
            random_flip = gdt.RandomFlipTopBottom()
        elif kwargs['flip_type'] == 3:
            random_flip = gdt.Compose([
                gdt.RandomFlipLeftRight(),
                gdt.RandomFlipTopBottom()
            ])
        random_color_jitter = gdt.RandomColorJitter(brightness=kwargs['rand_brightness'],
                                                    contrast=kwargs['rand_contrast'],
                                                    saturation=kwargs['rand_saturation'],
                                                    hue=kwargs['rand_hue'])
        random_resize_crop = gdt.RandomResizedCrop(kwargs['rrc_size'], kwargs['rrc_scale'], kwargs['rrc_ratio'])
        t_transformer = gdt.Compose([
            # random_resize_crop,
            resizer,
            random_flip,
            random_color_jitter,
        ])
        # custom_t_fn = self.custom_model.train_data_transform()
        self.t_loader.setup(batch_size, shuffle, t_transformer)
        if self.v_loader:
            # custom_v_fn = self.custom_model.val_data_transform()
            self.v_loader.setup(batch_size, False, resizer)

    def start_train(self, epoch_num, snap_pf, snap_itv, valid_itv):
        # t1 = time()
        # self.net.collect_params().reset_ctx(self.ctx)
        # print("reset net params to {}, time costs={}".format(self.ctx, time() - t1))
        self.logger.info("Classification start training")
        smoothing_constant = .01
        batch_size = self.t_loader.batch_size
        for epoch in range(epoch_num):
            btic = time()
            tbar = tqdm(self.t_loader.batch_loader)
            self.eval_metric.reset()
            self.net.hybridize()
            for idx, (data, label) in enumerate(tbar):
                data_list = split_and_load(data, ctx_list=self.ctx)
                label_list = split_and_load(label, ctx_list=self.ctx)
                with autograd.record(True):
                    outputs = [self.net(X.astype('float32', copy=False)) for X in data_list]
                    losses = [self.class_loss(o, ll.astype('float32', copy=False)) for (o, ll) in
                              zip(outputs, label_list)]
                for l in losses:
                    l.backward()
                self.lr_scheduler.update(idx, epoch)
                self.optimizer.step(batch_size)
                self.eval_metric.update(label_list, outputs)
                curr_loss = [nd.mean(l).asscalar() for l in losses]
                curr_loss = np.mean(curr_loss)
                moving_loss = (curr_loss if ((idx == 0) and (epoch == 0))
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

                # log
                tbar.set_description("epoch %d batch %d loss=%f" % (epoch, idx, moving_loss))
                if not idx % self.log_interval:
                    params = dict()
                    train_metric_name, train_metic_score = self.eval_metric.get()
                    params[train_metric_name] = train_metic_score
                    params['loss'] = moving_loss
                    speed = batch_size * self.log_interval / (time() - btic)
                    self.log_format(self.logger, epoch, idx, speed, self.optimizer.learning_rate, params)
                    btic = time()

    def validation(self, net):
        self.logger.info("validation to be continued")
        print("validation to be continued")


class Detection(ModelFactory):

    def __init__(self, custom_model, job_dir, gpus) -> None:
        super(Detection, self).__init__(custom_model, job_dir, gpus)

    def create_dataloader(self, train_db, valid_db, **kwargs):
        self.t_loader = LoaderFactory.set_source('object-detection', train_db, root=kwargs['t_root'], **kwargs)
        if valid_db:
            self.v_loader = LoaderFactory.set_source('object-detection', valid_db, root=kwargs['v_root'], **kwargs)

    def data_setup(self, batch_size, shuffle, **kwargs):
        t_batchify_fn = self.custom_obj.t_batchify_fn()
        t_fn = self.custom_obj.train_data_transform(net=self.net, **kwargs)
        self.t_loader.setup(batch_size=batch_size, shuffle=shuffle, fn=t_fn, batchify_fn=t_batchify_fn, **kwargs)
        if self.v_loader:
            v_batchify_fn = self.custom_obj.v_batchify_fn()
            if v_batchify_fn == None:
                v_batchify_fn = batchify.Tuple(batchify.Stack(), batchify.Pad(-1))
            v_fn = self.custom_obj.val_data_transform(net=self.net, **kwargs)
            self.v_loader.setup(batch_size=batch_size, shuffle=False, fn=v_fn, batchify_fn=v_batchify_fn, **kwargs)

    def start_train(self, epoch_num, snap_pf, snap_itv, valid_itv):
        self.epoch_num = epoch_num
        self.custom_obj.evaluations()
        eval_metric = self.custom_obj.eval_metric()

        t_volume = self.t_loader.niters
        self.t_week = t_volume / self.t_loader.batch_size
        v_volume = self.v_loader.niters
        self.v_week = v_volume / self.v_loader.batch_size

        for epoch in range(epoch_num):
            self.epoch = epoch  # used for training log calculation
            tic = time()
            # self.net.hybridize(static_alloc=True)
            self.net.hybridize()
            t_bar = tqdm(self.t_loader.batch_loader, desc='Training', unit='batches')
            # train a batch
            nd.waitall()
            self.custom_obj.train_batch(self.net, self.optimizer, epoch, t_bar, self.lr_scheduler,
                                        self.t_loader.batch_size, self.ctx, self.log_interval, self.training_log)
            # validation & snapshots
            if (epoch % valid_itv == 0) or (epoch % snap_itv == 0) or epoch == (epoch_num - 1):
                v_bar = tqdm(self.v_loader.batch_loader, desc='Validation ', unit='batches')
                nd.waitall()
                map_name, mean_ap = self.custom_obj.validate(self.net, v_bar, self.ctx, eval_metric)
                v_params = dict()
                for name, ap in zip(map_name, mean_ap):
                    v_params[name] = ap
                print_train_stats(self.logger, 2, self.v_week, self.epoch, self.epoch_num, self.v_loader.niters,
                                  v_params)
            # saving snapshots
            if epoch % snap_itv == 0 or epoch == (epoch_num - 1):
                self.net.export(snap_pf, epoch=epoch)
                print_snapshot_stats(self.logger, snap_pf, epoch)

            self.logger.info('[Epoch {}] Training cost: {:.3f}'.format(epoch, (time() - tic)))

    def training_log(self, idx, args_dict):
        print_train_stats(self.logger, 1, self.t_week, self.epoch, self.epoch_num, idx, args_dict)


def get_model_frame(ml_type, custom_model, job_dir, gpus):
    if ml_type == 'classification':
        return Classification(custom_model, job_dir, gpus)
    elif ml_type == 'object-detection':
        return Detection(custom_model, job_dir, gpus)
