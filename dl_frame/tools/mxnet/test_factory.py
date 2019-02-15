from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import logging
import os
from time import time

import mxnet as mx
from data.mx_data import LoaderFactory
from gluoncv.data import batchify
from tqdm import tqdm


def _dict2csv(csv_path, data, titles):
    with open(csv_path, 'w') as f:
        if not isinstance(titles, list) or len(titles) <= 0:
            raise ValueError("Testing Error: csv titles format error.")
        writer = csv.DictWriter(f, titles)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def _convert_array_label(data, labels):
    result = list()
    for item in data:
        item[0] = labels[item[0]]
        result.append(item)
    return result


class TestFactory(object):
    def __init__(self, custom_model, job_dir, gpus):
        self.logger = logging.getLogger('testing_logger')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(job_dir, 'mxnet_testing.log'))
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
        self.job_dir = job_dir
        self.ctx = mx.cpu()
        if len(gpus) > 0:
            self.ctx = [mx.gpu(i) for i in gpus]
        self.loader = None
        self.test_results = list()
        self.classes = None

    def create_dataloader(self, db_path, **kwargs):
        raise NotImplementedError

    def data_setup(self, **kwargs):
        raise NotImplementedError

    def net_config(self, weights):
        self.custom_obj = self.custom_model(self.classes, self.ctx, self.logger)
        self.net = self.custom_obj.net_struct()
        if not os.path.exists(weights):
            raise ValueError("weights file {} not found".format(weights))
        self.net.collect_params().load(weights)
        for param in self.net.collect_params().values():
            if param._data is None:
                raise ValueError('DL Model loading params error. Null Param:{}'.format(param))
        self.net.collect_params().reset_ctx(self.ctx)

    def start_test(self, result_type=0):
        raise NotImplementedError


class Detection(TestFactory):

    def __init__(self, custom_model, job_dir, gpus):
        super(Detection, self).__init__(custom_model, job_dir, gpus)
        self.file_list = None
        self.fname_idx = 0

    def create_dataloader(self, db_path, **kwargs):
        self.classes = [str(i) for i in kwargs['labels']]
        self.loader = LoaderFactory.set_source('object-detection', db_path, root=kwargs['data_root'], **kwargs)
        self.file_list = self.loader._dataset.get_item_list()

    def data_setup(self, **kwargs):
        batchify_fn = self.custom_obj.v_batchify_fn()
        if batchify_fn is None:
            batchify_fn = batchify.Tuple(batchify.Stack(), batchify.Pad(-1))
        cust_transform = self.custom_obj.val_data_transform(net=self.net, **kwargs)
        batch_size = len(self.ctx)
        self.loader.setup(batch_size=batch_size, shuffle=False, fn=cust_transform, batchify_fn=batchify_fn,
                          last_batch='keep', **kwargs)

    def start_test(self, result_type=0):
        print("start testing")
        # test configuration
        self.fname_idx = 0
        t1 = time()
        eval_metric = self.custom_obj.eval_metric()
        bar = tqdm(self.loader.batch_loader, desc='Testing ', unit='batches')
        self.net.hybridize()
        try:
            map_name, mean_ap = self.custom_obj.validate(self.net, bar, self.ctx, eval_metric, self.record_message)
            test_params = dict()
            for name, ap in zip(map_name, mean_ap):
                test_params[name] = ap
            self.logger.debug("DATASET STATISTICS: {}".format(test_params))
        except Exception as e:
            print(e)
        finally:
            # generate csv
            csv_name = "result.csv"
            titles = ['Image_Name', 'Ground_Truth', 'Prediction']
            _dict2csv(os.path.join(self.job_dir, csv_name), self.test_results, titles)

        print("Testing costs: {}".format(time() - t1))

    def record_message(self, pred, gt):
        pred_result = dict()
        pred_result['Image_Name'] = self.file_list[self.fname_idx]
        self.fname_idx += 1
        # Pred: label_id, score, bbox
        pred_result['Prediction'] = _convert_array_label(pred, self.classes)
        # Ground Truth: label_id, bbox
        pred_result['Ground_Truth'] = _convert_array_label(gt, self.classes)
        self.test_results.append(pred_result)


def get_test_frame(dl_type, custom_model, job_dir, gpus):
    if dl_type == 'classification':
        print('Classification not supported')
    elif dl_type == 'object-detection':
        return Detection(custom_model, job_dir, gpus)
    elif dl_type == 'segmentation':
        print('Segmentation not supported')
    else:
        raise ValueError('Deep learning task {} not supported'.format(dl_type))
