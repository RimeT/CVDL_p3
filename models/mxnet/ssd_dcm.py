from time import time

import gluoncv as gcv
import mxnet as mx
import copy
from custom_model import DetectionModel
from data.transform import SSDTrainTransform, SSDValTransform
from gluoncv.data import batchify
# from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import gluon, autograd


class CustomModel(DetectionModel):

    def net_struct(self):
        return get_model('ssd_512_vgg16_atrous_custom', pretrained=False, classes=self.classes, pretrained_base=False)

    @staticmethod
    def net_init():
        return mx.init.Xavier()

    def custom_initialization(self, net):
        return True

    def eval_metric(self):
        return VOC07MApMetric(iou_thresh=0.5, class_names=self.classes)

    def train_data_transform(self, **kwargs):
        with autograd.train_mode():
            net = copy.deepcopy(kwargs['net'])
            net.collect_params().reset_ctx(None)
            _, _, anchors = net(mx.nd.zeros(shape=(1, kwargs['channels'], kwargs['r_height'], kwargs['r_width'])))
        return SSDTrainTransform(kwargs['r_width'], kwargs['r_height'], anchors, dicom=True,
                                 window_center=kwargs['window_center'],
                                 window_width=kwargs['window_width'])

    def t_batchify_fn(self):
        return batchify.Tuple(batchify.Stack(), batchify.Stack(), batchify.Stack())

    def val_data_transform(self, **kwargs):
        # mx.nd.waitall()
        return SSDValTransform(kwargs['r_width'], kwargs['r_height'],
                               window_center=kwargs['window_center'],
                               window_width=kwargs['window_width'])

    def v_batchify_fn(self):
        return batchify.Tuple(batchify.Stack(), batchify.Pad(pad_val=-1))

    def evaluations(self):
        """
        metrics will be invoked before training
        :return:
        """
        self.sum_metric = mx.metric.Loss('SumLoss')
        self.mbox_loss = gcv.loss.SSDMultiBoxLoss()
        self.ce_metric = mx.metric.Loss('CrossEntropy')
        self.smoothl1_metric = mx.metric.Loss('SmoothL1')
        self.bast_map = [0]

    def validate(self, net, batch_loader, ctx, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        net.hybridize()
        for batch in batch_loader:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return eval_metric.get()

    def train_batch(self, net, optimizer, epoch, batch_loader, lr_scheduler, batch_size, ctx, log_interval, print_log):
        """
        Train batches in a epoch
        :param epoch:
        :param batch_loader:
        :param lr_scheduler:
        :return:
        """
        self.sum_metric.reset()
        self.ce_metric.reset()
        self.smoothl1_metric.reset()
        btic = time()
        for idx, batch in enumerate(batch_loader):
            lr_scheduler.update(idx, epoch)
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record(True):
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, anchors = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = self.mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                mx.nd.waitall()
                autograd.backward(sum_loss)
            optimizer.step(1)
            self.sum_metric.update(0, [l * batch_size for l in sum_loss])
            self.ce_metric.update(0, [l * batch_size for l in cls_loss])
            self.smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if not (idx + 1) % log_interval:
                params = dict()
                name0, loss0 = self.sum_metric.get()
                name1, loss1 = self.ce_metric.get()
                name2, loss2 = self.smoothl1_metric.get()
                params[name0] = loss0
                params[name1] = loss1
                params[name2] = loss2
                params['lr'] = optimizer.learning_rate
                speed = batch_size / (time() - btic)
                params['speed'] = speed
                print_log(idx, params)
                self.dynamic_bar(batch_loader, epoch, idx, lr=optimizer.learning_rate, loss=loss0)
            btic = time()
