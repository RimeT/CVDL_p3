import mxnet as mx
from custom_model import DetectionModel
from data.transform import YOLO3TrainTransform, YOLO3ValTransform
from gluoncv.data import batchify
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import gluon, autograd


class CustomModel(DetectionModel):

    def net_struct(self):
        classes = self.classes
        # classes.append("dummy")
        return get_model('yolo3_darknet53_custom', pretrained=False, classes=classes,
                         pretrained_base=False)

    def eval_metric(self):
        return VOC07MApMetric(iou_thresh=0.5, class_names=self.classes)

    @staticmethod
    def net_init():
        return mx.init.Uniform()

    def custom_initialization(self, net):
        return True

    def train_data_transform(self, **kwargs):
        return YOLO3TrainTransform(kwargs['r_width'], kwargs['r_height'], kwargs['net'], channels=kwargs['channels'],
                                   window_center=kwargs['window_center'],
                                   window_width=kwargs['window_width'])

    def t_batchify_fn(self):
        return batchify.Tuple(*([batchify.Stack() for _ in range(6)] + [batchify.Pad(axis=0, pad_val=-1) for _ in range(
            1)]))

    def val_data_transform(self, **kwargs):
        return YOLO3ValTransform(kwargs['r_width'], kwargs['r_height'], window_center=kwargs['window_center'],
                                 window_width=kwargs['window_width'], channels=kwargs['channels'])

    def v_batchify_fn(self):
        return batchify.Tuple(batchify.Stack(), batchify.Pad(pad_val=-1))

    def evaluations(self):
        self.sum_metrics = mx.metric.Loss('SumLoss')
        self.obj_metrics = mx.metric.Loss('ObjLoss')
        self.center_metrics = mx.metric.Loss('BoxCenterLoss')
        self.scale_metrics = mx.metric.Loss('BoxScaleLoss')
        self.cls_metrics = mx.metric.Loss('ClassLoss')

    def validate(self, net, val_data, ctx, eval_metric, test_cb=None):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        mx.nd.waitall()
        net.hybridize()
        for batch in val_data:
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

            # testing callback
            if test_cb:
                for img_result in zip(det_ids, det_scores, det_bboxes, gt_ids, gt_bboxes):
                    pred_cb = list()
                    for z_pred in zip(img_result[0][0], img_result[1][0], img_result[2][0]):
                        if z_pred[0] >= 0:
                            a_box = [zp.asnumpy().tolist() for zp in z_pred]
                            a_box[0] = int(a_box[0][0])
                            a_box[1] = a_box[1][0]
                            pred_cb.append(a_box)
                    gt_cb = list()
                    for z_gt in zip(img_result[3][0], img_result[4][0]):
                        if z_gt[0] >= 0:
                            a_box = [zg.asnumpy().tolist() for zg in z_gt]
                            a_box[0] = int(a_box[0][0])
                            gt_cb.append(a_box)
                    test_cb(pred_cb, gt_cb)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return eval_metric.get()

    def train_batch(self, net, optimizer, epoch, batch_loader, lr_scheduler, batch_size, ctx, log_interval,
                    print_log):
        btic = time()
        self.obj_metrics.reset()
        self.center_metrics.reset()
        self.scale_metrics.reset()
        self.cls_metrics.reset()
        for idx, batch in enumerate(batch_loader):
            lr_scheduler.update(idx, epoch)
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                      *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                mx.nd.waitall()
                autograd.backward(sum_losses)
            optimizer.step(batch_size)
            self.sum_metrics.update(0, sum_losses)
            self.obj_metrics.update(0, obj_losses)
            self.center_metrics.update(0, center_losses)
            self.scale_metrics.update(0, scale_losses)
            self.cls_metrics.update(0, cls_losses)
            if not (idx + 1) % log_interval:
                name0, loss0 = self.sum_metrics.get()
                name1, loss1 = self.obj_metrics.get()
                name2, loss2 = self.center_metrics.get()
                name3, loss3 = self.scale_metrics.get()
                name4, loss4 = self.cls_metrics.get()
                params = dict()
                params[name0] = loss0
                params[name1] = loss1
                params[name2] = loss2
                params[name3] = loss3
                params[name4] = loss4
                params['lr'] = optimizer.learning_rate
                speed = batch_size / (time() - btic)
                params['speed'] = speed
                print_log(idx, params)
                self.dynamic_bar(batch_loader, epoch, idx, speed=speed, sum_loss=loss0)
            btic = time()
