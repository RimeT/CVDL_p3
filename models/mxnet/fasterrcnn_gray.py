import gluoncv as gcv
import mxnet as mx
from custom_model import DetectionModel
from data.transform import FasterRCNNTrainTransform, FasterRCNNValTransform
from gluoncv.data import batchify
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import autograd


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        # pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class CustomModel(DetectionModel):

    def net_struct(self):
        # return get_model('faster_rcnn_resnet50_v1b_custom', classes=self.classes, pretrained_base=False, transfer='voc')
        return get_model('faster_rcnn_resnet50_v1b_custom', pretrained=False, classes=self.classes, pretrained_base=False)

    @staticmethod
    def net_init():
        return mx.init.Xavier()

    def custom_initialization(self, net):
        return True

    def eval_metric(self):
        return VOC07MApMetric(iou_thresh=0.5, class_names=self.classes)

    def train_data_transform(self, **kwargs):
        net = kwargs['net']
        return FasterRCNNTrainTransform(net.short, net.max_size, net)

    def t_batchify_fn(self):
        return batchify.Tuple(*[batchify.Append() for _ in range(5)])

    def val_data_transform(self, **kwargs):
        net = kwargs['net']
        return FasterRCNNValTransform(net.short, net.max_size)

    def v_batchify_fn(self):
        return batchify.Tuple(*[batchify.Append() for _ in range(3)])

    def evaluations(self):
        self.rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
        self.rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
        self.metrics = [mx.metric.Loss('RPN_Conf'),
                        mx.metric.Loss('RPN_SmoothL1'),
                        mx.metric.Loss('RCNN_CrossEntropy'),
                        mx.metric.Loss('RCNN_SmoothL1'), ]
        rpn_acc_metric = RPNAccMetric()
        rpn_bbox_metric = RPNL1LossMetric()
        rcnn_acc_metric = RCNNAccMetric()
        rcnn_bbox_metric = RCNNL1LossMetric()
        self.metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    def split_and_load(self, batch, ctx_list):
        """Split data to 1 batch each device."""
        num_ctx = len(ctx_list)
        new_batch = []
        for i, data in enumerate(batch):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
            new_batch.append(new_data)
        return new_batch

    def validate(self, net, val_data, ctx, eval_metric):
        """Test on validation dataset."""
        clipper = gcv.nn.bbox.BBoxClipToImage()
        eval_metric.reset()
        net.hybridize(static_alloc=True)
        for batch in val_data:
            batch = self.split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y, im_scale in zip(*batch):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(clipper(bboxes, x))
                # rescale to original resolution
                im_scale = im_scale.reshape((-1)).asscalar()
                det_bboxes[-1] *= im_scale
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_bboxes[-1] *= im_scale
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes,
                                                                            gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
        return eval_metric.get()

    def train_batch(self, net, optimizer, epoch, batch_loader, lr_scheduler, batch_size, ctx, log_interval, print_log):
        for metric in self.metrics:
            metric.reset()
        btic = time()
        mix_ratio = 1.0
        for idx, batch in enumerate(batch_loader):
            lr_scheduler.update(idx, epoch)
            batch = self.split_and_load(batch, ctx)
            losses = []
            metric_losses = [[] for _ in self.metrics]
            add_losses = [[] for _ in self.metrics2]
            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)
                    # losses of rpn
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = self.rpn_cls_loss(rpn_score, rpn_cls_targets,
                                                  rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = self.rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos
                    # rpn overall loss, use sum rather than average
                    rpn_loss = rpn_loss1 + rpn_loss2
                    # generate targets for rcnn
                    cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
                    # losses of rcnn
                    num_rcnn_pos = (cls_targets >= 0).sum()
                    rcnn_loss1 = self.rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0) * cls_targets.size / \
                                 cls_targets.shape[0] / num_rcnn_pos
                    rcnn_loss2 = self.rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[
                        0] / num_rcnn_pos
                    rcnn_loss = rcnn_loss1 + rcnn_loss2
                    # overall losses
                    losses.append(rpn_loss.sum() * mix_ratio + rcnn_loss.sum() * mix_ratio)
                    metric_losses[0].append(rpn_loss1.sum() * mix_ratio)
                    metric_losses[1].append(rpn_loss2.sum() * mix_ratio)
                    metric_losses[2].append(rcnn_loss1.sum() * mix_ratio)
                    metric_losses[3].append(rcnn_loss2.sum() * mix_ratio)
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])
                mx.nd.waitall()
                autograd.backward(losses)
                for metric, record in zip(self.metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(self.metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            optimizer.step(1)
            if not (idx + 1) % log_interval:
                params = dict()
                for metric in self.metrics + self.metrics2:
                    name, loss = metric.get()
                    params[name] = loss
                sum_loss = sum(params[p] for p in params)
                params['SumLoss'] = sum_loss
                params['lr'] = optimizer.learning_rate
                params['speed'] = time() - btic
                print_log(idx, params)
                self.dynamic_bar(batch_loader, epoch, idx, sum_loss=sum_loss, speed=params['speed'])
            btic = time()
