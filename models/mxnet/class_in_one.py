import mxnet as mx
import mxnet.gluon.loss as gloss
from mxnet.gluon import nn
from custom_model import ClassificationModel
from gluoncv.model_zoo import get_model


class CustomModel(ClassificationModel):

    def net_struct(self):
        # train from scratch
        kwargs = {'pretrained': False, 'classes': len(self.classes)}
        return get_model('se_resnext50_32x4d', **kwargs)
        # pretrained
        # kwargs = {'pretrained': True}
        # return get_model('resnet50_v2', **kwargs)

    def custom_initialization(self, net):
        # train from scratch
        return False
        # transfer learning
        # with net.name_scope():
        #     net.output = nn.Dense(len(self.classes))
        # net.output.initialize()
        # return True

    def class_loss(self):
        if self.label_smoothing() or self.mixup()['mixup']:
            return gloss.SoftmaxCrossEntropyLoss(sparse_label=False)
        else:
            return gloss.SoftmaxCrossEntropyLoss()

    def train_eval_metric(self):
        if self.mixup()['mixup']:
            return mx.metric.RMSE()
        else:
            return mx.metric.Accuracy()

    def val_eval_metric(self):
        t_n = min(5, len(self.classes))
        # return mx.metric.TopKAccuracy(t_n)
        return mx.metric.Accuracy()

    def mixup(self):
        mixup_params = {
            "mixup": False,
            "mixup_alpha": 0.2,
            "mixup_off_epoch": 0
        }
        return mixup_params

    def label_smoothing(self):
        return True

    @staticmethod
    def net_init():
        return mx.init.Xavier()
