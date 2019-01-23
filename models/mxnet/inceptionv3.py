import mxnet as mx
import mxnet.gluon.loss as gloss
from custom_model import ClassificationModel
from gluoncv.model_zoo import get_model


class CustomModel(ClassificationModel):

    def net_struct(self):
        kwargs = {'pretrained': False, 'classes': len(self.classes)}
        return get_model('inceptionv3', **kwargs)

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
        return mx.metric.TopKAccuracy(t_n)

    def mixup(self):
        mixup_params = {
            "mixup": True,
            "mixup_alpha": 0.2,
            "mixup_off_epoch": 0
        }
        return mixup_params

    def label_smoothing(self):
        return True

    @staticmethod
    def net_init():
        return mx.init.Xavier()
