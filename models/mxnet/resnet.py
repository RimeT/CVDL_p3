import mxnet as mx
import mxnet.gluon.loss as gloss
from gluoncv.model_zoo.resnet import get_resnet
from custom_model import ClassificationModel


class CustomModel(ClassificationModel):

    def net_struct(self):
        return get_resnet(2, 101, classes=len(self.classes))

    def class_loss(self):
        return gloss.SoftmaxCrossEntropyLoss()

    def eval_metric(self):
        return mx.metric.Accuracy()

    @staticmethod
    def net_init():
        return mx.init.Xavier()
