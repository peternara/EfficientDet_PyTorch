import torch
import torch.nn as nn
import math

from models.efficientnet import EfficientNet
from models.bifpn import BIFPN
from .retinahead import RetinaHead
from models.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from .losses import FocalLoss

# from torchvision.ops import nms
# from .boxes_implement import torch_nms
from .box_utils_pytorch import soft_nms

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}

'''
    @ Part function of efficient detection, from backbone to bifpn, without nms.
    @ Author: cema
    @ Date: 2020.03.17, Tuesday
'''


class EfficientDetBiFPN(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88):
        super(EfficientDetBiFPN, self).__init__()
        self.backbone = EfficientNet.get_network_from_name(MODEL_MAP[network])
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)

        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x

    def forward(self, inputs):
        x = self.extract_feat(inputs)

        outs = self.bbox_head(x)

        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        return regression, classification


def detection(classification, regression, inputs, score_thresh=0.01, iou_thresh=0.5):
    anchors = Anchors()
    anchors = anchors(inputs)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, inputs)

    scores = torch.max(classification, dim=2, keepdim=True)[0]

    over_score_thresh_idx = (scores > score_thresh)[0, :, 0]

    classfiy_over_thresh = classification[:, over_score_thresh_idx, :]
    anchors_over_thresh = transformed_anchors[:, over_score_thresh_idx, :]
    scores_over_thresh = scores[:, over_score_thresh_idx, :]

    # nms
    score = scores_over_thresh[0, :, :]
    anchors_score = torch.cat([anchors_over_thresh[0, :, :], score], dim=1)
    anchors_nms_idx, _ = soft_nms(anchors_score, score_threshold=iou_thresh)

    nms_scores, nms_class = classfiy_over_thresh[0, anchors_nms_idx, :].max(
        dim=1)
    nms_anchors = anchors_over_thresh[0, anchors_nms_idx, :]
    return [nms_scores, nms_class, nms_anchors]


class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.01,
                 iou_threshold=0.5):
        super(EfficientDet, self).__init__()
        # self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.backbone = EfficientNet.get_network_from_name(MODEL_MAP[network])

        # print backbone parameters
        # params = list(self.backbone.named_parameters())
        # for param_key, param_value in params:
        #     print("{},   {}".format(param_key, param_value.shape))
        #
        # for features in self.backbone.get_list_features():
        #     print(features)

        self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)

        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()
        self.criterion = FocalLoss()

    def forward(self, inputs):
        if self.is_training:
            inputs, annotations = inputs
        else:
            inputs = inputs
        x = self.extract_feat(inputs)

        outs = self.bbox_head(x)

        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        anchors = self.anchors(inputs)

        if self.is_training:
            return self.criterion(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > self.threshold)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                print('No boxes to NMS')
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            # anchors_nms_idx = nms(
            #     transformed_anchors[0, :, :], scores[0, :, 0], iou_threshold=self.iou_threshold)
            score = scores[0, :, :]
            anchors_score = torch.cat([transformed_anchors[0, :, :], score], dim=1)
            anchors_nms_idx, _ = soft_nms(anchors_score, score_threshold=self.iou_threshold)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(
                dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        print("after backbone: {}".format(x[-1].shape))
        x = self.neck(x[-5:])
        print("after neck: {}".format(x[-1].shape))
        return x
