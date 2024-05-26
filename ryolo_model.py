from typing import Type, Union, Mapping, Any, Optional
from functools import partial
import torchvision
import numpy as np
from torch import nn
import math
import torch
import time
import colorsys
import cv2
import re
from pathlib import Path
import torchvision.transforms as transforms
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import imgaug.augmenters as iaa
from easydict import EasyDict as edict
import sys
import os
import yaml
import torch.nn.functional as F
from torch.optim.optimizer import required
from torchinfo import summary
from PIL import Image, ImageDraw, ImageFont
import random
import string
from math import sin, cos, radians, sqrt
import glob
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from torch.optim.optimizer import Optimizer
from matplotlib.pyplot import figure


from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

"""# **Config**"""

__C = edict()
cfg = __C

__C.YOLO = edict()
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.CLASSES = ''
__C.YOLO.YAML_PATH = ''
__C.YOLO.NUM_CLASSES = 80
__C.YOLO.TOPK = 13
__C.YOLO.MODEL_TYPE = 's' # [s, m, l]
__C.YOLO.ANGLE_MIN = 0
__C.YOLO.ANGLE_MAX = 360

__C.YOLO.TRAIN = edict()
__C.YOLO.TRAIN.BATCH_SIZE = 8
__C.YOLO.TRAIN.EPOCHS = 200
__C.YOLO.TRAIN.EPOCHS_FOR_PRETRAINING = 3
__C.YOLO.TRAIN.EPOCHS_FOR_FINE_TUNING = 200
__C.YOLO.TRAIN.MODEL_SIZE = (640, 640)
__C.YOLO.TRAIN.ANNOT_PATH = ''
__C.YOLO.TRAIN.SAVED_MODEL_DIR = ''
__C.YOLO.TRAIN.DATA_AUG = True
__C.YOLO.TRAIN.HORIZONTAL_FLIP = True
__C.YOLO.TRAIN.VERTICAL_FLIP = True
__C.YOLO.TRAIN.RANDOM_CROP = True
__C.YOLO.TRAIN.RANDOM_SCALE = True
__C.YOLO.TRAIN.RANDOM_TRANSLATE = True
__C.YOLO.TRAIN.RANDOM_ROTATE = True
__C.YOLO.TRAIN.USE_COLORJITTER = True
__C.YOLO.TRAIN.LR_INIT = 1e-4
__C.YOLO.TRAIN.OPTIMIZER_TYPE = 'adam'
__C.YOLO.TRAIN.VISUAL_LEARNING_PROCESS = True
__C.YOLO.TRAIN.TRANSFER = 'transfer'
__C.YOLO.TRAIN.ADD_IMG_PATH = ''
__C.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES = 'siou' # giou, diou, ciou, siou
__C.YOLO.TRAIN.CONF_THRESHOLD = 0.5
__C.YOLO.TRAIN.IOU_THRESHOLD = 0.25
__C.YOLO.TRAIN.MAX_BBOX_PER_IMAGE = 50
__C.YOLO.TRAIN.BACKBONE_DATE = None
__C.YOLO.TRAIN.NECK_DATA = None
__C.YOLO.TRAIN.PATIENCE = 20
__C.YOLO.TRAIN.N_SAMPLES_PER_EPOCH = 20000
__C.YOLO.TRAIN.USE_VALID_DATASET = False
__C.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = ''


# generate fake images
__C.GENERATE_FAKE_IMAGE = edict()
__C.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR =  ''
__C.GENERATE_FAKE_IMAGE.FONT_DIR = ""
__C.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN = 20
__C.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX = 400
__C.GENERATE_FAKE_IMAGE.TEXT_COLOR = None
__C.GENERATE_FAKE_IMAGE.IMAGE_SIZE = (640, 640)
__C.GENERATE_FAKE_IMAGE.WORD_COUNT = 5
__C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN = 1
__C.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX = 7
__C.GENERATE_FAKE_IMAGE.ANGLE_MIN = -180
__C.GENERATE_FAKE_IMAGE.ANGLE_MAX =  180


# icdar
__C.YOLO.TRAIN.ICDAR13_ANNOT_DIR = ''
__C.YOLO.TRAIN.ICDAR13_IMAGE_DIR_FOR_TRAINING = ''
__C.YOLO.TRAIN.ICDAR15_ANNOT_DIR = ''
__C.YOLO.TRAIN.ICDAR15_IMAGE_DIR_FOR_TRAINING = ''
__C.YOLO.TRAIN.ICDAR17_ANNOT_PATH = ''
__C.YOLO.TRAIN.ICDAR17_IMAGE_DIR_FOR_TRAINING = ''

# total text
__C.YOLO.TRAIN.TOTAL_TEXT_ANNOT_PATH = ''
__C.YOLO.TRAIN.TOTAL_TEXT_IMAGE_DIR_FOR_TRAINING = ''

"""# **Utils**"""

def allowed_file(filename):
    return ("." in filename and filename.rsplit(".", 1)[1].lower() in ["png", "jpg", "jpeg", "bmp"])


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def width_multiplier(original, factor, divisor=None):
    if divisor is None:
        return int(original * factor)
    else:
        return math.ceil(int(original * factor) / divisor) * divisor


def get_yaml_and_weight_path(model_type):
    if model_type == 's':
        yaml_path = 'assets/yaml-weight/yolo_nas_s_arch_params.yaml'
        weight_path = 'assets/yaml-weight/yolo_nas_s_coco.pth'
        return yaml_path, weight_path
    elif model_type == 'm':
        yaml_path = 'assets/yaml-weight/yolo_nas_m_arch_params.yaml'
        weight_path = 'assets/yaml-weight/yolo_nas_m_coco.pth'
        return yaml_path, weight_path
    else:
        yaml_path = 'assets/yaml-weight/yolo_nas_l_arch_params.yaml'
        weight_path = 'assets/yaml-weight/yolo_nas_l_coco.pth'
        return yaml_path, weight_path


def get_model_info_from_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    backbone_data = {}
    neck_data = {}
    head_data = {}

    # backbone
    backbone_data['out_stem_channels'] = yaml_data['backbone']['NStageBackbone']['stem']['YoloNASStem']['out_channels']
    backbone_data['num_blocks_list'] = [x['YoloNASStage']['num_blocks']
                                        for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['out_stage_channels_list'] = [x['YoloNASStage']['out_channels']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['hidden_channels_list'] = [x['YoloNASStage']['hidden_channels']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['concat_intermediates_list'] = [x['YoloNASStage']['concat_intermediates']
                                                for x in yaml_data['backbone']['NStageBackbone']['stages']]
    backbone_data['output_context_channels'] = yaml_data['backbone']['NStageBackbone']['context_module']['SPP']['output_channels']
    backbone_data['k'] = yaml_data['backbone']['NStageBackbone']['context_module']['SPP']['k']

    # neck
    neck_data['out_channels_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['out_channels']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['num_blocks_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['num_blocks']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['hidden_channels_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['hidden_channels']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['width_mult_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['width_mult']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]
    neck_data['depth_mult_list'] = [yaml_data['neck']['YoloNASPANNeckWithC2'][key1][key2]['depth_mult']
                                      for key1 in yaml_data['neck']['YoloNASPANNeckWithC2']
                                      for key2 in yaml_data['neck']['YoloNASPANNeckWithC2'][key1]]

    # head
    head_data['inter_channels_list'] = [x['YoloNASDFLHead']['inter_channels'] for x in yaml_data['heads']['NDFLHeads']['heads_list']]
    head_data['width_mult_list'] = [x['YoloNASDFLHead']['width_mult'] for x in yaml_data['heads']['NDFLHeads']['heads_list']]

    return backbone_data, neck_data, head_data


def make_anchors(imgsz, strides=[8, 16, 32], grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, stride in enumerate(strides):
        h, w = imgsz[0] // stride, imgsz[1] // stride
        sx = torch.arange(end=w, device=device, dtype=torch.float32) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=torch.float32) + grid_cell_offset  # shift y
        # sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def fuse_conv_bn(model: nn.Module, replace_bn_with_identity: bool = False):
    children = list(model.named_children())
    counter = 0
    for i in range(len(children) - 1):
        if isinstance(children[i][1], torch.nn.Conv2d) and isinstance(children[i + 1][1], torch.nn.BatchNorm2d):
            setattr(model, children[i][0], torch.nn.utils.fuse_conv_bn_eval(children[i][1], children[i + 1][1]))
            if replace_bn_with_identity:
                setattr(model, children[i + 1][0], nn.Identity())
            else:
                delattr(model, children[i + 1][0])
            counter += 1
    for child_name, child in children:
        counter += fuse_conv_bn(child, replace_bn_with_identity)

    return counter


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIOU=False, eps=1e-7):
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1 + eps, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1 + eps, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or SIOU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU or SIOU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif SIOU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area
    return iou  # IoU


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)


def xcycwha2xyxya(boxes):
    boxes[:, [0, 1]], boxes[:, [2, 3]] = boxes[:, [0, 1]] - boxes[:, [2, 3]] / 2.0, boxes[:, [0, 1]] + boxes[:, [2, 3]] / 2.0
    return boxes


def xyxya2xcycwha(boxes):
    boxes[..., [0, 1]], boxes[..., [2, 3]] = boxes[..., [0, 1]] + (boxes[..., [2, 3]] - boxes[..., [0, 1]]) / 2.0, boxes[..., [2, 3]] - boxes[..., [0, 1]]
    return boxes


def get_new_state_dict(checkpoint_base, checkpoint_model):
    for key in checkpoint_base.keys():
        checkpoint_model[key] = checkpoint_base[key]
    return checkpoint_model


def check_new_state_dict(checkpoint_base, new_checkpoint):
    for key in checkpoint_base.keys():
        if torch.eq(checkpoint_base[key], new_checkpoint[key]).any():
            continue
        else:
            print(key)

def post_processing(num_word, group_chars, classes_map=string.ascii_letters + string.digits):
    result = []
    if num_word == 0:
        return result
    for char_boxes_of_current_word in group_chars:
       chars_list = [classes_map[int(char_box[-1])] for char_box in char_boxes_of_current_word]
       result.append("".join(char for char in chars_list))
    return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""# **common modules**"""

class Residual(nn.Module):
    def forward(self, x):
        return x

class QARepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, activation_type=nn.ReLU, activation_kwargs=None,
                 se_type=nn.Identity, se_kwargs=None, build_residual_branches=True, use_residual_connection=True, use_alpha=False,
                 use_1x1_bias=True, use_post_bn=True):
        super(QARepVGGBlock, self).__init__()
        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation_type = activation_type
        self.activation_kwargs = activation_kwargs
        self.se_type = se_type
        self.se_kwargs = se_kwargs
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha
        self.use_1x1_bias = use_1x1_bias
        self.use_post_bn = use_post_bn

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        self.branch_3x3 = nn.Sequential()
        self.branch_3x3.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation
            )
        )

        self.branch_3x3.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

        self.branch_1x1 =  nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=use_1x1_bias
        )

        if use_residual_connection:
            assert out_channels == in_channels and stride == 1

            self.identity = Residual()

            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1.0

            self.id_tensor: Optional[torch.Tensor]

            self.register_buffer(
                name='id_tensor',
                tensor=id_tensor.to(dtype=self.branch_1x1.weight.dtype, device=self.branch_1x1.weight.device),
                persistent=False
            )
        else:
            self.identity = None

        if use_alpha:
            self.alpha = torch.nn.Parameters(torch.tensor([1, 0]), requires_grad=True)
        else:
            self.alpha = 1.0

        if self.use_post_bn:
            self.post_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.post_bn = nn.Identity()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True
        )

        self.partially_fused = False
        self.fully_fused = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()


    def forward(self, inputs):
        if self.fully_fused:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))

        if self.partially_fused:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        if self.identity is None:
            id_out = 0.0
        else:
            id_out = self.identity(inputs)

        x_3x3 = self.branch_3x3(inputs)
        x_1x1 = self.alpha * self.branch_1x1(inputs)

        branches = x_3x3 + x_1x1 + id_out

        out = self.nonlinearity(self.post_bn(branches))
        se = self.se(out)

        return se


    def _get_equivalent_kernel_bias_for_branches(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(
            self.branch_3x3.conv.weight,
            0,
            self.branch_3x3.bn.running_mean,
            self.branch_3x3.bn.running_var,
            self.branch_3x3.bn.weight,
            self.branch_3x3.bn.bias,
            self.branch_3x3.bn.eps,
        )
        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.branch_1x1.weight)
        bias1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else 0
        kernelid = self.id_tensor if self.identity is not None else 0
        biasid = 0
        eq_kernel_3x3 = kernel3x3 + self.alpha * kernel1x1 + kernelid
        eq_bias_3x3 = bias3x3 + self.alpha * bias1x1 + biasid
        return eq_kernel_3x3, eq_bias_3x3

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, kernel, bias, running_mean, running_var, gamma, beta, eps):
        std = torch.sqrt(running_var + eps)
        b = beta - gamma * running_mean / std
        A = gamma / std
        A_ = A.expand_as(kernel.transpose(0, -1)).transpose(0, -1)
        fused_kernel = kernel * A_
        fused_bias = bias * A + b
        return fused_kernel, fused_bias

    def full_fusion(self):
        if self.fully_fused:
            return

        if not self.partially_fused:
            self.partial_fusion()

        if self.use_post_bn:
            eq_kernel, eq_bias = self._fuse_bn_tensor(
                self.rbr_reparam.weight,
                self.rbr_reparam.bias,
                self.post_bn.running_mean,
                self.post_bn.running_var,
                self.post_bn.weight,
                self.post_bn.bias,
                self.post_bn.eps,
            )

            self.rbr_reparam.weight.data = eq_kernel

            self.rbr_reparam.bias.data = eq_bias

        for para in self.parameters():
            para.detach_()

        if hasattr(self, "post_bn"):
            self.__delattr__("post_bn")

        self.partially_fused = False
        self.fully_fused = True


    def partial_fusion(self):
        if self.partially_fused:
            return

        if self.fully_fused:
            raise NotImplementedError("QARepVGGBlock can't be converted to partially fused from fully fused")

        kernel, bias = self._get_equivalent_kernel_bias_for_branches()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        if hasattr(self, "alpha"):
            self.__delattr__("alpha")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.partially_fused = True
        self.fully_fused = False

    def fuse_block_residual_branches(self):
        self.partial_fusion()

    def prep_model_for_conversion(self, input_size=None, full_fusion=True, **kwargs):
        if full_fusion:
            self.full_fusion()
        else:
            self.partial_fusion()


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation_type, stride, dilation, groups=1,
                 bias=True, padding_mode='zeros', use_normalization=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None, activation_kwargs=None):
        super(ConvBNAct, self).__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        self.seq.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            )
        )

        if use_normalization:
            self.seq.add_module(
                'bn',
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                               track_running_stats=track_running_stats, device=device, dtype=dtype)
            )

        if activation_type is not None:
            self.seq.add_module(
                'act',
                activation_type(**activation_kwargs)
            )

    def forward(self, x):
        return self.seq(x)


class ConvBNReLU(ConvBNAct):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 use_normalization=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None,
                 use_activation=True, inplace=False):
        super(ConvBNReLU, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=nn.ReLU if use_activation else None,
            activation_kwargs=dict(inplace=inplace) if inplace else None,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            use_normalization=use_normalization,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding

class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type, padding=None, groups=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel, stride, autopad(kernel, padding), groups=groups or 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class YoloNASBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, block_type, activation_type, shortcut, use_alpha):
        super(YoloNASBottleneck, self).__init__()
        self.cv1 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.cv2 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

    def forward(self, x):
        return self.alpha * self.shortcut(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SequentialWithIntermediates(nn.Sequential):
    def __init__(self, output_intermediates, *kwargs):
        super(SequentialWithIntermediates, self).__init__(*kwargs)
        self.output_intermediates = output_intermediates

    def forward(self, input):
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        return [super(SequentialWithIntermediates, self).forward(input)]

class YoloNASCSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, block_type, activation_type, shortcut=True, use_alpha=True,
                 expansion=0.5, hidden_channels=None, concat_intermediates=False):
        super(YoloNASCSPLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv3 = Conv(hidden_channels * (2 + concat_intermediates * num_bottlenecks), out_channels, 1, stride=1, activation_type=activation_type)
        module_list = [YoloNASBottleneck(hidden_channels, hidden_channels, block_type, activation_type, shortcut, use_alpha)
                      for _ in range(num_bottlenecks)]
        self.bottlenecks = SequentialWithIntermediates(concat_intermediates, *module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        return self.conv3(x)

class YoloNASStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloNASStem, self).__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(in_channels, out_channels, stride=2, use_residual_connection=False)

    def forward(self, x):
        return self.conv(x)

    @property
    def out_channels(self):
        return self._out_channels

class YoloNASStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, activation_type, hidden_channels=None, concat_intermediates=False):
        super(YoloNASStage, self).__init__()
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(in_channels, out_channels, stride=2, activation_type=activation_type, use_residual_connection=False)
        self.blocks = YoloNASCSPLayer(out_channels, out_channels, num_blocks, QARepVGGBlock, activation_type, True,
                                      hidden_channels=hidden_channels, concat_intermediates=concat_intermediates)

    def forward(self, x):
        return self.blocks(self.downsample(x))

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASUpStage(nn.Module):
    def __init__(self, in_channels, out_channels, width_mult, num_blocks, depth_mult, activation_type, hidden_channels=None,
                 concat_intermediates=False, reduce_channels=False):
        super(YoloNASUpStage, self).__init__()
        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels

        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = Conv(skip_in_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = Conv(skip_in_channels1, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
            self.reduce_skip2 = Conv(skip_in_channels2, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        self.conv = Conv(in_channels, out_channels, 1, 1, activation_type)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        if num_inputs == 3:
            self.downsample = Conv(out_channels if reduce_channels else skip_in_channels2, out_channels, kernel=3, stride=2,
                                   activation_type=activation_type)

        self.reduce_after_concat = Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type=activation_type) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels

        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates
        )

        self._out_channels = [out_channels, out_channels]

    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], dim=1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x

    @property
    def out_channels(self):
        return self._out_channels

class YoloNASDownStage(nn.Module):
    def __init__(self, in_channels, out_channels, width_mult, num_blocks, depth_mult, activation_type, hidden_channels=None,
                 concat_intermediates=False):
        super(YoloNASDownStage, self).__init__()
        in_channels, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = Conv(in_channels, out_channels // 2, 3, 2, activation_type=activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(Conv, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates
        )

        self._out_channels = out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.blocks(x)
        return x

    @property
    def out_channels(self):
        return self._out_channels


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k, activation_type):
        super(SPP, self).__init__()
        self._out_channels = out_channels

        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * (len(k) + 1), out_channels, 1, 1, activation_type)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))

    @property
    def out_channels(self):
        return self._out_channels


class NStageBackbone(nn.Module):
    def __init__(self, in_channels=3, out_stem_channels=48, out_stage_channels_list=[96, 192, 384, 768], hidden_channels_list=[32, 64, 96, 192],
                 num_blocks_list=[2, 3, 5, 2], out_layers=['stage1', 'stage2', 'stage3', 'context_module'], activation_type=nn.ReLU,
                 concat_intermediates_list=[False, False, False, False], stem='YoloNASStem', context_module='SPP',
                 stages=['YoloNASStage', 'YoloNASStage', 'YoloNASStage', 'YoloNASStage'], output_context_channels=768,
                 k=[5, 9, 13]):
        super(NStageBackbone, self).__init__()
        self.num_stages = len(stages)
        self.stem = eval(stem)(in_channels, out_stem_channels)
        prev_channels = self.stem.out_channels
        for i in range(self.num_stages):
            new_stage = eval(stages[i])(prev_channels, out_stage_channels_list[i], num_blocks_list[i], activation_type,
                                        hidden_channels_list[i], concat_intermediates_list[i])
            setattr(self, f"stage{i + 1}", new_stage)
            prev_channels = new_stage.out_channels

        self.context_module = eval(context_module)(prev_channels, output_context_channels, k, activation_type)

        self.out_layers = out_layers

        self._out_channels = self._define_out_channels()

    def _define_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    def forward(self, x):
        outputs = []
        all_layers = ['stem'] + [f"stage{i}" for i in range(1, self.num_stages + 1)] + ['context_module']
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASPANNeckWithC2(nn.Module):
    def __init__(self, in_channels, neck_module_list=['YoloNASUpStage', 'YoloNASUpStage', 'YoloNASDownStage', 'YoloNASDownStage'],
                 out_channels_list=[192, 96, 192, 384], hidden_channels_list=[64, 48, 64, 64], activation_type=nn.ReLU,
                 num_blocks_list=[2, 2, 2, 2], width_mult_list=[1, 1, 1, 1], depth_mult_list=[1, 1, 1, 1],
                 reduce_channels_list=[True, True]):
        super(YoloNASPANNeckWithC2, self).__init__()
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        self.neck1 = YoloNASUpStage(
            in_channels=[c5_out_channels, c4_out_channels, c3_out_channels],
            out_channels=out_channels_list[0],
            width_mult=width_mult_list[0],
            num_blocks=num_blocks_list[0],
            depth_mult=depth_mult_list[0],
            hidden_channels=hidden_channels_list[0],
            reduce_channels=reduce_channels_list[0],
            activation_type=activation_type
        )

        self.neck2 = YoloNASUpStage(
            in_channels=[self.neck1.out_channels[1], c3_out_channels, c2_out_channels],
            out_channels=out_channels_list[1],
            width_mult=width_mult_list[1],
            num_blocks=num_blocks_list[1],
            depth_mult=depth_mult_list[1],
            hidden_channels=hidden_channels_list[1],
            reduce_channels=reduce_channels_list[1],
            activation_type=activation_type
        )

        self.neck3 = YoloNASDownStage(
            in_channels=[self.neck2.out_channels[1], self.neck2.out_channels[0]],
            out_channels=out_channels_list[2],
            width_mult=width_mult_list[2],
            num_blocks=num_blocks_list[2],
            depth_mult=depth_mult_list[2],
            hidden_channels=hidden_channels_list[2],
            activation_type=activation_type
        )

        self.neck4 = YoloNASDownStage(
            in_channels=[self.neck3.out_channels, self.neck1.out_channels[0]],
            out_channels=out_channels_list[3],
            width_mult=width_mult_list[3],
            num_blocks=num_blocks_list[3],
            depth_mult=depth_mult_list[3],
            hidden_channels=hidden_channels_list[3],
            activation_type=activation_type
        )

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs
        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5

    @property
    def out_channels(self):
        return self._out_channels


class YoloNASDFLHead(nn.Module):
    def __init__(self, in_channels, inter_channels, width_mult, first_conv_group_size, num_classes, stride, reg_max,
                 angle_min=-90, angle_max=90):
        super(YoloNASDFLHead, self).__init__()
        inter_channels = width_multiplier(inter_channels, width_mult, 8)

        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.inter_channels = inter_channels

        self.num_classes = num_classes
        self.stem = ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_angle_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.angle_convs = nn.Sequential(*first_angle_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.cls_pred = nn.Conv2d(inter_channels, 80, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.angle_pred = nn.Conv2d(inter_channels, 1 * (angle_max - angle_min + 1), 1, 1, 0)

        self.prior_prob = 1e-2


    def replace_num_classes(self):
        if self.num_classes != 80:
            self.cls_pred = nn.Conv2d(self.inter_channels, self.num_classes, 1, 1, 0)
            self._initialize_base()


    def _initialize_base(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)

    def forward(self, x):
        x = self.stem(x)

        reg_feat = self.reg_convs(x)
        reg_distri = self.reg_pred(reg_feat)

        cls_feat = self.cls_convs(x)
        cls_logit = self.cls_pred(cls_feat)

        angle_feat = self.angle_convs(x)
        angle_distri = self.angle_pred(angle_feat)

        return reg_distri, cls_logit, angle_distri



class NDFHeads(nn.Module):
    def __init__(self, in_channels, num_classes=80, inter_channels_list=[128, 256, 512], stride_list=[8, 16, 32],
                 reg_max=16, angle_min=-90, angle_max=90, width_mult_list=[0.5, 0.5, 0.5], first_conv_group_size=0):
        super(NDFHeads, self).__init__()
        self.in_channels = in_channels
        self.inter_channels_list = inter_channels_list
        self.stride_list = stride_list
        self.num_classes = num_classes
        self.reg_max = reg_max
        proj = torch.linspace(0, self.reg_max, self.reg_max +1).reshape([1, reg_max + 1, 1, 1])
        self.register_buffer('proj_conv', proj, persistent=False)
        self.angle_min = angle_min
        self.angle_max = angle_max
        angle_proj = torch.linspace(self.angle_min, self.angle_max, self.angle_max - self.angle_min + 1). reshape([1, self.angle_max - self.angle_min + 1, 1, 1])
        self.register_buffer('angle_proj_conv', angle_proj, persistent=False)


        self.head1 = YoloNASDFLHead(in_channels=self.in_channels[0], inter_channels=self.inter_channels_list[0],
                                    width_mult=width_mult_list[0], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[0], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)

        self.head2 = YoloNASDFLHead(in_channels=self.in_channels[1], inter_channels=self.inter_channels_list[1],
                                    width_mult=width_mult_list[1], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[1], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)

        self.head3 = YoloNASDFLHead(in_channels=self.in_channels[2], inter_channels=self.inter_channels_list[2],
                                    width_mult=width_mult_list[2], first_conv_group_size=first_conv_group_size,
                                    num_classes=self.num_classes, stride=self.stride_list[2], reg_max=reg_max,
                                    angle_min=angle_min, angle_max=angle_max)


    def forward(self, feats):
        cls_score_list, reg_distri_list, reg_dist_reduced_list, angle_distri_list, angle_list = [], [], [], [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.size()
            reg_distri, cls_logit, angle_distri = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1])) # [bs, num_total_anchor, 4 * (reg_max + 1)]
            angle_distri_list.append(torch.permute(angle_distri.flatten(2), [0, 2, 1])) # [bs, num_total_anchor, 1 * (angle_max - angle_min + 1)]

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, h * w]), [0, 2, 3, 1]) # [bs, reg_max + 1, num_total_anchor, 4]
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1) # [bs, num_total_anchor, 4]


            angle = torch.permute(angle_distri.reshape([-1, 1, self.angle_max - self.angle_min + 1, h * w]), [0, 2, 3, 1])
            angle = torch.nn.functional.conv2d(torch.nn.functional.softmax(angle, dim=1), weight=self.angle_proj_conv).squeeze(1)

            cls_score_list.append(cls_logit.reshape([b, self.num_classes, h * w])) # [bs, num_classes, num_total_anchor]
            reg_dist_reduced_list.append(reg_dist_reduced)
            angle_list.append(angle)

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [bs, num_classes, num_total_anchor]
        pred_scores = torch.permute(cls_score_list, [0, 2, 1]).contiguous()  # # [bs, num_total_anchor, num_classes]

        pred_distri = torch.cat(reg_distri_list, dim=1)  # [bs, num_total_anchor, 4 * (self.reg_max + 1)]
        pred_bboxes = torch.cat(reg_dist_reduced_list, dim=1)  # [bs, num_total_anchor, 4]

        pred_angle_distri = torch.cat(angle_distri_list, dim=1) # [bs, num_total_anchor, 1 * (self.angle_max - self.angle_min + 1)]
        pred_angles = torch.cat(angle_list, dim=1) # [bs, num_total_anchor, 1]

        return pred_scores, pred_bboxes.contiguous(), pred_distri.contiguous(), pred_angles.contiguous(), pred_angle_distri.contiguous()

# new code
class RotatedNMSLayer(nn.Module):
    def __init__(self, num_classes, anchor_points, stride, iou_threshold=0.5):
        super(RotatedNMSLayer, self).__init__()
        self.iou_threshold = iou_threshold
        self.anchor_points = anchor_points
        self.stride_tensor = stride
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.half_pi_bin = np.pi / 180

    def forward(self, inputs):
        pred_scores, pred_bboxes, _, pred_angles, _, conf_threshold = inputs

        pred_bboxes = pred_bboxes.view(-1, 4)
        pred_angles = pred_angles.view(-1, 1)
        pred_scores = pred_scores.sigmoid().view(-1, self.num_classes)
        pred_conf = pred_scores.max(dim=-1)[0]
        pred_cls = torch.argmax(pred_scores, dim=-1)

        keep_indicates = pred_conf >= conf_threshold.view(-1)[0]

        pred_bboxes = self.dist2bbox(pred_bboxes, self.anchor_points) * self.stride_tensor

        pred_bboxes = pred_bboxes[keep_indicates]
        pred_angles = pred_angles[keep_indicates]
        pred_conf = pred_conf[keep_indicates]
        pred_cls = pred_cls[keep_indicates]

        #  (x1, y1, x2, y2, angle_degrees) format
        keep_indicates = torchvision.ops.batched_nms(pred_bboxes, pred_conf, pred_cls, self.iou_threshold)

        pred_bboxes = pred_bboxes[keep_indicates]
        pred_angles = pred_angles[keep_indicates]
        pred_cls = pred_cls.unsqueeze(-1)[keep_indicates].float()
        pred_conf  = pred_conf.unsqueeze(-1)[keep_indicates]

        outputs = torch.cat([torch.cat([pred_bboxes, pred_angles], dim=-1), pred_conf, pred_cls], axis=-1)
        return outputs

    def dist2bbox(self, distance, anchor_points):
        lt, rb = torch.split(distance, [2, 2], dim=-1)
        x1y1 = anchor_points - lt
        x2y2 = rb + anchor_points
        return torch.cat([x1y1, x2y2], dim=-1)

"""# **Model**"""

class RYoloNAS(nn.Module):
    def __init__(self, imgsz=(640, 640), num_classes=80, iou_threshold=0.5, backbone_data=None, neck_data=None, head_data=None,
                 angle_min=-180, angle_max=180):
        super(RYoloNAS, self).__init__()
        self.imgsz =imgsz
        if backbone_data is not None:
            self.backbone = NStageBackbone(
                out_stem_channels=backbone_data['out_stem_channels'],
                out_stage_channels_list=backbone_data['out_stage_channels_list'],
                hidden_channels_list=backbone_data['hidden_channels_list'],
                num_blocks_list=backbone_data['num_blocks_list'],
                concat_intermediates_list=backbone_data['concat_intermediates_list'],
                output_context_channels=backbone_data['output_context_channels'],
                k=backbone_data['k']
            )
        else:
            self.backbone = NStageBackbone()
        backbone_out_channels = self.backbone.out_channels
        if neck_data is not None:
            self.neck = YoloNASPANNeckWithC2(
                in_channels=backbone_out_channels,
                out_channels_list=neck_data['out_channels_list'],
                hidden_channels_list=neck_data['hidden_channels_list'],
                num_blocks_list=neck_data['num_blocks_list'],
                width_mult_list=neck_data['width_mult_list'],
                depth_mult_list=neck_data['depth_mult_list'],
            )
        else:
            self.neck = YoloNASPANNeckWithC2(in_channels=backbone_out_channels)

        neck_out_channels = self.neck.out_channels
        if head_data is not None:
            self.heads = NDFHeads(
                in_channels=neck_out_channels,
                num_classes=num_classes,
                inter_channels_list=head_data['inter_channels_list'],
                width_mult_list=head_data['width_mult_list'],
                angle_min=angle_min,
                angle_max=angle_max
            )
        else:
            self.heads = NDFHeads(in_channels=neck_out_channels, num_classes=num_classes, angle_min=angle_min, angle_max=angle_max)

        self.anchor_points, self.stride_tensor = make_anchors(imgsz=imgsz)
        self.nms_layer = RotatedNMSLayer(num_classes, self.anchor_points, self.stride_tensor, iou_threshold=iou_threshold)

    def forward(self, inputs, conf_threshold=None):
        x = self.backbone(inputs)
        x = self.neck(x)
        x = self.heads(x)
        if self.training:
            return x
        return self.nms_layer([*x, conf_threshold])


    def prep_model_for_conversion(self, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(self.imgsz, **kwargs)

    def replace_header(self):
        for module in self.modules():
            if module != self and hasattr(module, "replace_num_classes"):
                module.replace_num_classes()
                print('the header is replaced successfully!')

    def replace_forward(self):
        for module in self.modules():
            if module != self and hasattr(module, 'fuseforward'):
                module.forward = module.fuseforward

    def init_weight(self):
        count = 0
        for module in self.modules():
            if module != self and isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
                count += 1
            elif module != self and isinstance(module, nn.BatchNorm2d):
                torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(module.bias.data, 0.0)
                count += 1
        print('Count: ', count)

"""# **Dataset**"""

class YOLODataset(Dataset):
    def __init__(self):
        super(YOLODataset, self).__init__()
        self.annot_path = cfg.YOLO.TRAIN.ANNOT_PATH
        self.input_size = cfg.YOLO.TRAIN.MODEL_SIZE
        self.strides = cfg.YOLO.STRIDES
        self.classes = self.read_class_name(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.max_bbox_per_image = cfg.YOLO.TRAIN.MAX_BBOX_PER_IMAGE
        # new code
        self.num_samples = cfg.YOLO.TRAIN.N_SAMPLES_PER_EPOCH
        self.image_background_paths = sorted(glob.glob(cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR + "/*/*"))
        self.font_paths = sorted(glob.glob(cfg.GENERATE_FAKE_IMAGE.FONT_DIR + "/*"))
        self.font_size_min = cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN
        self.font_size_max = cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX
        self.word_count = cfg.GENERATE_FAKE_IMAGE.WORD_COUNT
        self.text_color = cfg.GENERATE_FAKE_IMAGE.TEXT_COLOR
        self.word_length_min = cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN
        self.word_length_max = cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX
        self.angle_min = cfg.GENERATE_FAKE_IMAGE.ANGLE_MIN
        self.angle_max = cfg.GENERATE_FAKE_IMAGE.ANGLE_MAX
        self.idx = range(self.num_samples)
        self.cls = self.read_class_name_v2(cfg.YOLO.CLASSES)


    def image_preprocess(self, image, target_size, keep_ratio=True, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)

        if keep_ratio:
            image_resized = cv2.resize(image, (nw, nh))
            image_padded = np.full(shape=[ih, iw, 3], fill_value=0.0)
            padding_position = np.random.randint(low=0, high=4)
            dw, dh = (iw - nw), (ih - nh)
            if padding_position == 0: # padding top & right
                image_padded[dh:nh+dh, :nw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
                return image_padded, gt_boxes
            elif padding_position == 1: # padding top & left
                image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
                return image_padded, gt_boxes
            elif padding_position == 2: # padding bottom & right
                image_padded[:nh, :nw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
                return image_padded, gt_boxes
            else: # padding bottom & left
                image_padded[:nh, dw:nw+dw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
                return image_padded, gt_boxes
        else:
            image_resized = cv2.resize(image, (iw, ih))
            if gt_boxes is None:
                return image_resized
            else:
                return image_resized, gt_boxes

    def read_class_name(self, class_file_path):
        names = {}
        with open(class_file_path, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def read_class_name_v2(self, class_file_path):
        names = {}
        with open(class_file_path, 'r') as data:
            for ID, name in enumerate(data):
                names[name.strip('\n')] =  ID
        return names


    def rect_polygon(self, xc, yc, w, h, angle, center_x, center_y):
        xc_rotated, yc_rotated = self.rotated_point(xc, yc, angle, center_x, center_y)
        angle_rad = angle * math.pi/180
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        x1 = int(xc_rotated - w/2 * cos_a - h/2 * sin_a)
        y1 = int(yc_rotated - w/2 * sin_a + h/2 * cos_a)
        x2 = int(xc_rotated + w/2 * cos_a - h/2 * sin_a)
        y2 = int(yc_rotated + w/2 * sin_a + h/2 * cos_a)
        x3 = int(xc_rotated + w/2 * cos_a + h/2 * sin_a)
        y3 = int(yc_rotated + w/2 * sin_a - h/2 * cos_a)
        x4 = int(xc_rotated - w/2 * cos_a + h/2 * sin_a)
        y4 = int(yc_rotated - w/2 * sin_a - h/2 * cos_a)
        a = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)])
        return a

    def check_IoU(self, xc1, yc1, w1, h1, angle1, bbox, angle2, center_x, center_y, IoU_threshold=0): #rect_b: angle = 0
        xc2, yc2, w2, h2 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]
        polygon2 = self.rect_polygon(xc2, yc2, w2, h2, angle2, center_x, center_y)
        for i in range(len(xc1)):
            polygon1 = self.rect_polygon(xc1[i], yc1[i], w1[i], h1[i], angle1[i], center_x, center_y)
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.area + polygon2.area - intersect
            iou = intersect / union
            if iou > IoU_threshold:
                return False
        return True

    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    def rotated_point(self, xc, yc, angle, center_x, center_y):
        angle_rad = math.radians(angle)
        xp = (xc - center_x) * math.cos(angle_rad) + (yc - center_y) * math.sin(angle_rad) + center_x
        yp =  - (xc - center_x) * math.sin(angle_rad) + (yc - center_y) * math.cos(angle_rad) + center_y
        return xp, yp

    def get_pos(self, font, text, imgsz):
        height, width = imgsz
        magic_number_1 = (sqrt(2) - 1)/(2*sqrt(2))
        magic_number_2 = 1 - magic_number_1
        text_width, text_height = font.getsize(text)
        pos_x_min = int(width * magic_number_1) + 1
        pos_x_max = int(width * magic_number_2 - text_width)
        pos_y_min = int(height * magic_number_1) + 1
        pos_y_max = int(height * magic_number_2 - text_height)
        pos = (random.randint(pos_x_min, pos_x_max), random.randint(pos_y_min, pos_y_max))
        return pos

    def generate_fake_image(self, image_background_path, font_paths, font_size_min, font_size_max,
                            word_count, text_color, imgsz, word_length_min, word_length_max,
                            angle_min, angle_max):

        font_size = random.randint(font_size_min, font_size_max)

        height, width = imgsz
        center_x, center_y = width / 2, height / 2
        if text_color is None:
            text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        word_length = [random.randint(word_length_min, word_length_max) for i in range(word_count)]
        text = [self.generate_random_string(word_length[k]) for k in range(word_count)]
        rotated_text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rotated_text_image)

        angle = []
        mu = (angle_min + angle_max) / 2
        sigma = (angle_max - angle_min) / 6
        for i in range(word_count):
            condition = True
            while condition:
                num = random.gauss(mu, sigma)
                if num >= angle_min and num <= angle_max:
                    angle.append(num)
                    condition = False

        xc = [[] for i in range(word_count)]
        yc = [[] for i in range(word_count)]
        w = [[] for i in range(word_count)]
        h =  [[] for i in range(word_count)]

        xc_word, yc_word, w_word, h_word = [], [], [], []

        patience_threshold = 30

        result = np.ones((width, height, 3), dtype=np.uint8)
        for j in range(word_count):
            font_path = np.random.choice(font_paths)
            rotated_text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            condition = True
            while condition:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    pos = self.get_pos(font, text[j], imgsz)
                    condition = False
                except:
                    font_size = font_size_min if font_size - 5 < font_size_min else font_size - 5
            condition_IoU = True
            patience = 0
            while condition_IoU:
                bbox = draw.textbbox((pos[0], pos[1]), text[j], font=font)
                if self.check_IoU(xc_word, yc_word, w_word, h_word, angle, bbox, angle[j], center_x, center_y):
                    xc_word.append(int((bbox[0] + bbox[2]) / 2))
                    yc_word.append(int((bbox[1] + bbox[3]) / 2))
                    w_word.append(int(bbox[2] - bbox[0]))
                    h_word.append(int(bbox[3] - bbox[1]))
                    patience = 0
                    condition_IoU = False
                else:
                    font_size = font_size_min if font_size - 5 < font_size_min else font_size - 5
                    font = ImageFont.truetype(font_path, font_size)
                    pos = self.get_pos(font, text[j], imgsz)
                    patience += 1
                    if patience == patience_threshold:
                        break
            if patience == patience_threshold:
                word_count = j
                break


            draw = ImageDraw.Draw(rotated_text_image)
            draw.text(pos, text[j], font=font, fill='black')
            x, y = pos
            for c in text[j]:
                bbox = draw.textbbox((x, y), c, font=font)
                xc[j].append(int((bbox[0] + bbox[2]) / 2))
                yc[j].append(int((bbox[1] + bbox[3]) / 2))
                w[j].append(int(bbox[2] - bbox[0]))
                h[j].append(int(bbox[3] - bbox[1]))
                x += draw.textlength(c, font=font)

            rotated_text_image = rotated_text_image.rotate(angle[j], expand=False, center=(center_x, center_y))

            image = Image.new('RGB', rotated_text_image.size, (255, 255, 255))
            image.paste(rotated_text_image, mask=rotated_text_image.split()[3])
            image = np.array(image)

            result = result * (image // 255)

        angle_rad = [(angle[j] * np.pi/180)  for j in range(word_count)]
        annot = []
        for i in range(word_count):
            xc_word[i], yc_word[i] = self.rotated_point(xc_word[i], yc_word[i], angle[i], center_x, center_y)
            annot.append([int(xc_word[i]), int(yc_word[i]), w_word[i], h_word[i], angle_rad[i], 0])
        annot = np.array(annot)

        # image_background = Image.open(image_background_path)
        # image_background = image_background.resize((image.shape[0], image.shape[1]))
        # image_background = np.array(image_background)

        image_background = cv2.imread(image_background_path) # 3 channels
        image_background = cv2.cvtColor(image_background, cv2.COLOR_BGR2RGB)
        image_background = cv2.resize(image_background, dsize=(image.shape[1], image.shape[0]))

        if image_background.ndim == 2:
            image_background = np.concatenate([image_background[..., None], image_background[..., None], image_background[..., None]], axis=-1)
        elif image_background.ndim == 4:
            image_background = image_background[..., :3]


        a = np.all(result, axis=2)
        for i in range(width):
            for j in range(height):
                if not(a[i][j]):
                    image_background[i][j] = np.array(text_color)
        return image_background, annot

    def random_colorjitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5):
        if np.random.uniform() < p:
            return img

        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        brightness = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        img[:, :, 2] = img[:, :, 2] * brightness

        contrast = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        img[:, :, 2] = img[:, :, 2] * contrast + (1 - contrast)

        saturation = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        img[:, :, 1] = img[:, :, 1] * saturation

        hue = np.random.uniform(-hue, hue)
        img[:, :, 0] = img[:, :, 0] + hue
        img[:, :, 0][img[:, :, 0] > 360] = img[:, :, 0][img[:, :, 0] > 360] - 360
        img[:, :, 0][img[:, :, 0] < 0] = img[:, :, 0][img[:, :, 0] < 0] + 360

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = img * 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        return img


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_background_path = np.random.choice(self.image_background_paths)

        image, bboxes = self.generate_fake_image(
                        image_background_path=image_background_path,
                        font_paths=self.font_paths,
                        font_size_min=self.font_size_min,
                        font_size_max=self.font_size_max,
                        word_count=np.random.randint(1, self.word_count),
                        text_color=self.text_color,
                        imgsz=self.input_size,
                        word_length_min=self.word_length_min,
                        word_length_max=self.word_length_max,
                        angle_min=self.angle_min,
                        angle_max=self.angle_max
        )

        image = self.random_colorjitter(image)

        # bboxes [xc, yc, w, h, angle, cls] -> [x1, y1, x2, y2, angle, cls]
        bboxes = xcycwha2xyxya(bboxes)
        image, bboxes = self.image_preprocess(np.copy(image), [*self.input_size], True, np.copy(bboxes))
        image = image / 255.

        label, boxes, mask = self.preprocess_true_boxes(bboxes)

        label = torch.tensor(label, dtype=torch.int32)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.int32)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute([2, 0, 1]).contiguous()
        return image, label, boxes, mask

    def preprocess_true_boxes(self, bboxes):
        label = np.zeros(shape=(self.max_bbox_per_image, 1), dtype=np.int32)
        boxes = np.zeros(shape=(self.max_bbox_per_image, 5), dtype=np.float32)
        mask = np.zeros(shape=(self.max_bbox_per_image, 1), dtype=np.int32)

        for i, bbox in enumerate(bboxes):
            bbox_coor = bbox[:5]
            bbox_class_ind = int(bbox[5])
            label[i, :] = bbox_class_ind
            boxes[i, :] = bbox_coor
            mask[i, :] = 1.0

        return label, boxes, mask

    def iter(self):
        for i in self.idx:
            yield self[i]


class YOLODatasetForFineTuningICDAR(Dataset):
    def __init__(self):
        super(YOLODatasetForFineTuningICDAR, self).__init__()
        self.data_aug = cfg.YOLO.TRAIN.DATA_AUG
        self.use_horizontal_flip = cfg.YOLO.TRAIN.HORIZONTAL_FLIP
        self.use_vertical_flip = cfg.YOLO.TRAIN.VERTICAL_FLIP
        self.use_random_crop = cfg.YOLO.TRAIN.RANDOM_CROP
        self.use_random_scale = cfg.YOLO.TRAIN.RANDOM_SCALE
        self.use_random_translate = cfg.YOLO.TRAIN.RANDOM_TRANSLATE
        self.use_random_rotate = cfg.YOLO.TRAIN.RANDOM_ROTATE
        self.use_colorjitter = cfg.YOLO.TRAIN.USE_COLORJITTER
        self.input_size = cfg.YOLO.TRAIN.MODEL_SIZE
        self.strides = cfg.YOLO.STRIDES
        self.classes = self.read_class_name(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.max_bbox_per_image = cfg.YOLO.TRAIN.MAX_BBOX_PER_IMAGE


        self.icdar13_image_dir = cfg.YOLO.TRAIN.ICDAR13_IMAGE_DIR_FOR_TRAINING
        self.icdar15_image_dir = cfg.YOLO.TRAIN.ICDAR15_IMAGE_DIR_FOR_TRAINING
        self.icdar17_image_dir = cfg.YOLO.TRAIN.ICDAR17_IMAGE_DIR_FOR_TRAINING
        self.icdar13_image_paths = sorted(glob.glob(os.path.join(self.icdar13_image_dir, '*')))
        self.icdar15_image_paths = sorted(glob.glob(os.path.join(self.icdar15_image_dir, '*')))
        self.icdar17_image_paths = sorted(glob.glob(os.path.join(self.icdar17_image_dir, '*')))

        self.icdar13_annot_dir = cfg.YOLO.TRAIN.ICDAR13_ANNOT_DIR
        self.icdar13_annot_paths = sorted(glob.glob(os.path.join(self.icdar13_annot_dir, '*')))
        self.icdar15_annot_dir = cfg.YOLO.TRAIN.ICDAR15_ANNOT_DIR
        self.icdar15_annot_paths = sorted(glob.glob(os.path.join(self.icdar15_annot_dir, '*')))
        self.icdar17_annot_dir = cfg.YOLO.TRAIN.ICDAR17_ANNOT_DIR
        self.icdar17_annot_paths = sorted(glob.glob(os.path.join(self.icdar17_annot_dir, '*')))

        self.total_text_image_dir = cfg.YOLO.TRAIN.TOTAL_TEXT_IMAGE_DIR_FOR_TRAINING
        self.total_text_image_paths = sorted(glob.glob(os.path.join(self.total_text_image_dir, '*')))

        self.total_text_annot_dir = cfg.YOLO.TRAIN.TOTAL_TEXT_ANNOT_PATH
        self.total_text_annot_paths = sorted(glob.glob(os.path.join(self.total_text_annot_dir, '*')))

        self.n_icdar13_samples = len(self.icdar13_image_paths)
        self.n_icdar15_samples = len(self.icdar15_image_paths)
        self.n_icdar17_samples = len(self.icdar17_image_paths)
        self.n_total_text_samples = len(self.total_text_image_paths)
        self.num_samples = self.n_icdar13_samples + self.n_icdar15_samples + self.n_icdar17_samples + self.n_total_text_samples
        # self.num_samples = self.n_icdar15_samples

        self.images_list = self.icdar13_image_paths + self.icdar15_image_paths + self.icdar17_image_paths + self.total_text_image_paths
        self.annot_list = self.icdar13_annot_paths + self.icdar15_annot_paths + self.icdar17_annot_paths + self.total_text_annot_paths

        # self.images_list = self.icdar15_image_paths
        # self.annot_list = self.icdar15_annot_paths


    def load_annotation13(self, annot_path):
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            annotation = [line.strip().split()[:-1] + ['0'] + ['0'] for line in txt]
        return annotation

    def parse_annotation13(self, annotation):
        bboxes = np.array([[int(float(value)) for value in box] for box in annotation])
        _bboxes = []
        for bbox in bboxes:
            if bbox[2] - bbox[0] !=0 and bbox[3] - bbox[1] !=0:
                _bboxes.append(bbox)
        bboxes = np.array(_bboxes)
        return bboxes # [x1, y1, x2, y2, a, cls]

    def load_annotation15(self, annot_path):
        annotation = []
        with open(annot_path, 'r', encoding='utf-8-sig') as f:
            txt = f.readlines()
            for line in txt:
                annot = line.strip().split(',')
                if annot[-1] != '###':
                    annotation.append(annot[:8])
        return annotation

    def convert_format_ICDAR2015(self, annot):        # N x 8
        out = []
        for a in annot:
            (xc, yc), (w, h), angle = cv2.minAreaRect(a.reshape(-1, 2))
            angle = angle / 180 * np.pi
            angle = -angle
            if w < h:
                w, h = h, w
                angle += np.pi/2
            out.append([xc, yc, w, h, angle])
        return np.array(out)                    # N x 5 (xc, yc, w, h, angle)

    def parse_annotation15(self, annotation):
        bboxes = np.array([[int(float(value)) for value in box] for box in annotation])
        bboxes = self.convert_format_ICDAR2015(bboxes) # xywha
        bboxes = xcycwha2xyxya(bboxes)
        new_boxes = np.zeros(shape=(bboxes.shape[0], 6), dtype=np.float32)
        new_boxes[:, :5] = bboxes
        return new_boxes  # xyxya

    def load_annotation17(self, annot_path):
        annotation = []
        with open(annot_path, 'r', encoding='utf-8-sig') as f:
            txt = f.readlines()
            for line in txt:
                annot = line.strip().split(',')
                if annot[-1] != '###' and annot[8] == 'Latin':
                    annotation.append(annot[:8])
        return annotation

    def convert_format_ICDAR2017(self, annot):        # N x 8
        out = []
        for a in annot:
            (xc, yc), (w, h), angle = cv2.minAreaRect(a.reshape(-1, 2))
            angle = angle / 180 * np.pi
            angle = -angle
            if w < h:
                w, h = h, w
                angle += np.pi/2
            out.append([xc, yc, w, h, angle])
        return np.array(out)                    # N x 5 (xc, yc, w, h, angle)

    def parse_annotation17(self, annotation):
        bboxes = np.array([[int(float(value)) for value in box] for box in annotation])
        bboxes = self.convert_format_ICDAR2017(bboxes) # xywha
        bboxes = xcycwha2xyxya(bboxes)
        new_boxes = np.zeros(shape=(bboxes.shape[0], 6), dtype=np.float32)
        new_boxes[:, :5] = bboxes
        return new_boxes  # xyxya

    def load_annotation_total_text(self, annot_path):
        annotation = []
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            pattern = '\d+'
            for line in txt:
                annot = line.split(',')
                if annot[2].split(' ')[-1] != '[u\'#\']':
                    a = re.findall(pattern, annot[0] + annot[1])
                    annotation.append(a)
        return annotation

    def parse_annot_total_text(self, annotation):
        bboxes = np.array([[int(float(value)) for value in box] for box in annotation])
        out = []
        for box in bboxes:
            poly = np.array(box).reshape(2, -1).T
            (xc, yc), (w, h), angle = cv2.minAreaRect(poly)
            angle = angle / 180 * np.pi
            angle = -angle
            if w < h:
                w, h = h, w
                angle += np.pi/2
            out.append([xc, yc, w, h, angle])
        bboxes_xywha = np.array(out)
        bboxes_xyxya = xcycwha2xyxya(bboxes_xywha)
        new_boxes = np.zeros(shape=(bboxes_xyxya.shape[0], 6), dtype=np.float32)
        new_boxes[:, :5] = bboxes_xyxya
        return new_boxes


    def image_preprocess(self, image, target_size, keep_ratio=True, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        if keep_ratio:
            image_padded = np.full(shape=[ih, iw, 3], fill_value=0.0)
            padding_position = np.random.randint(low=0, high=4)
            dw, dh = (iw - nw), (ih - nh)
            if padding_position == 0: # padding top & right
                image_padded[dh:nh+dh, :nw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
                return image_padded, gt_boxes
            elif padding_position == 1: # padding top & left
                image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
                return image_padded, gt_boxes
            elif padding_position == 2: # padding bottom & right
                image_padded[:nh, :nw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
                return image_padded, gt_boxes
            else: # padding bottom & left
                image_padded[:nh, dw:nw+dw, :] = image_resized
                if gt_boxes is None:
                    return image_padded
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
                return image_padded, gt_boxes
        else:
            if gt_boxes is None:
                return image_resized
            else:
                return image_resized, gt_boxes


    def read_class_name(self, class_file_path):
        names = {}
        with open(class_file_path, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names


    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
            bboxes[:, 4] = -bboxes[:, 4]
        return image, bboxes

    def random_vertical_flip(self, image, bboxes):
        if random.random() < 0.5:
            h, _, _ = image.shape
            image = image[::-1, :, :]
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
            bboxes[:, 4] = np.pi - bboxes[:, 4]
        return image, bboxes

    def random_scale(self, image, bboxes):
        if random.random() < 0.5:
            scale = random.choice([1.25, 1.5, 1.75, 2.0])
            h, w, _ = image.shape
            nh, nw = int(h * scale), int(w * scale)
            image = cv2.resize(image, dsize=(nw, nh))
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * nw / w
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * nh / h
            return image, bboxes
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def random_rotate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            new_bboxes = bboxes
            x1 = new_bboxes[:, 0]
            y1 = new_bboxes[:, 1]
            x2 = new_bboxes[:, 2]
            y2 = new_bboxes[:, 3]
            angles = new_bboxes[:, 4]

            angle = np.random.randint(-180, 180)
            rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
            for i in range(len(bboxes)):
                new_x1y1 = np.dot(rotation_matrix, np.array([[x1[i]], [y1[i]], [1]]))
                new_x2y2 = np.dot(rotation_matrix, np.array([[x2[i]], [y2[i]], [1]]))
                x1[i] = new_x1y1[0][0]
                y1[i] = new_x1y1[1][0]
                x2[i] = new_x2y2[0][0]
                y2[i] = new_x2y2[1][0]
                angles[i] -= np.deg2rad(angle)
                # angles[i] += np.deg2rad(angle)

            new_bboxes[:, 0] = x1
            new_bboxes[:, 1] = y1
            new_bboxes[:, 2] = x2
            new_bboxes[:, 3] = y2
            new_bboxes[:, 4] = angles % (2 * np.pi) - np.pi
            return image, new_bboxes

        return image, bboxes


    def random_colorjitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5):
        if np.random.uniform() < p:
            return img

        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        brightness = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        img[:, :, 2] = img[:, :, 2] * brightness

        contrast = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        img[:, :, 2] = img[:, :, 2] * contrast + (1 - contrast)

        saturation = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        img[:, :, 1] = img[:, :, 1] * saturation

        hue = np.random.uniform(-hue, hue)
        img[:, :, 0] = img[:, :, 0] + hue
        img[:, :, 0][img[:, :, 0] > 360] = img[:, :, 0][img[:, :, 0] > 360] - 360
        img[:, :, 0][img[:, :, 0] < 0] = img[:, :, 0][img[:, :, 0] < 0] + 360

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = img * 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path, annot_path = self.images_list[index], self.annot_list[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if 'icdar13' in annot_path:
            annot = self.load_annotation13(annot_path)
            bboxes = self.parse_annotation13(annot)
        elif 'icdar15' in annot_path:
            annot = self.load_annotation15(annot_path)
            while len(annot) == 0:
                index = random.randint(0, self.n_icdar15_samples - 1)
                image_path, annot_path = self.icdar15_image_paths[index], self.icdar15_annot_paths[index]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                annot = self.load_annotation15(annot_path)
            bboxes = self.parse_annotation15(annot)
        elif 'icdar17' in annot_path:
            annot = self.load_annotation17(annot_path)
            while len(annot) == 0:
                index = random.randint(0, self.n_icdar17_samples - 1)
                image_path, annot_path = self.icdar17_image_paths[index], self.icdar17_annot_paths[index]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                annot = self.load_annotation17(annot_path)
            bboxes = self.parse_annotation17(annot)
        else:
            annot = self.load_annotation_total_text(annot_path)
            while len(annot) == 0:
                index = random.randint(0, self.n_total_text_samples - 1)
                image_path, annot_path = self.total_text_image_paths[index], self.total_text_annot_paths[index]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                annot = self.load_annotation_total_text(annot_path)
            bboxes = self.parse_annot_total_text(annot)


        if self.data_aug:
            if self.use_horizontal_flip:
                image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            if self.use_vertical_flip:
                image, bboxes = self.random_vertical_flip(np.copy(image), np.copy(bboxes))
            if self.use_random_crop:
                image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            if self.use_random_scale:
                image, bboxes = self.random_scale(np.copy(image), np.copy(bboxes))
            if self.use_random_translate:
                image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
            if self.use_random_rotate:
                image, bboxes = self.random_rotate(np.copy(image), np.copy(bboxes))
            if self.use_colorjitter:
                image = self.random_colorjitter(np.copy(image))

        self.current_annot_path = annot_path
        self.current_image_path = image_path
        # print('\nannot_path: ', annot_path)
        image, bboxes = self.image_preprocess(np.copy(image), [*self.input_size], True, np.copy(bboxes))
        image = image / 255.

        label, boxes, mask = self.preprocess_true_boxes(bboxes)

        label = torch.tensor(label, dtype=torch.int32)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.int32)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute([2, 0, 1]).contiguous()
        return image, label, boxes, mask

    def preprocess_true_boxes(self, bboxes):
        label = np.zeros(shape=(self.max_bbox_per_image, 1), dtype=np.int32)
        boxes = np.zeros(shape=(self.max_bbox_per_image, 5), dtype=np.float32)
        mask = np.zeros(shape=(self.max_bbox_per_image, 1), dtype=np.int32)

        for i, bbox in enumerate(bboxes):
            bbox_coor = bbox[:5]
            bbox_class_ind = int(bbox[5])
            label[i, :] = bbox_class_ind
            boxes[i, :] = bbox_coor
            mask[i, :] = 1.0

        return label, boxes, mask

    def iter(self):
        for i in self.idx:
            yield self[i]

"""# **R-TAL Assingner**"""

class RotatedTaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, iou_type='ciou', eps=1e-9):
        super(RotatedTaskAlignedAssigner, self).__init__()
        self.topk =topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.bg_idx = num_classes
        self.iou_type = iou_type

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        '''
          pd_bboxes: format [x1, y1, x2, y2, angle_rad]
          gt_boxes: format [x1, y1, x2, y2, angle_rad]
        '''
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes[:, :, :4].view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)

        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())

        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.long().squeeze(-1)

        bbox_scores = pd_scores[ind[0], :, ind[1]]

        if self.iou_type == 'ciou':
            overlaps = bbox_iou(gt_bboxes[..., :4].unsqueeze(2), pd_bboxes[..., :4].unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        elif self.iou_type == 'giou':
            overlaps = bbox_iou(gt_bboxes[..., :4].unsqueeze(2), pd_bboxes[..., :4].unsqueeze(1), xywh=False, GIoU=True).squeeze(3).clamp(0)
        elif self.iou_type == 'siou':
            overlaps = bbox_iou(gt_bboxes[..., :4].unsqueeze(2), pd_bboxes[..., :4].unsqueeze(1), xywh=False, SIOU=True).squeeze(3).clamp(0)
        else: # diou
            overlaps = bbox_iou(gt_bboxes[..., :4].unsqueeze(2), pd_bboxes[..., :4].unsqueeze(1), xywh=False, DIoU=True).squeeze(3).clamp(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        target_bboxes = gt_bboxes.view(-1, 5)[target_gt_idx]

        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        num_anchors = metrics.shape[-1]

        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])

        topk_idxs = torch.where(topk_mask, topk_idxs, 0)

        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)

        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)

        return is_in_topk.to(metrics.dtype)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

"""# **loss**"""

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, angle_min=-90, angle_max=90, use_dfl=True, iou_type='ciou'):
        super(BboxLoss, self).__init__()
        self.reg_max = reg_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.use_dfl = use_dfl
        self.iou_type = iou_type
        self.half_pi_bin = np.pi / 180
        self.dfl_gain = [1.0, 1.0] # bboxes, angles


    def forward(self, pred_bboxes_dist, pred_bboxes, pred_angles_distri, pred_angles, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask):
        '''
            pred_bboxes: format x1 y1 x2 y2, angle_rad
            target_bboxes: format x1, y1, x2,y2, angle_rad
        '''
        # IOU Loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        if self.iou_type == 'ciou':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask][..., :4], xywh=False, CIoU=True)
        elif self.iou_type == 'giou':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask][..., :4], xywh=False, GIoU=True)
        elif self.iou_type == 'siou':
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask][..., :4], xywh=False, SIOU=True)
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask][..., :4], xywh=False, DIoU=True)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        l1 = ((pred_angles[fg_mask].view(-1, 1) - target_bboxes[fg_mask][..., 4].view(-1, 1)).abs() * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            # [x, y, x, y]
            target_ltrb = bbox2dist(anchor_points, target_bboxes[..., :4], self.reg_max)

            loss_dfl_for_bboxes = self._df_loss(pred_bboxes_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight

            loss_dfl_for_bboxes = loss_dfl_for_bboxes.sum() / target_scores_sum

            # angles
            target_angles = target_bboxes[..., 4:5]

            loss_dfl_for_angles = self._df_loss(pred_angles_distri[fg_mask].view(-1, self.angle_max - self.angle_min + 1) + abs(self.angle_min),
                                    (target_angles[fg_mask] / self.half_pi_bin).clamp(self.angle_min, self.angle_max - 0.01) + abs(self.angle_min)) * weight

            loss_dfl_for_angles = loss_dfl_for_angles.sum() / target_scores_sum

            # new code
            loss_dfl = loss_dfl_for_bboxes * self.dfl_gain[0] + loss_dfl_for_angles * self.dfl_gain[1]
        else:
            loss_dfl = torch.tensor(0.0).to(pred_bboxes_dist.device)

        return loss_iou, loss_dfl, l1

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tr.shape) * wr).mean(-1, keepdim=True)


class Loss:
    def __init__(self, imgsz=(640, 640), topk=13, num_classes=80, use_dfl=True, iou_type='ciou',
                 angle_min=-90, angle_max=90):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.use_dfl = use_dfl
        self.reg_max = 16
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.topk = topk
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.iou_type = iou_type
        self.assigner = RotatedTaskAlignedAssigner(topk=topk, num_classes=num_classes, alpha=0.5, beta=6.0, iou_type=iou_type).to(self.device)
        self.bbox_loss = BboxLoss(reg_max=self.reg_max, angle_min=self.angle_min, angle_max=self.angle_max,
                                  use_dfl=self.use_dfl, iou_type=iou_type).to(self.device)
        self.anchor_points, self.stride_tensor = make_anchors(imgsz)
        self.gain_list = [0.5, 7.5, 1.5, 15.0]

    def bbox_decode(self, pred_bboxes, anchor_points):
        return dist2bbox(pred_bboxes, anchor_points, xywh=False)


    def __call__(self, preds, gt_labels, gt_bboxes, mask_gt):
        loss = torch.zeros(4, device=self.device)

        pred_scores, pred_bboxes, pred_bboxes_distri, pred_angles, pred_angles_distri = preds

        dtype = pred_scores.dtype

        batch_size = pred_scores.shape[0]

        pred_bboxes = self.bbox_decode(pred_bboxes, self.anchor_points)  # bs x h*w x 4 with format xyxy

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.sigmoid(),
                                                                    torch.cat([pred_bboxes * self.stride_tensor, pred_angles], dim=-1),
                                                                    self.anchor_points * self.stride_tensor, gt_labels,
                                                                    gt_bboxes, mask_gt)

        target_bboxes[..., :4] = target_bboxes[..., :4] / self.stride_tensor # format [x, y, x, y, angle_rad]

        target_scores_sum = target_scores.sum()

        # cls loss
        loss[0] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum


        # bbox loss
        if fg_mask.sum():
            loss[1], loss[2], loss[3] = self.bbox_loss(pred_bboxes_distri, pred_bboxes, pred_angles_distri, pred_angles, self.anchor_points,
                                                      target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss[0] *= self.gain_list[0]  # cls gain
        loss[1] *= self.gain_list[1]  # box gain
        loss[2] *= self.gain_list[2]  # dfl gain
        loss[3] *= self.gain_list[3]  # l1 gain

        return loss.sum() * batch_size, loss[0].detach(), loss[1].detach(), loss[2].detach(), loss[3].detach()

"""# **Optimizer**"""

class SophiaG(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho = 0.04, weight_decay=1e-1, *, maximize=False, capturable=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)


    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  bs=bs,
                  beta1=beta1,
                  beta2=beta2,
                  rho=group['rho'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss


def sophiag(params, grads, exp_avgs, hessian, state_steps, capturable=False, *, bs, beta1, beta2, rho, lr, weight_decay, maximize):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag
    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_sophiag(params, grads, exp_avgs, hessian, state_steps, *, bs, beta1, beta2, rho, lr, weight_decay, maximize, capturable):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step = step_t
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)




# Gradient Centralization for Adam optimizer
def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
            else:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,use_gc=False, gc_conv_only=False,gc_loc=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        self.gc_loc=gc_loc
        self.use_gc=use_gc
        self.gc_conv_only=gc_conv_only

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                if self.gc_loc:
                   grad=centralized_gradient(grad,use_gc=self.use_gc,gc_conv_only=self.gc_conv_only)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                #GC operation
                G_grad=exp_avg/denom
                if self.gc_loc==False:
                    G_grad=centralized_gradient(G_grad,use_gc=self.use_gc,gc_conv_only=self.gc_conv_only)

                p.add_( G_grad, alpha=-step_size)

        return loss


# Gradient Centralization for SGD optimizer
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,use_gc=False, gc_conv_only=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, use_gc=use_gc,gc_conv_only=gc_conv_only)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                #GC operation
                d_p =centralized_gradient(d_p ,use_gc=group['use_gc'],gc_conv_only=group['gc_conv_only'])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf


                p.add_(d_p, alpha=-group['lr'])

        return loss



# MirrorMADGRAD optimizer
class MirrorMADGRAD(torch.optim.Optimizer):
    def __init__(self, params, lr= 1e-2, momentum=0.9, weight_decay=0, eps=0, decouple_decay=False):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr < 0:
            raise ValueError(f"Learning rate {lr} must be non-negative")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum,
            weight_decay=weight_decay, decouple_decay=decouple_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure=None)-> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        k = self.state['k'].item()

        update_ratio = math.pow(k/(k+1), 1/2)
        lamb = math.pow(k+1, 1/3)

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            if lr != 0.0:
                lr = lr + eps # For stability
            decay = group["weight_decay"]
            momentum = group["momentum"]
            decouple_decay = group.get("decouple_decay", False)

            ck = 1 - momentum

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                state = self.state[p]

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()


                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p_data_fp32).detach()
                    state["z"] = torch.clone(p_data_fp32).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                z = state["z"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    if decouple_decay:
                        z.data.add_(z.data, alpha=-lr*decay)
                    else:
                        grad.add_(p_data_fp32, alpha=decay)

                grad_sum_sq.mul_(update_ratio)
                # Accumulate second moments
                grad_sum_sq.addcmul_(grad, grad, value=1)
                rms = grad_sum_sq.pow(1 / 3).add_(eps)

                if eps == 0:
                    rms[rms == 0] = float('inf')

                # Update z
                z.data.addcdiv_(grad, rms, value=-lr*lamb)

                # Step
                p_data_fp32.mul_(1 - ck).add_(z, alpha=ck)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        self.state['k'] += 1
        return loss



''' Amos Optimizer '''
class Amos(Optimizer, BaseOptimizer):
    r"""An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. A float slightly < 1. We recommend setting `1 - beta` to the same order of magnitude
        as the learning rate. similarity with beta2 in Adam.
    :param momentum: float. Exponential decay rate for optional moving average of updates.
    :param extra_l2: float. Additional L2 regularization.
    :param c_coef: float. Coefficient for decay_factor_c.
    :param d_coef: float. Coefficient for decay_factor_d.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        eps: float = 1e-18,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(extra_l2, 'extra_l2')
        self.validate_non_negative(eps, 'eps')

        self.c_coef = c_coef
        self.d_coef = d_coef

        defaults: DEFAULTS = {
            'lr': lr,
            'beta': beta,
            'momentum': momentum,
            'extra_l2': extra_l2,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Amos'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg_sq'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                state['decay'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['momentum'] > 0.0:
                    state['exp_avg'] = torch.zeros_like(p)

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        r"""Get expected scale for model weights."""
        if len(p.shape) == 1:  # expected 'bias'
            return 0.5
        if len(p.shape) == 2:  # expected Embedding, Linear, ...
            return math.sqrt(2 / p.size(1))
        return math.sqrt(1 / p.size(1))


    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            momentum, beta = group['momentum'], group['beta']

            lr_sq: float = math.sqrt(group['lr'])
            bias_correction: float = 1.0 - beta ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg_sq'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    state['decay'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    if group['momentum'] > 0.0:
                        state['exp_avg'] = torch.zeros_like(p)

                g2 = grad.pow(2).mean()
                init_lr: float = group['lr'] * self.get_scale(p)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta).add_(g2, alpha=1.0 - beta)

                r_v_hat = bias_correction / (exp_avg_sq + group['eps'])

                b = state['decay']
                decay_factor_c = torch.rsqrt(1.0 + self.c_coef * lr_sq * b)
                decay_factor_d = torch.reciprocal(1.0 + self.d_coef * math.sqrt(init_lr) * b)

                gamma = decay_factor_c * (group['lr'] ** 2) * r_v_hat * g2

                update = p.clone()
                update.mul_((gamma - group['extra_l2']) / 2.0)
                update.add_(r_v_hat.sqrt() * grad, alpha=init_lr)
                update.mul_(decay_factor_d)

                b.mul_(1.0 + gamma).add_(gamma)

                if momentum > 0.0:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(momentum).add_(update, alpha=1.0 - momentum)

                    update.copy_(exp_avg)

                p.add_(-update)

        return loss

"""# **Trainer**"""

class YOLOTrainer():
    def __init__(self, imgsz=(640, 640), epochs=100, topk=13, batch_size=8, saved_model_dir='', model_type='s', transfer='transfer',
                 optimizer_type='sophiag', lr_init=1e-3, class_file='', iou_threshold=0.5, conf_threshold=0.5, patience=10,
                 iou_type='siou', angle_min=0, angle_max=360, use_valid_dataset=False, image_dir_for_testing='', visual_learning_process=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes_map = read_class_names(class_file)
        self.num_classes = len(self.classes_map)
        self.input_size = imgsz
        self.saved_model_dir = saved_model_dir
        self.epochs = epochs
        self.model_type = model_type
        self.transfer = transfer
        self.optimizer_type = optimizer_type
        self.lr_init = lr_init
        self.visual_learning_process = visual_learning_process
        self.topk = topk
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.patience = patience
        self.iou_type = iou_type
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.use_valid_dataset = use_valid_dataset
        self.dataset_dir_for_testing = image_dir_for_testing
        self.test_image_paths = sorted(glob.glob(image_dir_for_testing + '/*'))
        self.build_model()

    def build_model(self):
        print(f'Creating {self.model_type} yolo-nas model....')
        self.yaml_path, self.weight_path = get_yaml_and_weight_path(self.model_type)
        backbone_data, neck_data, head_data = get_model_info_from_yaml(self.yaml_path)
        self.model = RYoloNAS(imgsz=self.input_size, num_classes=self.num_classes, iou_threshold=self.iou_threshold,
                             backbone_data=backbone_data, neck_data=neck_data, head_data=head_data,
                             angle_min=self.angle_min, angle_max=self.angle_max)
        self.model.init_weight()
        self.loss_fn = Loss(imgsz=self.input_size, topk=self.topk, num_classes=self.num_classes, iou_type=self.iou_type,
                            angle_min=self.angle_min, angle_max=self.angle_max)

    def train_step(self, images, gt_labels, gt_bboxes, gt_mask):
        self.model.train()

        preds = self.model(images)
        total_loss, cls_loss, box_loss, dfl_loss, l1_loss = self.loss_fn(preds, gt_labels, gt_bboxes, gt_mask)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), cls_loss.item(), box_loss.item(), dfl_loss.item(), l1_loss.item()


    def train(self, train_dataset):
        prev_epoch_loss = None

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=0, pin_memory=True)

        if self.transfer == 'transfer':
            print("Transfer learning for model")
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            checkpoint_model = self.model.state_dict()
            new_checkpoint = get_new_state_dict(checkpoint['net'], checkpoint_model)
            self.model.load_state_dict(new_checkpoint)
            self.model.replace_header()
        elif self.transfer == 'resume':
            print("Load weights from latest checkpoint")
            latest_weight_path = os.path.join(self.saved_model_dir, 'model.pth')
            checkpoint = torch.load(latest_weight_path, map_location=self.device)
            self.model.replace_header()
            self.model.load_state_dict(checkpoint)
        else:
            print("Training model from scratch")
            self.model.replace_header()

        print(summary(
                      model=self.model,
                      input_size=[(1, 3, *self.input_size), (1,)],
                      col_names=['input_size', 'output_size', 'num_params', 'trainable'],
                      col_width=20,
                      row_settings=['var_names']
        ))

        # optimizer
        if self.optimizer_type == 'sophia':
            self.optimizer = SophiaG(self.model.parameters(), lr=self.lr_init)
        elif self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_init, weight_decay=1e-5)
        elif self.optimizer_type == 'adamgc':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr_init, use_gc=True, gc_conv_only=False, gc_loc=False)
        elif self.optimizer_type == 'sgdgc':
            self.optimizer = SGD(self.model.parameters(), lr=self.lr_init, momentum=0.9, use_gc=True, gc_conv_only=False)
        elif self.optimizer_type == 'mirrormadgrad':
            self.optimizer = MirrorMADGRAD(self.model.parameters(), lr=self.lr_init)
        elif self.optimizer_type == 'amos':
            self.optimizer = Amos(self.model.parameters(), lr=self.lr_init)
        else:
            print('This optimizer type is not supported yet!!!')

        self.model.to(self.device)

        current_patience = 0
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            count = 0
            stream = tqdm(train_dataloader)
            if self.visual_learning_process and epoch % 1 == 0 and epoch !=1:
                self.visual_learning_process_fn()
            for step, (images, gt_labels, gt_bboxes, gt_mask) in enumerate(stream):
                images, gt_labels, gt_bboxes, gt_mask = images.to(self.device), gt_labels.to(self.device), gt_bboxes.to(self.device), gt_mask.to(self.device)
                self.image_test = images
                self.gt_boxes = gt_bboxes
                self.gt_labels = gt_labels

                loss, cls_loss, box_closs, dfl_loss, l1_loss = self.train_step(images, gt_labels, gt_bboxes, gt_mask)
                stream.set_description(
                    'TRAIN [{:>4d}/{}] '
                    'cls_loss: {cls_loss:>6.3f} '
                    'box_closs: {box_closs:>6.3f} '
                    'dfl_loss: {dfl_loss:>6.3f} '
                    'l1_loss: {l1_loss:>6.3f} '
                    'LR: {lr:.3e} '.format(
                        epoch, self.epochs,
                        cls_loss=cls_loss,
                        box_closs=box_closs,
                        dfl_loss=dfl_loss,
                        l1_loss=l1_loss,
                        lr=self.optimizer.param_groups[0]['lr']
                    )
                )
                count += 1

                total_loss += loss

            if epoch % 1 == 0:
                loss_mean = total_loss / count
                if prev_epoch_loss is None or prev_epoch_loss > loss_mean:
                    prev_epoch_loss = loss_mean
                    print('Saving weights for epoch {} at {}'.format(epoch, os.path.join(self.saved_model_dir, "model.pth")))
                    torch.save(self.model.state_dict(), os.path.join(self.saved_model_dir, 'model.pth'))
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience == self.patience:
                        break

        self.export_model()
        return self.model

    def export_model(self):
        print('Quantizating model...')
        checkpoint = torch.load(os.path.join(self.saved_model_dir, 'model.pth'), map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.prep_model_for_conversion()
        fuse_conv_bn(self.model, False)
        self.model.replace_forward()
        torch.save(self.model.state_dict(), os.path.join(self.saved_model_dir, 'qamodel.pth'))

        print('Exporting ONNX model...')
        input_shape = (1, 3, *self.input_size)
        input_conf_threshold_shape = (1,)
        input_data = torch.randn(input_shape, requires_grad=False).to(self.device)
        input_conf_threshold = torch.randn(input_conf_threshold_shape, dtype=torch.float32, requires_grad=False).to(self.device)

        onnx_path = os.path.join(self.saved_model_dir, 'model.onnx')

        torch.onnx.export(
            self.model,
            (input_data, input_conf_threshold),
            onnx_path,
            export_params=True,
            opset_version=12,
            input_names=['input_data', 'input_conf_threshold'],
            output_names=['output'],
        )
        print('Done!!!')


    def visual_learning_process_fn(self):
        self.model.eval()
        conf_threshold = torch.tensor([self.conf_threshold], dtype=torch.float32).to(self.device)
        # new code
        if self.use_valid_dataset:
            image_path = random.choice(self.test_image_paths)
            # image_input = Image.open(image_path)
            # image_input = image_input.resize((self.input_size[0], self.input_size[1]))
            # image_input = np.array(image_input, 'uint8')
            image_input = cv2.imread(image_path)
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image_input = cv2.resize(image_input, dsize=(self.input_size[1], self.input_size[0]))
            if image_input.ndim == 2:
                image_input = np.concatenate([image_input[..., None], image_input[..., None], image_input[..., None]], axis=-1)
            elif image_input.ndim == 4:
                image_input = image_input[..., :3]
            image = image_input
            image_input = image_input / 255.
            image_input = torch.tensor(image_input, dtype=torch.float32)
            image_input = image_input.permute([2, 0, 1]).contiguous().to(self.device) # c, h, w
            word_boxes = self.model(image_input[None], conf_threshold)
        else:
            image_input = self.image_test[0]
            image = image_input * 255.
            image = image.cpu().numpy()
            image = np.transpose(image, [1, 2, 0])
            image = np.ascontiguousarray(image, dtype=np.uint8)
            word_boxes = self.model(image_input[None], conf_threshold)
        word_boxes = word_boxes.cpu().detach().numpy() # [num_boxes, 7] [x, y, x, y, angle, conf, cls]

        img = image
        if not self.use_valid_dataset:
            image_gt = self.draw_rotated_box(np.array(img), None, None, None, num_word=0, gt_boxes=self.gt_boxes[0].cpu().numpy(),
                                            gt_label=self.gt_labels[0].cpu().numpy())
            cv2.imwrite("./original_img_with_epoch.jpg",  cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

        image_predict = self.draw_rotated_box(np.array(img), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
        cv2.imwrite("./predict_img_with_epoch.jpg",  cv2.cvtColor(image_predict, cv2.COLOR_RGB2BGR))



    def rotate_rectangle(self, image, xc, yc, w, h, angle_rad, color=(255, 0, 0), thickness=1):
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        x1 = (xc - w/2 * cos_a - h/2 * sin_a)
        y1 = (yc - w/2 * sin_a + h/2 * cos_a)
        x2 = (xc + w/2 * cos_a - h/2 * sin_a)
        y2 = (yc + w/2 * sin_a + h/2 * cos_a)
        x3 = (xc + w/2 * cos_a + h/2 * sin_a)
        y3 = (yc + w/2 * sin_a - h/2 * cos_a)
        x4 = (xc - w/2 * cos_a + h/2 * sin_a)
        y4 = (yc - w/2 * sin_a - h/2 * cos_a)
        try:
            for x1, y1, x2, y2, x3, y3, x4, y4 in zip(x1, y1, x2, y2, x3, y3, x4, y4):
                image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
                image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
                image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
                image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
        except:
            image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
            image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
            image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
            image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
        return image

    def draw_rotated_box(self, image, word_boxes=None, char_boxes=None, group_chars=None, num_word=0, gt_boxes=None, gt_label=None):
        colors = [(255, 0, 0), (0, 0, 255)] # char_color, word_color

        image_h, image_w, _ = image.shape

        bbox_thick = int(0.6 * (image_h + image_w) / 600)

        char_color = colors[0]
        word_color = colors[1]

        classes = gt_label

        if gt_boxes is not None:
            word_boxes = gt_boxes

        xc, yc, w, h, angle, bbox_color = self.convert_format(word_boxes, char_color, word_color, for_chars=False)
        image = self.rotate_rectangle(image, xc, yc, w, h, angle, bbox_color, bbox_thick)

        return image

    def convert_format(self, box, char_color, word_color=None, for_chars=True):
        xc = (box[:, 2] - box[:, 0]) / 2.0 + box[:, 0]
        yc = (box[:, 3] - box[:, 1]) / 2.0 + box[:, 1]
        w = (box[:, 2] - box[:, 0])
        h = (box[:, 3] - box[:, 1])
        angle = box[:, 4]
        bbox_color = char_color if for_chars else word_color
        return xc, yc, w, h, angle, bbox_color

"""# **test trainer**"""

# cfg.YOLO.STRIDES = [8, 16, 32]
# cfg.YOLO.CLASSES = '/content/drive/MyDrive/YOLO-NAS/cls.txt'
# cfg.YOLO.YAML_PATH = ''
# cfg.YOLO.NUM_CLASSES = 1
# cfg.YOLO.TOPK = 13
# cfg.YOLO.MODEL_TYPE = 's' # [s, m, l]
# cfg.YOLO.ANGLE_MIN = -180
# cfg.YOLO.ANGLE_MAX = 180

# cfg.YOLO.BATCH_SIZE = 8
# cfg.YOLO.TRAIN.EPOCHS = 500
# cfg.YOLO.TRAIN.EPOCHS_FOR_PRETRAINING = 3
# cfg.YOLO.TRAIN.EPOCHS_FOR_FINE_TUNING = 200
# cfg.YOLO.TRAIN.MODEL_SIZE = (640, 640)
# cfg.YOLO.TRAIN.ANNOT_PATH = ''
# cfg.YOLO.TRAIN.SAVED_MODEL_DIR = '/content/drive/MyDrive/YOLO-NAS/model/pretrained_small_model/'
# cfg.YOLO.TRAIN.DATA_AUG = True
# cfg.YOLO.TRAIN.HORIZONTAL_FLIP = True
# cfg.YOLO.TRAIN.VERTICAL_FLIP = False
# cfg.YOLO.TRAIN.RANDOM_CROP = True
# cfg.YOLO.TRAIN.RANDOM_TRANSLATE = True
# cfg.YOLO.TRAIN.RANDOM_ROTATE = True
# cfg.YOLO.TRAIN.USE_COLORJITTER = True
# cfg.YOLO.TRAIN.LR_INIT = 1e-4
# cfg.YOLO.TRAIN.OPTIMIZER_TYPE = 'adam'
# cfg.YOLO.TRAIN.VISUAL_LEARNING_PROCESS = True
# cfg.YOLO.TRAIN.TRANSFER = 'transfer' #'transfer'
# cfg.YOLO.TRAIN.ADD_IMG_PATH = ''
# cfg.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES = 'siou' # probiou
# cfg.YOLO.TRAIN.CONF_THRESHOLD = 0.5
# cfg.YOLO.TRAIN.IOU_THRESHOLD = 0.25
# cfg.YOLO.TRAIN.MAX_BBOX_PER_IMAGE = 310
# cfg.YOLO.TRAIN.BACKBONE_DATE = None
# cfg.YOLO.TRAIN.NECK_DATA = None
# cfg.YOLO.TRAIN.PATIENCE = 20
# cfg.YOLO.TRAIN.N_SAMPLES_PER_EPOCH = 20000
# cfg.YOLO.TRAIN.USE_VALID_DATASET = False
# # cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = './icdar_dataset/icdar17/test/images'
# cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = '/content/drive/MyDrive/YOLO-NAS/ICDAR2015'


# # generate fake image
# cfg.GENERATE_FAKE_IMAGE = edict()
# cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR = '/content/drive/MyDrive/ImageNet/imagenet-10'
# # cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR = './imagenet_100k_512px'
# cfg.GENERATE_FAKE_IMAGE.FONT_DIR = "/content/drive/MyDrive/Fonts"
# cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN = 30
# cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX = 500
# cfg.GENERATE_FAKE_IMAGE.TEXT_COLOR = None
# cfg.GENERATE_FAKE_IMAGE.IMAGE_SIZE = (640, 640)
# cfg.GENERATE_FAKE_IMAGE.WORD_COUNT = 6
# cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN = 1
# cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX = 7
# cfg.GENERATE_FAKE_IMAGE.ANGLE_MIN = -180
# cfg.GENERATE_FAKE_IMAGE.ANGLE_MAX =  180


# # icdar
# cfg.YOLO.TRAIN.ICDAR13_ANNOT_DIR = 'icdar_dataset/icdar13/Challenge2_Training_Task1_GT'
# cfg.YOLO.TRAIN.ICDAR13_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar13/Challenge2_Training_Task12_Images'
# cfg.YOLO.TRAIN.ICDAR15_ANNOT_DIR = 'icdar_dataset/icdar15/ch4_training_localization_transcription_gt'
# cfg.YOLO.TRAIN.ICDAR15_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar15/ch4_training_images'
# cfg.YOLO.TRAIN.ICDAR17_ANNOT_DIR = 'icdar_dataset/icdar17/train/gt'
# cfg.YOLO.TRAIN.ICDAR17_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar17/train/images'

# # total text
# cfg.YOLO.TRAIN.TOTAL_TEXT_ANNOT_PATH = 'total_text_dataset/gt/Train'
# cfg.YOLO.TRAIN.TOTAL_TEXT_IMAGE_DIR_FOR_TRAINING = 'total_text_dataset/Images/Train'


# # Pretrain model

# # cfg.YOLO.TRAIN.EPOCHS = cfg.YOLO.TRAIN.EPOCHS_FOR_PRETRAINING
# # cfg.YOLO.TRAIN.TRANSFER = 'transfer'

# # train_dataset = YOLODataset()

# # classes_map = train_dataset.classes

# # trainer = YOLOTrainer(imgsz=cfg.YOLO.TRAIN.MODEL_SIZE, epochs=cfg.YOLO.TRAIN.EPOCHS, topk=cfg.YOLO.TOPK, batch_size=cfg.YOLO.BATCH_SIZE,
# #                       saved_model_dir=cfg.YOLO.TRAIN.SAVED_MODEL_DIR, model_type=cfg.YOLO.MODEL_TYPE, transfer=cfg.YOLO.TRAIN.TRANSFER,
# #                       optimizer_type=cfg.YOLO.TRAIN.OPTIMIZER_TYPE, lr_init=cfg.YOLO.TRAIN.LR_INIT, class_file=cfg.YOLO.CLASSES,
# #                       iou_threshold=cfg.YOLO.TRAIN.IOU_THRESHOLD, conf_threshold=cfg.YOLO.TRAIN.CONF_THRESHOLD,
# #                       patience=cfg.YOLO.TRAIN.PATIENCE, iou_type=cfg.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES,
# #                       angle_min=cfg.YOLO.ANGLE_MIN, angle_max=cfg.YOLO.ANGLE_MAX,
# #                       use_valid_dataset=cfg.YOLO.TRAIN.USE_VALID_DATASET , image_dir_for_testing=cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING,
# #                       visual_learning_process=cfg.YOLO.TRAIN.VISUAL_LEARNING_PROCESS)

# # model = trainer.train(train_dataset)


# # fine-tuning model

# cfg.YOLO.TRAIN.EPOCHS = cfg.YOLO.TRAIN.EPOCHS_FOR_FINE_TUNING
# cfg.YOLO.TRAIN.DATA_AUG = True
# cfg.YOLO.TRAIN.HORIZONTAL_FLIP = True
# cfg.YOLO.TRAIN.VERTICAL_FLIP = False
# cfg.YOLO.TRAIN.RANDOM_CROP = True
# cfg.YOLO.TRAIN.RANDOM_SCALE = True
# cfg.YOLO.TRAIN.RANDOM_TRANSLATE = False # bug
# cfg.YOLO.TRAIN.RANDOM_ROTATE = False # bug
# cfg.YOLO.TRAIN.USE_COLORJITTER = True
# cfg.YOLO.TRAIN.TRANSFER = 'resume'
# cfg.YOLO.TRAIN.USE_VALID_DATASET = False

# # train_dataset = YOLODatasetForFineTuningICDAR() # icdar 13, 15, 17, total text

# # classes_map = train_dataset.classes

# # trainer = YOLOTrainer(imgsz=cfg.YOLO.TRAIN.MODEL_SIZE, epochs=cfg.YOLO.TRAIN.EPOCHS, topk=cfg.YOLO.TOPK, batch_size=cfg.YOLO.BATCH_SIZE,
# #                       saved_model_dir=cfg.YOLO.TRAIN.SAVED_MODEL_DIR, model_type=cfg.YOLO.MODEL_TYPE, transfer=cfg.YOLO.TRAIN.TRANSFER,
# #                       optimizer_type=cfg.YOLO.TRAIN.OPTIMIZER_TYPE, lr_init=cfg.YOLO.TRAIN.LR_INIT, class_file=cfg.YOLO.CLASSES,
# #                       iou_threshold=cfg.YOLO.TRAIN.IOU_THRESHOLD, conf_threshold=cfg.YOLO.TRAIN.CONF_THRESHOLD,
# #                       patience=cfg.YOLO.TRAIN.PATIENCE, iou_type=cfg.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES,
# #                       angle_min=cfg.YOLO.ANGLE_MIN, angle_max=cfg.YOLO.ANGLE_MAX,
# #                       use_valid_dataset=cfg.YOLO.TRAIN.USE_VALID_DATASET , image_dir_for_testing=cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING,
# #                       visual_learning_process=cfg.YOLO.TRAIN.VISUAL_LEARNING_PROCESS)

# # model = trainer.train(train_dataset)

"""# **test predict**"""

def image_preprocess(image, target_size, keep_ratio=True, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    if keep_ratio:
        image_padded = np.full(shape=[ih, iw, 3], fill_value=0.0)
        padding_position = np.random.randint(low=0, high=4)
        dw, dh = (iw - nw), (ih - nh)
        if padding_position == 0: # padding top & right
            image_padded[dh:nh+dh, :nw, :] = image_resized
            if gt_boxes is None:
                return image_padded, padding_position
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_padded, gt_boxes
        elif padding_position == 1: # padding top & left
            image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
            if gt_boxes is None:
                return image_padded, padding_position
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_padded, gt_boxes
        elif padding_position == 2: # padding bottom & right
            image_padded[:nh, :nw, :] = image_resized
            if gt_boxes is None:
                return image_padded, padding_position
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
            return image_padded, gt_boxes
        else: # padding bottom & left
            image_padded[:nh, dw:nw+dw, :] = image_resized
            if gt_boxes is None:
                return image_padded, padding_position
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
            return image_padded, gt_boxes
    else:
        if gt_boxes is None:
            return image_resized
        else:
            return image_resized, gt_boxes


def rotate_rectangle(image, xc, yc, w, h, angle_rad, color=(255, 0, 0), thickness=1):
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    x1 = (xc - w/2 * cos_a - h/2 * sin_a)
    y1 = (yc - w/2 * sin_a + h/2 * cos_a)
    x2 = (xc + w/2 * cos_a - h/2 * sin_a)
    y2 = (yc + w/2 * sin_a + h/2 * cos_a)
    x3 = (xc + w/2 * cos_a + h/2 * sin_a)
    y3 = (yc + w/2 * sin_a - h/2 * cos_a)
    x4 = (xc - w/2 * cos_a + h/2 * sin_a)
    y4 = (yc - w/2 * sin_a - h/2 * cos_a)
    try:
        for x1, y1, x2, y2, x3, y3, x4, y4 in zip(x1, y1, x2, y2, x3, y3, x4, y4):
            image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
            image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
            image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
            image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
    except:
        image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
        image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness=thickness)
        image = cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness=thickness)
        image = cv2.line(image, (int(x4), int(y4)), (int(x1), int(y1)), color, thickness=thickness)
    return image

# new code
def draw_rotated_box(image, word_boxes=None, char_boxes=None, group_chars=None, num_word=0, gt_boxes=None, gt_label=None):
    colors = [(255, 0, 0), (255, 0, 0)] # char_color, word_color

    image_h, image_w, _ = image.shape

    bbox_thick = max(int(1.0 * (image_h + image_w) / 600), 1)

    char_color = colors[0]
    word_color = colors[1]

    classes = gt_label

    if gt_boxes is not None:
        word_boxes = gt_boxes

    xc, yc, w, h, angle, bbox_color = convert_format(word_boxes, char_color, word_color, for_chars=False)
    image = rotate_rectangle(image, xc, yc, w, h, angle, bbox_color, bbox_thick)
    return image

def convert_format( box, char_color, word_color=None, for_chars=True):
    xc = (box[:, 2] - box[:, 0]) / 2.0 + box[:, 0]
    yc = (box[:, 3] - box[:, 1]) / 2.0 + box[:, 1]
    w = (box[:, 2] - box[:, 0])
    h = (box[:, 3] - box[:, 1])
    angle = box[:, 4]
    bbox_color = char_color if for_chars else word_color
    return xc, yc, w, h, angle, bbox_color

def load_total_text(annot_path):
    gts = [[]]
    annotation = []
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        pattern = '\d+'
        for line in txt:
            dic = {}
            annot = line.split(',')
            a = re.findall(pattern, annot[0] + annot[1])
            n = len(a)
            dic['points'] = [(int(a[i]), int(a[i + int(n/2)])) for i in range(int(n/2))]
            dic['ignore'] = False if annot[2].split(' ')[-1] != '[u\'#\']' else True
            gts[0].append(dic)
    return gts

def load_icdar2013_v2(annot_path):
    gts = [[]]
    annotation = []
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        for line in txt:
            annot = line.split(',')
            dic = {}
            dic['points'] = [(int(annot[0]), int(annot[1])), (int(annot[0]), int(annot[3])), (int(annot[2]), int(annot[3])), (int(annot[2]), int(annot[1]))]
            dic['ignore'] = False
            gts[0].append(dic)
    return gts

def load_icdar2015_v2(annot_path):
    gts = [[]]
    annotation = []
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        for line in txt:
            annot = line.split(',')
            dic = {}
            dic['points'] = [(int(annot[0]), int(annot[1])), (int(annot[2]), int(annot[3])), (int(annot[4]), int(annot[5])), (int(annot[6]), int(annot[7]))]
            dic['ignore'] = False if annot[8].strip() != '###' else True
            gts[0].append(dic)
    return gts



def load_total_text_v2(annot_path):
    gts = [[]]
    annotation = []
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        pattern = '\d+'
        for line in txt:
            dic = {}
            annot = line.split(',')
            a = re.findall(pattern, annot[0] + annot[1])
            poly = np.array(a).reshape(2, -1).T.astype('int')
            (xc, yc), (w, h), angle = cv2.minAreaRect(poly)
            angle = angle / 180 * np.pi
            angle = -angle
            if w < h:
                w, h = h, w
                angle += np.pi/2
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            x1 = (xc - w/2 * cos_a - h/2 * sin_a)
            y1 = (yc - w/2 * sin_a + h/2 * cos_a)
            x2 = (xc + w/2 * cos_a - h/2 * sin_a)
            y2 = (yc + w/2 * sin_a + h/2 * cos_a)
            x3 = (xc + w/2 * cos_a + h/2 * sin_a)
            y3 = (yc + w/2 * sin_a - h/2 * cos_a)
            x4 = (xc - w/2 * cos_a + h/2 * sin_a)
            y4 = (yc - w/2 * sin_a - h/2 * cos_a)
            dic['points'] = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
            dic['ignore'] = False if annot[2].split(' ')[-1] != '[u\'#\']' else True
            gts[0].append(dic)
    return gts

def convert_word_boxes(word_boxes):
    preds = [[]]
    xc = (word_boxes[:, 2] - word_boxes[:, 0]) / 2.0 + word_boxes[:, 0]
    yc = (word_boxes[:, 3] - word_boxes[:, 1]) / 2.0 + word_boxes[:, 1]
    w = (word_boxes[:, 2] - word_boxes[:, 0])
    h = (word_boxes[:, 3] - word_boxes[:, 1])
    angle = word_boxes[:, 4]

    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    x1 = (xc - w/2 * cos_a - h/2 * sin_a)
    y1 = (yc - w/2 * sin_a + h/2 * cos_a)
    x2 = (xc + w/2 * cos_a - h/2 * sin_a)
    y2 = (yc + w/2 * sin_a + h/2 * cos_a)
    x3 = (xc + w/2 * cos_a + h/2 * sin_a)
    y3 = (yc + w/2 * sin_a - h/2 * cos_a)
    x4 = (xc - w/2 * cos_a + h/2 * sin_a)
    y4 = (yc - w/2 * sin_a - h/2 * cos_a)
    try:
        for x1, y1, x2, y2, x3, y3, x4, y4 in zip(x1, y1, x2, y2, x3, y3, x4, y4):
            dic = {}
            dic['points'] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            dic['ignore'] = False
            preds[0].append(dic)
    except:
        dic = {}
        dic['points'] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        dic['ignore'] = False
        preds[0].append(dic)
    return preds

def convert_bboxes(gt_boxes, position_padding_idx, original_image_size, target_size):
    ih, iw = target_size
    h, w, _ = original_image_size.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    dw, dh = (iw - nw), (ih - nh)
    if position_padding_idx == 0: # padding top & right
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / scale
        gt_boxes[:, [1, 3]] = (gt_boxes[:, [1, 3]] - dh) / scale
        return gt_boxes
    elif position_padding_idx == 1: # padding top & left
        gt_boxes[:, [0, 2]] = (gt_boxes[:, [0, 2]]- dw)/ scale
        gt_boxes[:, [1, 3]] = (gt_boxes[:, [1, 3]]- dh)/ scale
        return gt_boxes
    elif position_padding_idx == 2: # padding bottom & right
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / scale
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / scale
        return gt_boxes
    else: # padding bottom & left
        gt_boxes[:, [0, 2]] = (gt_boxes[:, [0, 2]] - dw) / scale
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / scale
        return gt_boxes



########## build lại model
# model_type = 'l'
# iou_threshold = 0.25
# input_size = (640, 640)
# num_classes = 1
# angle_min =-180
# angle_max = 180
# conf_threshold = 0.5
# if model_type == 's':
#     saved_model_dir = 'assets/model/pretrained_small_model'
# elif model_type == 'm':
#     saved_model_dir = 'assets/model/pretrained_medium_model'
# else:
#     saved_model_dir = 'assets/model/pretrained_large_model'
# use_valid_dataset = True
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Creating yolo-nas {model_type} model....')
# yaml_path, weight_path = get_yaml_and_weight_path(model_type)
# backbone_data, neck_data, head_data = get_model_info_from_yaml(yaml_path)
# model = RYoloNAS(imgsz=input_size, num_classes=num_classes, iou_threshold=iou_threshold,
#                         backbone_data=backbone_data, neck_data=neck_data, head_data=head_data,
#                         angle_min=angle_min, angle_max=angle_max)

# latest_weight_path = os.path.join(saved_model_dir, 'model.pth')
# checkpoint = torch.load(latest_weight_path, map_location=torch.device('cpu'))
# model.replace_header()
# model.load_state_dict(checkpoint)
# model.eval()
# model.to(device)
# print('Done')
##### done

def build_model(model_type='l', iou_threshold=0.25, input_size=(640, 640), num_classes=1, angle_min=-180, angle_max=180, conf_threshold=0.5, use_valid_dataset=True):
    if model_type == 's':
        saved_model_dir = 'assets/model/pretrained_small_model'
    elif model_type == 'm':
        saved_model_dir = 'assets/model/pretrained_medium_model'
    else:
        saved_model_dir = 'assets/model/pretrained_large_model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Creating yolo-nas {model_type} model....')
    yaml_path, weight_path = get_yaml_and_weight_path(model_type)
    backbone_data, neck_data, head_data = get_model_info_from_yaml(yaml_path)
    model = RYoloNAS(imgsz=input_size, num_classes=num_classes, iou_threshold=iou_threshold,
                            backbone_data=backbone_data, neck_data=neck_data, head_data=head_data,
                            angle_min=angle_min, angle_max=angle_max)

    latest_weight_path = os.path.join(saved_model_dir, 'model.pth')
    checkpoint = torch.load(latest_weight_path, map_location=torch.device('cpu'))
    model.replace_header()
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print('Done')
    return model

def model_predict(model, image_path, input_size=(640, 640), conf_threshold=0.2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf_threshold = torch.tensor([conf_threshold], dtype=torch.float32).to(device)

    image_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    ori_image_input = image_input
    image_input, pos_padding = image_preprocess(ori_image_input, input_size)
    image = image_input.astype('uint8')
    image_input = image_input / 255.
    image_input = torch.tensor(image_input, dtype=torch.float32)
    image_input = image_input.permute([2, 0, 1]).contiguous().to(device) # c, h, w

    start = time.time()
    word_boxes = model(image_input[None], conf_threshold)
    end = time.time()
    total_time = end-start
    print('total time: ', total_time)

    word_boxes = word_boxes.cpu().detach().numpy() # [num_boxes, 7] [x, y, x, y, angle, conf, cls]

    return word_boxes, image, total_time

model = build_model(model_type='l')
image_path = 'assets/test_images/img_490.jpg'
word_boxes, image, total_time = model_predict(model, image_path)

image_predict = draw_rotated_box(np.array(image), word_boxes=word_boxes, gt_boxes=None, gt_label=None)
plt.imshow(image_predict)
plt.show()

# conf_threshold = 0.2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# conf_threshold = torch.tensor([conf_threshold], dtype=torch.float32).to(device)
# input_size = (640, 640)

# image_path = 'assets/test_images/img_490.jpg'

# image_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
# image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
# ori_image_input = image_input
# image_input, pos_padding = image_preprocess(ori_image_input, input_size)
# image = image_input.astype('uint8')
# image_input = image_input / 255.
# image_input = torch.tensor(image_input, dtype=torch.float32)
# image_input = image_input.permute([2, 0, 1]).contiguous().to(device) # c, h, w
# start = time.time()
# word_boxes = model(image_input[None], conf_threshold)
# end = time.time()
# total_time = end-start
# print('total time: ', total_time)
# word_boxes = word_boxes.cpu().detach().numpy() # [num_boxes, 7] [x, y, x, y, angle, conf, cls]


# for image_path in test_image_paths:
#     if use_valid_dataset:
#         # image_path = random.choice(test_image_paths)
#         # image_path = 'new_images/1.jpg'
#         condition = True
#         while condition:
#             try:
#                 # url = 'https://th.bing.com/th/id/OIP._SkcR7MOh5VoCCECWRwduAHaEL?pid=ImgDet&rs=1'
#                 # image_input = io.imread(url) # specific image
#                 image_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                 image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
#                 # image_input = cv2.resize(image_input, dsize=(input_size[1], input_size[0]))
#                 ori_image_input = image_input
#                 image_input, pos_padding = image_preprocess(ori_image_input, input_size)
#                 condition = False
#             except:
#                 condition = False
#                 image_path = random.choice(test_image_paths)
#         image = image_input.astype('uint8')
#         image_input = image_input / 255.
#         image_input = torch.tensor(image_input, dtype=torch.float32)
#         image_input = image_input.permute([2, 0, 1]).contiguous().to(device) # c, h, w
#         start = time.time()
#         word_boxes = model(image_input[None], conf_threshold)
#         end = time.time()
#         total_time = end-start
#         print('total time: ', total_time)
#     else:
#         for step, (images, gt_labels, gt_bboxes, gt_mask) in enumerate(stream):
#             images, gt_labels, gt_bboxes, gt_mask = images.to(device), gt_labels.to(device), gt_bboxes.to(device), gt_mask.to(device)
#             image_test = images
#             gt_boxes = gt_bboxes
#             gt_labels = gt_labels
#             break

#         image_input = image_test[0]
#         image = image_input * 255.
#         image = image.cpu().numpy()
#         image = np.transpose(image, [1, 2, 0])
#         image = np.ascontiguousarray(image, dtype=np.uint8)
#         word_boxes = model(image_input[None], conf_threshold)
#     word_boxes = word_boxes.cpu().detach().numpy() # [num_boxes, 7] [x, y, x, y, angle, conf, cls]
#     # new code
#     word_boxes = convert_bboxes(word_boxes, pos_padding, ori_image_input, input_size)

#     test_gt_path = test_gt_dir + '/gt_'+ image_path.split('/')[-1].split('.')[0] + '.txt'
#     gts = load_icdar2015_v2(test_gt_path)
#     preds = convert_word_boxes(word_boxes)
#     # print(f'gts: {gts}\npreds: {preds}')
#     evaluator = DetectionIoUEvaluator()
#     results = []
#     for gt, pred in zip(gts, preds):
#         results.append(evaluator.evaluate_image(gt, pred))
#     metrics = evaluator.combine_results(results)
#     # print(metrics)
#     final_metrics['recall'] += metrics['recall']
#     final_metrics['precision'] += metrics['precision']
#     final_metrics['hmean'] += metrics['hmean']
#     avg_total_time += total_time
#     counter += 1
#     for gt in gts[0]:
#         image = cv2.line(np.array(image), (int(gt['points'][0][0]), int(gt['points'][0][1])), (int(gt['points'][1][0]), int(gt['points'][1][1])), color=(0, 255, 0))
#         image = cv2.line(np.array(image), (int(gt['points'][1][0]), int(gt['points'][1][1])), (int(gt['points'][2][0]), int(gt['points'][2][1])), color=(0, 255, 0))
#         image = cv2.line(np.array(image), (int(gt['points'][2][0]), int(gt['points'][2][1])), (int(gt['points'][3][0]), int(gt['points'][3][1])), color=(0, 255, 0))
#         image = cv2.line(np.array(image), (int(gt['points'][3][0]), int(gt['points'][3][1])), (int(gt['points'][0][0]), int(gt['points'][0][1])), color=(0, 255, 0))


#     #
#     # img = cv2.line

#     img = image
#     if not use_valid_dataset:
#         image_gt = draw_rotated_box(np.array(img), None, None, None, num_word=0, gt_boxes=gt_boxes[0].cpu().numpy(),
#                                         gt_label=gt_labels[0].cpu().numpy())

#     image_predict = draw_rotated_box(np.array(img), word_boxes=word_boxes, gt_boxes=None, gt_label=None)

# plt.imshow(image_predict)