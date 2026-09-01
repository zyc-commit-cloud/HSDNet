# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import torch.nn as nn
import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, GroupConv
from .transformer import TransformerBlock
import math
__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'MANet', 'C2f', 'C3x',
           'C3TR', 'C3Ghost','GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'HyperComputeModule','MANet_SimAM','MAN_sea','SEAttention',' C2f_Ghost','MANet_pconv','Bottleneck_PConv','Partial_conv3','Conv_GN','Scale','DySample','MAN_eca','EfficientChannelAttention','SimAM','MANet_Ghost','SCDown','SPDConv','MANet_Faster','Faster_Block','DropPath','MPCA','DPFE','HaarWaveletConv','WaveletConv','WaveletConvNeck','WCMANet','GroupBatchnorm2d','SRU','CRU','ScConv','ScConvNeck','SCCMANet','LSKBlock_SA','LSKBlock','LSKConvNeck','LSKCMANet','RepConvN','RepConvNeck','RCMANet','DPDown','DBBDPFE','PPADPFE','FCMANet','DRCMANet','DBBMANet','PPAMANet','PPALocalCAMANet','LocalCAMANet','PPAECAMANet','PPASimAMMANet','PPAEMAMANet','PPAELAMANet')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))

#----------------------------------------------
class LocalCAMANet(MANet):
    """
    Original MANet with one branch-only Coordinate Attention.

    - Original Bottleneck path is unchanged.
    - CoordAtt is applied only after cv_block_2.
    - No global CoordAtt after concatenation.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__(
            c1,
            c2,
            n,
            shortcut,
            p,
            kernel_size,
            g,
            e,
        )

        # 仅在局部分支使用一次CoordAtt
        self.ca = CoordAtt(
            in_channels=self.c,
            reduction=32,
        )

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)

        # 原MANet的卷积分支，只增加一次CoordAtt
        branch_1 = self.cv_block_2(features)
        branch_1 = self.ca(branch_1)

        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [
            branch_0,
            branch_1,
            branch_2,
            branch_3,
        ]

        # 保持原MANet的Bottleneck重复路径
        for module in self.m:
            outputs.append(module(outputs[-1]))

        fused = torch.cat(outputs, dim=1)

        # 不使用ca_global
        return self.cv_final(fused)
class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()


    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
# YOLO原生依赖导入
from ultralytics.nn.modules.conv import Conv, GroupConv
from ultralytics.nn.modules.block import Bottleneck


# -------------------------- 零尺寸偏差的标准 CoordAtt 实现 --------------------------
class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 全局平均池化，绝对不改变特征图高宽
        x_h = torch.mean(x, dim=3, keepdim=True)  # [n,c,h,1]
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # [n,c,w,1]

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 输出和输入尺寸100%一致
        return identity * a_w * a_h

#--------------------------------------------------------------------------------------
import torch.nn.functional as F

class HaarWaveletConv(nn.Module):
    def __init__(self, in_channels, grad=False):
        super().__init__()
        self.in_channels = in_channels

        haar_weights = torch.ones(4, 1, 2, 2)

        # horizontal / vertical / diagonal high-frequency filters
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1

        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1

        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        haar_weights = torch.cat([haar_weights] * in_channels, dim=0)

        if grad:
            self.haar_weights = nn.Parameter(haar_weights)
        else:
            self.register_buffer("haar_weights", haar_weights)

    def forward(self, x):
        b, _, h, w = x.size()
        x = F.pad(x, [0, 1, 0, 1], value=0)
        out = F.conv2d(
            x,
            self.haar_weights,
            bias=None,
            stride=1,
            groups=self.in_channels
        ) / 4.0

        out = out.reshape(b, self.in_channels, 4, h, w)
        out = torch.transpose(out, 1, 2)
        out = out.reshape(b, self.in_channels * 4, h, w)

        low, high_h, high_v, high_d = out.chunk(4, dim=1)
        high = high_h + high_v + high_d
        return low, high

class WaveletConv(nn.Module):
    def __init__(self, c1, c2, grad=False):
        super().__init__()
        self.wavelet = HaarWaveletConv(c1, grad=grad)
        self.fuse = Conv(2 * c1, c2, 1, 1)

    def forward(self, x):
        low, high = self.wavelet(x)
        return self.fuse(torch.cat((low, high), dim=1))
class WaveletConvNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, grad=False):
        super().__init__()
        c_ = int(c2 * e)
        self.wv1 = WaveletConv(c1, c_, grad=grad)
        self.wv2 = WaveletConv(c_, c2, grad=grad)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.wv2(self.wv1(x))
        return x + y if self.add else y
# -------------------------- DPFE 主实现（彻底解决所有报错） --------------------------
class DPFE(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(in_channels=(4 + n) * self.c, reduction=32)

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_= self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)
#-------------------------------------------------------------------------------
class WCMANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(WaveletConvNeck(self.c, self.c, shortcut, g,  e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(in_channels=(4 + n) * self.c, reduction=32)

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_= self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)
# -------------------------- SimAM（彻底解决所有报错） --------------------------
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# -------------------------- MANet_sim（彻底解决所有报错） --------------------------
class MANet_SimAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = SimAM()
        self.ca_global = SimAM()

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_ = self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)

# -------------------------- sea（彻底解决所有报错） --------------------------
class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# -------------------------- sea（彻底解决所有报错） --------------------------
class MAN_sea(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = SEAttention(channel=self.c, reduction=16)           # ← 改这里
        self.ca_global = SEAttention(channel=(4 + n) * self.c, reduction=16)  # ← 改这里

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_= self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)


# -------------------------- ela（彻底解决所有报错） --------------------------
class ELA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).squeeze(-1)).unsqueeze(-1)
        x_w = self.conv1x1(self.pool_w(x).squeeze(-2)).unsqueeze(-2)
        return x * x_h * x_w

class MAN_ela(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = ELA(channels=self.c)
        self.ca_global = ELA(channels=(4 + n) * self.c)

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_ = self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)
        return self.cv_final(y_out)

class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return out * x

class MAN_eca(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = EfficientChannelAttention(c=self.c)
        self.ca_global = EfficientChannelAttention(c=(4 + n) * self.c)

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_ = self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)
        return self.cv_final(y_out)
# -------------------------- ela（彻底解决所有报错） --------------------------
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class MAN_ema(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
        self.ca = EMA(channels=self.c)           
        self.ca_global = EMA(channels=(4+n)*self.c)  

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1_= self.ca(y1)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1_, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)

# -------------------------- （彻底解决所有报错） --------------------------
class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)
# -------------------------- （彻底解决所有报错） --------------------------
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# -------------------------- （彻底解决所有报错） --------------------------
class C2f_Ghost(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# -------------------------- （彻底解决所有报错） -------------------------
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class Bottleneck_PConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Partial_conv3(c1)
        self.cv2 = Partial_conv3(c2)

class MANet_pconv(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_PConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))

#-----head---------------------------
class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

#_____------------------------------------------upsample
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

# -------------------------- （彻底解决所有报错） -------------------------
class MANet_Ghost(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c) for _ in range(n)  # ← Ghost
        )
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))

# -------------------------- （彻底解决所有报错） -------------------------
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

# -------------------------- （彻底解决所有报错） -------------------------
class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.
    From YOLOv10 (Tsinghua University, 2024)
    Spatial-Channel Decoupled Downsampling
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
# -------------------------- （彻底解决所有报错） -------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor






class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class MANet_Faster(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Faster_Block(self.c, self.c) for _ in range(n)
        )
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))
# -------------------------- （彻底解决所有报错） -------------------------
class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels, 1)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)
    
    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()

#-----------------------------------------------------------------------------
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        super().__init__()
        self.group_num = group_num
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(n, c, h, w)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self, op_channel, group_num=16, gate_treshold=0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(op_channel, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / (torch.sum(self.gn.gamma) + 1e-6)
        reweights = self.sigmoid(gn_x * w_gamma)

        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold

        x_1 = info_mask * x
        x_2 = noninfo_mask * x

        return self.reconstruct(x_1, x_2)

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.chunk(x_1, 2, dim=1)
        x_21, x_22 = torch.chunk(x_2, 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(
        self,
        op_channel,
        alpha=0.5,
        squeeze_radio=2,
        group_size=2,
        group_kernel_size=3,
    ):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel

        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_radio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(
            self.up_channel // squeeze_radio,
            op_channel,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
        )
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(
            self.low_channel // squeeze_radio,
            op_channel - self.low_channel // squeeze_radio,
            kernel_size=1,
            bias=False,
        )

        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)

        up = self.squeeze1(up)
        low = self.squeeze2(low)

        y1 = self.GWC(up) + self.PWC1(up)
        y2 = torch.cat([self.PWC2(low), low], dim=1)

        out = torch.cat([y1, y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out

        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(
        self,
        op_channel,
        group_num=16,
        gate_treshold=0.5,
        alpha=0.5,
        squeeze_radio=2,
        group_size=2,
        group_kernel_size=3,
    ):
        super().__init__()
        self.SRU = SRU(
            op_channel,
            group_num=group_num,
            gate_treshold=gate_treshold,
        )
        self.CRU = CRU(
            op_channel,
            alpha=alpha,
            squeeze_radio=squeeze_radio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class ScConvNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.scconv = ScConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.scconv(self.cv1(x)))
        return x + y if self.add else y


class SCCMANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            ScConvNeck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1),
        )

        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(in_channels=(4 + n) * self.c, reduction=32)

    def forward(self, x):
        y = self.cv_first(x)

        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y1 = self.ca(y1)

        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)

#--------------------------------------------------------------------------------------
class LSKBlock_SA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)

        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)

        return x * attn


class LSKBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKBlock_SA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class LSKConvNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.lsk = LSKBlock(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.lsk(self.cv1(x)))
        return x + y if self.add else y

class LSKCMANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            LSKConvNeck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1),
        )

        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(in_channels=(4 + n) * self.c, reduction=32)

    def forward(self, x):
        y = self.cv_first(x)

        y0 = self.cv_block_1(y)

        y1 = self.cv_block_2(y)
        y1 = self.ca(y1)

        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)

#---------------------------------------------
class RepConvN(nn.Module):
    """RepConv block with 3x3 and 1x1 branches."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.forward_fuse(x)
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)

        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps

        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "conv"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)

        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        for para in self.parameters():
            para.detach_()

        self.__delattr__("conv1")
        self.__delattr__("conv2")

        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class RepConvNeck(nn.Module):
    """ConvNeck with two RepConvN layers."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k=3, s=1, p=1, g=1)
        self.cv2 = RepConvN(c_, c2, k=3, s=1, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RCMANet(nn.Module):
    """Re-parameterized Coordinate Mixed Aggregation Network."""

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            RepConvNeck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1),
        )

        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(in_channels=(4 + n) * self.c, reduction=32)

    def forward(self, x):
        y = self.cv_first(x)

        y0 = self.cv_block_1(y)

        y1 = self.cv_block_2(y)
        y1 = self.ca(y1)

        y2, y3 = y.chunk(2, 1)

        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        concat_out = torch.cat(y, dim=1)
        y_out = self.ca_global(concat_out)

        return self.cv_final(y_out)

#-------------------------------------------------------
class DPDown(nn.Module):
    """Dual-path detail-preserving downsampling."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()

        c_semantic = int(c2 * e)
        c_detail = c2 - c_semantic

        if c_semantic < 1 or c_detail < 1:
            raise ValueError(f"Invalid DPDown channels: c1={c1}, c2={c2}, e={e}")

        # Learned semantic downsampling branch.
        self.semantic = Conv(
            c1,
            c_semantic,
            k=3,
            s=2,
            act=False
        )

        # Rearrange every 2x2 neighborhood into channels before compression.
        self.detail = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            Conv(
                4 * c1,
                c_detail,
                k=1,
                s=1,
                act=False
            )
        )

        # Fuse semantic and detail features.
        self.fuse = Conv(
            c2,
            c2,
            k=1,
            s=1
        )

    def forward(self, x):
        semantic = self.semantic(x)
        detail = self.detail(x)
        return self.fuse(torch.cat((semantic, detail), dim=1))

#-------------------------------------------------
def dbb_conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


def dbb_fuse_bn(kernel, bn):
    std = (bn.running_var + bn.eps).sqrt()
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    return kernel * scale, bn.bias - bn.running_mean * bn.weight / std


def dbb_add_branches(kernels, biases):
    return sum(kernels), sum(biases)


def dbb_fuse_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        kernel = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        bias = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        kernel_slices = []
        bias_slices = []
        k1_t = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups

        for group_idx in range(groups):
            k1_slice = k1_t[
                :,
                group_idx * k1_group_width : (group_idx + 1) * k1_group_width,
                :,
                :,
            ]
            k2_slice = k2[
                group_idx * k2_group_width : (group_idx + 1) * k2_group_width,
                :,
                :,
                :,
            ]
            b1_slice = b1[
                group_idx * k1_group_width : (group_idx + 1) * k1_group_width
            ]
            kernel_slices.append(F.conv2d(k2_slice, k1_slice))
            bias_slices.append(
                (k2_slice * b1_slice.reshape(1, -1, 1, 1)).sum((1, 2, 3))
            )

        kernel = torch.cat(kernel_slices, dim=0)
        bias = torch.cat(bias_slices, dim=0)

    return kernel, bias + b2


def dbb_avg_kernel(channels, kernel_size, groups):
    input_dim = channels // groups
    kernel = torch.zeros(
        (channels, input_dim, kernel_size, kernel_size), dtype=torch.float32
    )
    indices = np.arange(channels)
    kernel[indices, np.tile(np.arange(input_dim), groups), :, :] = (
        1.0 / kernel_size**2
    )
    return kernel


def dbb_pad_kernel(kernel, target_kernel_size):
    current_size = kernel.size(2)
    if current_size == target_kernel_size:
        return kernel
    total_padding = target_kernel_size - current_size
    left = total_padding // 2
    right = total_padding - left
    return F.pad(kernel, [left, right, left, right])


class DBBIdentityConv1x1(nn.Conv2d):
    def __init__(self, channels, groups=1):
        super().__init__(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        input_dim = channels // groups
        identity = torch.zeros(channels, input_dim, 1, 1)
        for channel_idx in range(channels):
            identity[channel_idx, channel_idx % input_dim, 0, 0] = 1.0
        self.register_buffer("identity_kernel", identity)
        nn.init.zeros_(self.weight)

    def get_actual_kernel(self):
        return self.weight + self.identity_kernel.to(
            device=self.weight.device, dtype=self.weight.dtype
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.get_actual_kernel(),
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.groups,
        )


class DBBBNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.pad_pixels = pad_pixels

    def forward(self, x):
        output = self.bn(x)
        if self.pad_pixels <= 0:
            return output

        if self.bn.affine:
            pad_values = self.bn.bias.detach() - (
                self.bn.running_mean
                * self.bn.weight.detach()
                / torch.sqrt(self.bn.running_var + self.bn.eps)
            )
        else:
            pad_values = -self.bn.running_mean / torch.sqrt(
                self.bn.running_var + self.bn.eps
            )

        output = F.pad(output, [self.pad_pixels] * 4)
        pad_values = pad_values.view(1, -1, 1, 1)
        output[:, :, : self.pad_pixels, :] = pad_values
        output[:, :, -self.pad_pixels :, :] = pad_values
        output[:, :, :, : self.pad_pixels] = pad_values
        output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):
    """Diverse Branch Block with equivalent single-convolution deployment."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        internal_channels_1x1_3x3=None,
        deploy=False,
        single_init=False,
    ):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size % 2 == 0:
            raise ValueError("DBB requires an odd integer kernel_size")
        if dilation != 1:
            raise ValueError("This DBB integration supports dilation=1 only")

        padding = kernel_size // 2 if padding is None else padding
        if padding != kernel_size // 2:
            raise ValueError("DBB padding must equal kernel_size // 2")

        self.deploy = deploy
        self.nonlinear = copy.deepcopy(Conv.default_act)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if deploy:
            self.dbb_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.dbb_origin = dbb_conv_bn(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            )

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                )
                self.dbb_avg.add_module(
                    "bn", DBBBNAndPadLayer(padding, out_channels)
                )
                self.dbb_avg.add_module(
                    "avg",
                    nn.AvgPool2d(kernel_size, stride=stride, padding=0),
                )
                self.dbb_1x1 = dbb_conv_bn(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                )
            else:
                self.dbb_avg.add_module(
                    "avg",
                    nn.AvgPool2d(kernel_size, stride=stride, padding=padding),
                )
            self.dbb_avg.add_module("avgbn", nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = (
                    in_channels if groups < out_channels else 2 * in_channels
                )

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module(
                    "idconv1", DBBIdentityConv1x1(in_channels, groups)
                )
            else:
                self.dbb_1x1_kxk.add_module(
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        internal_channels_1x1_3x3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                )
            self.dbb_1x1_kxk.add_module(
                "bn1",
                DBBBNAndPadLayer(padding, internal_channels_1x1_3x3),
            )
            self.dbb_1x1_kxk.add_module(
                "conv2",
                nn.Conv2d(
                    internal_channels_1x1_3x3,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
            )
            self.dbb_1x1_kxk.add_module("bn2", nn.BatchNorm2d(out_channels))

        if single_init:
            self.single_init()

    def forward(self, x):
        if hasattr(self, "dbb_reparam"):
            return self.nonlinear(self.dbb_reparam(x))

        output = self.dbb_origin(x)
        if hasattr(self, "dbb_1x1"):
            output = output + self.dbb_1x1(x)
        output = output + self.dbb_avg(x)
        output = output + self.dbb_1x1_kxk(x)
        return self.nonlinear(output)

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = dbb_fuse_bn(
            self.dbb_origin[0].weight, self.dbb_origin[1]
        )

        if hasattr(self, "dbb_1x1"):
            k_1x1, b_1x1 = dbb_fuse_bn(
                self.dbb_1x1[0].weight, self.dbb_1x1[1]
            )
            k_1x1 = dbb_pad_kernel(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, "idconv1"):
            k_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_first = self.dbb_1x1_kxk.conv1.weight
        k_first, b_first = dbb_fuse_bn(k_first, self.dbb_1x1_kxk.bn1)
        k_second, b_second = dbb_fuse_bn(
            self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2
        )
        k_1x1_kxk, b_1x1_kxk = dbb_fuse_1x1_kxk(
            k_first, b_first, k_second, b_second, self.groups
        )

        k_avg = dbb_avg_kernel(
            self.out_channels, self.kernel_size, self.groups
        ).to(
            device=self.dbb_avg.avgbn.weight.device,
            dtype=self.dbb_avg.avgbn.weight.dtype,
        )
        k_avg_second, b_avg_second = dbb_fuse_bn(k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, "conv"):
            k_avg_first, b_avg_first = dbb_fuse_bn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn
            )
            k_avg_merged, b_avg_merged = dbb_fuse_1x1_kxk(
                k_avg_first,
                b_avg_first,
                k_avg_second,
                b_avg_second,
                self.groups,
            )
        else:
            k_avg_merged, b_avg_merged = k_avg_second, b_avg_second

        return dbb_add_branches(
            (k_origin, k_1x1, k_1x1_kxk, k_avg_merged),
            (b_origin, b_1x1, b_1x1_kxk, b_avg_merged),
        )

    def switch_to_deploy(self):
        if hasattr(self, "dbb_reparam"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        source_conv = self.dbb_origin[0]
        self.dbb_reparam = nn.Conv2d(
            source_conv.in_channels,
            source_conv.out_channels,
            source_conv.kernel_size,
            stride=source_conv.stride,
            padding=source_conv.padding,
            dilation=source_conv.dilation,
            groups=source_conv.groups,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            self.dbb_reparam.weight.copy_(kernel)
            self.dbb_reparam.bias.copy_(bias)

        for parameter in self.parameters():
            parameter.detach_()
        del self.dbb_origin
        del self.dbb_avg
        if hasattr(self, "dbb_1x1"):
            del self.dbb_1x1
        del self.dbb_1x1_kxk
        self.deploy = True

    def init_gamma(self, value):
        if hasattr(self, "dbb_origin"):
            nn.init.constant_(self.dbb_origin[1].weight, value)
        if hasattr(self, "dbb_1x1"):
            nn.init.constant_(self.dbb_1x1[1].weight, value)
        if hasattr(self, "dbb_avg"):
            nn.init.constant_(self.dbb_avg.avgbn.weight, value)
        if hasattr(self, "dbb_1x1_kxk"):
            nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            nn.init.constant_(self.dbb_origin[1].weight, 1.0)


def _dbb_kernel_size(kernel):
    if isinstance(kernel, (tuple, list)):
        if len(kernel) != 2 or kernel[0] != kernel[1]:
            raise ValueError(f"DBB requires a square kernel, received {kernel}")
        return int(kernel[0])
    return int(kernel)


class Bottleneck_DBB(nn.Module):
    """Standard bottleneck with DBB replacing both spatial convolutions."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        hidden_channels = int(c2 * e)
        k1 = _dbb_kernel_size(k[0])
        k2 = _dbb_kernel_size(k[1])
        self.cv1 = DiverseBranchBlock(c1, hidden_channels, k1, stride=1)
        self.cv2 = DiverseBranchBlock(
            hidden_channels, c2, k2, stride=1, groups=g
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        output = self.cv2(self.cv1(x))
        return x + output if self.add else output


class DBBDPFE(nn.Module):
    """DPFE with DBB-based ConvNeck units."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_DBB(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=(3, 3),
                e=1.0,
            )
            for _ in range(n)
        )
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        hidden_channels = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_channels, 1, 1),
            GroupConv(hidden_channels, hidden_channels, kernel_size, 1),
            Conv(hidden_channels, self.c, 1, 1),
        )
        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(
            in_channels=(4 + n) * self.c, reduction=32
        )

    def forward(self, x):
        features = self.cv_first(x)
        branch_0 = self.cv_block_1(features)
        branch_1 = self.ca(self.cv_block_2(features))
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [branch_0, branch_1, branch_2, branch_3]
        outputs.extend(module(outputs[-1]) for module in self.m)
        fused = self.ca_global(torch.cat(outputs, dim=1))
        return self.cv_final(fused)


def switch_dbb_to_deploy(model):
    """Convert every DBB in a model to its equivalent single convolution."""
    for module in list(model.modules()):
        if isinstance(module, DiverseBranchBlock):
            module.switch_to_deploy()
    return model
#-----------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPAEfficientChannelAttention(nn.Module):
    """Efficient channel attention used by PPA."""

    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.avg_pool(x)
        weights = self.conv(weights.squeeze(-1).transpose(-1, -2))
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class PPASpatialAttention(nn.Module):
    """Channel-pooled spatial attention used by PPA."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        weights = self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))
        return x * weights


class PPALocalGlobalAttention(nn.Module):
    """Patch-aware local-global attention with robust spatial padding."""

    def __init__(self, output_dim, patch_size):
        super().__init__()
        hidden_dim = max(output_dim // 2, 1)
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = nn.Parameter(torch.randn(output_dim))
        self.top_down_transform = nn.Parameter(torch.eye(output_dim))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        if channels != self.output_dim:
            raise ValueError(
                f"PPA attention expected {self.output_dim} channels, received {channels}"
            )

        patch_size = self.patch_size
        pad_h = (patch_size - height % patch_size) % patch_size
        pad_w = (patch_size - width % patch_size) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        padded_h, padded_w = x.shape[-2:]
        x = x.permute(0, 2, 3, 1)

        # unfold on BHWC appends patch dimensions after C, so move C last first.
        patches = x.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        patches = patches.permute(0, 1, 2, 4, 5, 3).contiguous()
        patches = patches.reshape(
            batch_size, -1, patch_size * patch_size, channels
        )
        patches = patches.mean(dim=-1)

        local_features = self.mlp2(self.norm(self.mlp1(patches)))
        local_attention = F.softmax(local_features, dim=-1)
        local_features = local_features * local_attention

        prompt = F.normalize(self.prompt, dim=0).view(1, -1, 1)
        similarity = F.normalize(local_features, dim=-1) @ prompt
        local_features = local_features * similarity.clamp(0, 1)
        local_features = local_features @ self.top_down_transform

        local_features = local_features.reshape(
            batch_size,
            padded_h // patch_size,
            padded_w // patch_size,
            self.output_dim,
        )
        local_features = local_features.permute(0, 3, 1, 2)
        local_features = F.interpolate(
            local_features,
            size=(padded_h, padded_w),
            mode="bilinear",
            align_corners=False,
        )
        local_features = local_features[:, :, :height, :width]
        return self.conv(local_features)


class PPA(nn.Module):
    """Parallelized patch-aware feature enhancement block."""

    def __init__(self, in_features, filters):
        super().__init__()
        if in_features != filters:
            raise ValueError(
                "This PPA integration requires equal input and output channels"
            )

        self.skip = Conv(in_features, filters, act=False)
        self.c1 = Conv(filters, filters, 3)
        self.c2 = Conv(filters, filters, 3)
        self.c3 = Conv(filters, filters, 3)
        self.lga2 = PPALocalGlobalAttention(filters, 2)
        self.lga4 = PPALocalGlobalAttention(filters, 4)
        self.channel_attention = PPAEfficientChannelAttention(filters)
        self.spatial_attention = PPASpatialAttention()
        self.dropout = nn.Dropout2d(0.1)
        self.bn = nn.BatchNorm2d(filters)
        self.act = nn.SiLU()

    def forward(self, x):
        skip = self.skip(x)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        output = x1 + x2 + x3 + skip
        output = output + self.lga2(skip) + self.lga4(skip)
        output = self.channel_attention(output)
        output = self.spatial_attention(output)
        return self.act(self.bn(self.dropout(output)))


class PPADPFE(nn.Module):
    """DPFE whose repeated ConvNeck units are replaced by PPA blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PPA(self.c, self.c) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        hidden_dim = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_dim, 1, 1),
            GroupConv(hidden_dim, hidden_dim, kernel_size, 1),
            Conv(hidden_dim, self.c, 1, 1),
        )
        self.ca = CoordAtt(in_channels=self.c, reduction=32)
        self.ca_global = CoordAtt(
            in_channels=(4 + n) * self.c, reduction=32
        )

    def forward(self, x):
        features = self.cv_first(x)
        branch_0 = self.cv_block_1(features)
        branch_1 = self.ca(self.cv_block_2(features))
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [branch_0, branch_1, branch_2, branch_3]
        for module in self.m:
            outputs.append(module(outputs[-1]))

        fused = torch.cat(outputs, dim=1)
        return self.cv_final(self.ca_global(fused))
#------------------------------------------------------------------
class FCMANet(nn.Module):
    """Fusion-level Coordinate Mixed Aggregation Network.

    Compared with the original DPFE:
        - keeps the original Bottleneck/ConvNeck;
        - removes branch-level CoordAtt;
        - removes pre-compression fusion CoordAtt;
        - applies one CoordAtt after complete fusion and channel compression.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1, 1)

        # Keep the original MANet ConvNeck unchanged.
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
            )
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        hidden_dim = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_dim, 1, 1),
            GroupConv(hidden_dim, hidden_dim, kernel_size, 1),
            Conv(hidden_dim, self.c, 1, 1),
        )

        # One attention module after complete multi-branch fusion.
        self.ca_out = CoordAtt(in_channels=c2, reduction=32)

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)
        branch_1 = self.cv_block_2(features)
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [branch_0, branch_1, branch_2, branch_3]
        for block in self.m:
            outputs.append(block(outputs[-1]))

        fused = torch.cat(outputs, dim=1)
        fused = self.cv_final(fused)
        return self.ca_out(fused)
#---------------------------------------------------------------
"""DRCMANet components for insertion into Ultralytics ``block.py``.

Prerequisites already available in ``block.py``:
    Conv, GroupConv, CoordAtt, RepConvN

The design intentionally uses:
    - one normal 3x3 convolution followed by one RepConvN in each ConvNeck;
    - the original ConvNeck residual connection;
    - no branch-level attention;
    - one CoordAtt only after multi-branch fusion and channel compression.

Use DRCMANet only in backbone Stage2 and Stage3. Keep Stage4 and Stage5 as
the original MANet for the first isolated ablation experiment.
"""

import torch
import torch.nn as nn


class DetailRepConvNeck(nn.Module):
    """Detail-preserving partially re-parameterized ConvNeck.

    The first 3x3 convolution preserves the stable behavior of the original
    ConvNeck. The second convolution is re-parameterized (3x3 + 1x1 during
    training) to strengthen local structure without replacing both original
    convolutions. The outer residual path retains the incoming shallow detail.
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=((3, 3), (3, 3)),
        e=0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)

        # Stable spatial feature extraction from the original ConvNeck.
        self.cv1 = Conv(c1, c_, k[0], 1)

        # Training-time 3x3 + 1x1 branches; deployable as one 3x3 conv.
        self.cv2 = RepConvN(c_, c2, k=3, s=1, p=1, g=g)

        # Preserve the original shallow feature as a detail bypass.
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DRCMANet(nn.Module):
    """Detail-preserving Re-parameterized Coordinate MANet.

    Multi-branch features are extracted without branch-specific attention.
    After concatenation, a point-wise convolution first performs complete
    feature fusion and channel compression. A single CoordAtt then calibrates
    the final fused representation along the horizontal and vertical axes.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1, 1)

        self.m = nn.ModuleList(
            DetailRepConvNeck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
            )
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        hidden_dim = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_dim, 1, 1),
            GroupConv(hidden_dim, hidden_dim, kernel_size, 1),
            Conv(hidden_dim, self.c, 1, 1),
        )

        # Only one attention module, applied to the completely fused output.
        self.ca_out = CoordAtt(in_channels=c2, reduction=32)

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)
        branch_1 = self.cv_block_2(features)
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [branch_0, branch_1, branch_2, branch_3]
        for block in self.m:
            outputs.append(block(outputs[-1]))

        fused = torch.cat(outputs, dim=1)
        fused = self.cv_final(fused)
        return self.ca_out(fused)
#---------------------------------------------------
"""DBB-only MANet ablation for insertion into Ultralytics block.py.

Prerequisites already present in block.py:
    torch, nn, Conv, GroupConv, Bottleneck_DBB

This module contains no attention. Use it only at Stage2 and Stage3, while
Stage4 and Stage5 remain the original MANet, to isolate the DBB ConvNeck
contribution from DPFE/CoordAtt.
"""

import torch
import torch.nn as nn


class DBBMANet(nn.Module):
    """Original MANet aggregation with DBB ConvNeck and no attention."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck_DBB(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=(3, 3),
                e=1.0,
            )
            for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        hidden_channels = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_channels, 1, 1),
            GroupConv(
                hidden_channels,
                hidden_channels,
                kernel_size,
                1,
            ),
            Conv(hidden_channels, self.c, 1, 1),
        )

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)
        branch_1 = self.cv_block_2(features)
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [branch_0, branch_1, branch_2, branch_3]
        for block in self.m:
            outputs.append(block(outputs[-1]))

        return self.cv_final(torch.cat(outputs, dim=1))

#--------------------------------------------------------
class PPAMANet(nn.Module):
    """PPA replacement in MANet without additional CoordAtt."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)

        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        # 只把MANet原重复路径替换成PPA
        self.m = nn.ModuleList(
            PPA(self.c, self.c) for _ in range(n)
        )

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        hidden_dim = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, hidden_dim, 1, 1),
            GroupConv(
                hidden_dim,
                hidden_dim,
                kernel_size,
                1,
            ),
            Conv(hidden_dim, self.c, 1, 1),
        )

        # 此处不再定义CoordAtt

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)
        branch_1 = self.cv_block_2(features)  # 不加CoordAtt
        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [
            branch_0,
            branch_1,
            branch_2,
            branch_3,
        ]

        for module in self.m:
            outputs.append(module(outputs[-1]))

        fused = torch.cat(outputs, dim=1)

        # 拼接后也不加ca_global
        return self.cv_final(fused)

#-----------------------------------------------
class PPALocalCAMANet(PPAMANet):
    """
    Stage-selective PPA + local Coordinate Attention.

    - PPA replaces the repeated MANet path.
    - CoordAtt is applied only after cv_block_2.
    - No global CoordAtt after concatenation.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            shortcut=shortcut,
            p=p,
            kernel_size=kernel_size,
            g=g,
            e=e,
        )

        # 只保留局部分支上的坐标注意力
        self.ca = CoordAtt(
            in_channels=self.c,
            reduction=32,
        )

    def forward(self, x):
        features = self.cv_first(x)

        branch_0 = self.cv_block_1(features)

        # 仅在cv_block_2分支后添加一次CoordAtt
        branch_1 = self.cv_block_2(features)
        branch_1 = self.ca(branch_1)

        branch_2, branch_3 = features.chunk(2, dim=1)

        outputs = [
            branch_0,
            branch_1,
            branch_2,
            branch_3,
        ]

        for module in self.m:
            outputs.append(module(outputs[-1]))

        fused = torch.cat(outputs, dim=1)

        # 不使用ca_global
        return self.cv_final(fused)

#------------------------------------------------------
class PPAECAMANet(PPADPFE):
    """
    PPADPFE with both external CoordAtt modules replaced by ECA.

    Attention positions remain unchanged:
    1. after cv_block_2;
    2. after multi-branch concatenation and before cv_final.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            shortcut=shortcut,
            p=p,
            kernel_size=kernel_size,
            g=g,
            e=e,
        )

        # Replace the two CoordAtt modules created by PPADPFE.
        self.ca = EfficientChannelAttention(self.c)
        self.ca_global = EfficientChannelAttention((4 + n) * self.c)

#---------------------------------------------
class PPASimAMMANet(PPADPFE):
    """
    PPADPFE with both external CoordAtt modules replaced by SimAM.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            shortcut=shortcut,
            p=p,
            kernel_size=kernel_size,
            g=g,
            e=e,
        )

        self.ca = SimAM()
        self.ca_global = SimAM()
#---------------------------------
class PPAEMAMANet(PPADPFE):
    """
    PPA ConvNeck with EMA at the same two attention positions:
    1. Local feature branch
    2. Concatenated feature output
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
        factor=8,
    ):
        super().__init__(
            c1,
            c2,
            n=n,
            shortcut=shortcut,
            p=p,
            kernel_size=kernel_size,
            g=g,
            e=e,
        )

        self.ca = EMA(
            channels=self.c,
            factor=factor,
        )
        self.ca_global = EMA(
            channels=(4 + n) * self.c,
            factor=factor,
        )
#----------------------------------------------
class PPAELAMANet(PPADPFE):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5,
    ):
        super().__init__(
            c1,
            c2,
            n=n,
            shortcut=shortcut,
            p=p,
            kernel_size=kernel_size,
            g=g,
            e=e,
        )

        self.ca = ELA(self.c)
        self.ca_global = ELA((4 + n) * self.c)