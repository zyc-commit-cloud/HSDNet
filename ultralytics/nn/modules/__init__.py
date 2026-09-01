# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, MANet, HyperComputeModule, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f,
                    C3Ghost, C3x, GhostBottleneck, HGBlock, HGStem, Proto, RepC3, ResNetLayer, SimAM, MANet_SimAM,
    DPFE,CoordAtt,MAN_sea,SEAttention,MAN_ela,ELA,MAN_eca,EfficientChannelAttention,MAN_ema,EMA,C2f_Ghost,MANet_pconv,Bottleneck_PConv,Partial_conv3,Conv_GN,Scale,DySample,MANet_Ghost,SCDown,SPDConv,MANet_Faster,Faster_Block,DropPath,MPCA,HaarWaveletConv,WaveletConv,WaveletConvNeck,WCMANet,GroupBatchnorm2d,SRU,CRU,ScConv,ScConvNeck,SCCMANet,LSKBlock_SA,LSKBlock,LSKConvNeck,LSKCMANet,RepConvN,RepConvNeck,RCMANet,DPDown,DBBDPFE,PPADPFE,FCMANet,DRCMANet,DBBMANet,PPAMANet,PPALocalCAMANet,LocalCAMANet,PPAECAMANet,PPASimAMMANet,PPAEMAMANet,PPAELAMANet)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, GroupConv, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment,Detect_LSCD
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer',
           'MANet','C2f_Ghost', 'HyperComputeModule','GroupConv','SimAM','MANet_SimAM','CoordAtt','DPFE','ELA','MAN_ela','EfficientChannelAttention','MAN_eca','MAN_ema','EMA','Partial_conv3','MANet_pconv','Bottleneck_PConv','Detect_LSCD','DySample','MANet_Ghost','SCDown','SPDConv','MANet_Faster','Faster_Block','DropPath','MPCA','HaarWaveletConv','WaveletConv','WaveletConvNeck','WCMANet','GroupBatchnorm2d','SRU','CRU','ScConv','ScConvNeck','SCCMANet','LSKBlock_SA','LSKBlock','LSKConvNeck','LSKCMANet','RepConvN','RepConvNeck','RCMANet','DBBDPFE','PPADPFE','FCMANet','DRCMANet','DBBMANet','PPAMANet','PPALocalCAMANet','LocalCAMANet','PPAECAMANet','PPASimAMMANet','PPAEMAMANet','PPAELAMANet')


