from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
import numpy as np
from mamba_ssm import Mamba
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalProcessLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TemporalProcessLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)        
        return x

class VideoProcessTransformerLayer(nn.Module):
    def __init__(self, in_features, temporal_out_features):
        super(VideoProcessTransformerLayer, self).__init__()
        self.temporal_processLayer1 = TemporalProcessLayer(in_features, temporal_out_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, features, sequence_length]
        x = self.temporal_processLayer1(x)
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, features, sequence_length]
        return x

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()

        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        self.conv_pad_sizes = []
        # for krnl in kernel_sizes:
        #     self.conv_pad_sizes.append([i // 2 for i in krnl])
        for krnl in kernel_sizes:
            if isinstance(krnl, int):  # If it's a single integer
                self.conv_pad_sizes.append(krnl // 2)
            else:  # If it's a list or tuple of integers
                self.conv_pad_sizes.append([i // 2 for i in krnl])
        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )


        input_channels = stem_channels

        stages = []
        for s in range(n_stages):

            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )


            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output

class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)

        in_dim = 512 + 1 if self.args.border else 512
        # in_dim_processLayer = 2048
        # features_per_stage=(32, 64, 128, 256, 320, 320)
        self.gru = nn.GRU(in_dim, 1024, 4, batch_first=True, bidirectional=True, dropout=0.2)
        features_per_stage=(25, 25)
        
        self.v_cls = nn.Linear(2048, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=dropout, activation='gelu', batch_first=True, norm_first=False)
        self.VideoProcessTransformerLayer = VideoProcessTransformerLayer(in_dim, 512)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.encoder = UNetResEncoder(
            input_channels=25,
            n_stages=2,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv1d,
            kernel_sizes=[3, 3],
            strides=(1, 1),
            n_blocks_per_stage=(3, 3),
            conv_bias=True,
            norm_op=nn.InstanceNorm1d,
            norm_op_kwargs={},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            return_skips=True,
            stem_channels=1
        )

    def forward(self, v, border=None):
        fr = v.shape[1]
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)
        f_v = f_v.float()

        # Add border information if present
        if self.args.border:
            border = border[:, :, None]
            f_v = torch.cat([f_v, border], -1)
        # bigru
        h_bigru, _ = self.gru(f_v)

        x = self.encoder(f_v)
        x = self.VideoProcessTransformerLayer(f_v)
        x = self.transformer_encoder(x)

        h_tr = x
        
        h = torch.cat([h_bigru,h_tr], dim=-1)
        
        y_v = self.v_cls(self.dropout(h.mean(1)))
        # y_v = self.v_cls(h)

        return y_v


