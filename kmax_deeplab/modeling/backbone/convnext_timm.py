""" ConvNeXt

Papers:
* `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}

* `ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}

Original code and weights from:
* https://github.com/facebookresearch/ConvNeXt, original copyright below
* https://github.com/facebookresearch/ConvNeXt-V2, original copyright below

Model defs atto, femto, pico, nano and _ols / _hnf variants are timm originals.

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# ConvNeXt
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license

# ConvNeXt-V2
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (Attribution-NonCommercial 4.0 International (CC BY-NC 4.0))
# No code was used directly from ConvNeXt-V2, however the weights are CC BY-NC 4.0 so beware if using commercially.

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.layers import NormMlpClassifierHead, ClassifierHead
# from ._manipulate import named_apply, checkpoint_seq
# from ._registry import generate_default_cfgs, register_model, register_model_deprecations
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torch.cuda.amp import autocast

__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        with autocast(enabled=False):
            x = x.float()
            if self.data_format == "channels_last":
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x
   

class ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                # norm_layer(in_chs),
                LayerNorm(in_chs, eps=1e-6, data_format="channels_first"),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            depths: Tuple[int, ...] = (3, 3, 9, 3),
            dims: Tuple[int, ...] = (96, 192, 384, 768),
            kernel_sizes: Union[int, Tuple[int, ...]] = 7,
            ls_init_value: Optional[float] = 1e-6,
            stem_type: str = 'patch',
            patch_size: int = 4,
            head_init_scale: float = 1.,
            head_norm_first: bool = False,
            head_hidden_size: Optional[int] = None,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Union[str, Callable]] = None,
            # norm_eps: Optional[float] = None,
            norm_eps: float = 1e-5,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            out_indices=[0, 1, 2, 3]
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            # norm_layer = LayerNorm2d
            norm_layer = LayerNorm
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        self.num_features = dims
        # import pdb; pdb.set_trace()
        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                # norm_layer(dims[0]),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

        self.out_indices = out_indices
        
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        outs = {}
        x = F.pad(x, (1, 2, 1, 2, 0, 0, 0, 0), "constant", 0)
        x = self.stem(x)
        for i in range(4):
            # We add zero padding here for downstream tasks.
            # ref: https://github.com/google-research/deeplab2/blob/main/model/pixel_encoder/convnext.py#L128
            if i != 0:
                x = F.pad(x, (0, 1, 0, 1, 0, 0, 0, 0), "constant", 0)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs["res{}".format(i + 2)] = x

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module



def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


@BACKBONE_REGISTRY.register()
class D2ConvNeXtTimm(ConvNeXt, Backbone):
    def __init__(self, cfg, input_shape):

        in_chans = cfg.MODEL.CONVNEXT.IN_CHANNELS
        depths = cfg.MODEL.CONVNEXT.DEPTHS
        dims = cfg.MODEL.CONVNEXT.DIMS
        drop_path_rate = cfg.MODEL.CONVNEXT.DROP_PATH_RATE

        super().__init__(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate
        )

        self._out_features = cfg.MODEL.CONVNEXT.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"ConvNeXt takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return -1

# @register_model
# def convnext_atto(pretrained=False, **kwargs):
#     # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
#     model_args = dict(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True)
#     model = _create_convnext('convnext_atto', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_atto_ols(pretrained=False, **kwargs):
#     # timm femto variant with overlapping 3x3 conv stem, wider than non-ols femto above, current param count 3.7M
#     model_args = dict(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, stem_type='overlap_tiered')
#     model = _create_convnext('convnext_atto_ols', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_femto(pretrained=False, **kwargs):
#     # timm femto variant
#     model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True)
#     model = _create_convnext('convnext_femto', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_femto_ols(pretrained=False, **kwargs):
#     # timm femto variant
#     model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, stem_type='overlap_tiered')
#     model = _create_convnext('convnext_femto_ols', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_pico(pretrained=False, **kwargs):
#     # timm pico variant
#     model_args = dict(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True)
#     model = _create_convnext('convnext_pico', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_pico_ols(pretrained=False, **kwargs):
#     # timm nano variant with overlapping 3x3 conv stem
#     model_args = dict(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True,  stem_type='overlap_tiered')
#     model = _create_convnext('convnext_pico_ols', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_nano(pretrained=False, **kwargs):
#     # timm nano variant with standard stem and head
#     model_args = dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True)
#     model = _create_convnext('convnext_nano', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_nano_ols(pretrained=False, **kwargs):
#     # experimental nano variant with overlapping conv stem
#     model_args = dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, stem_type='overlap')
#     model = _create_convnext('convnext_nano_ols', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_tiny_hnf(pretrained=False, **kwargs):
#     # experimental tiny variant with norm before pooling in head (head norm first)
#     model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True)
#     model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_tiny(pretrained=False, **kwargs):
#     model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
#     model = _create_convnext('convnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_small(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
#     model = _create_convnext('convnext_small', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_base(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
#     model = _create_convnext('convnext_base', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_large(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
#     model = _create_convnext('convnext_large', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_large_mlp(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], head_hidden_size=1536)
#     model = _create_convnext('convnext_large_mlp', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_xlarge(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
#     model = _create_convnext('convnext_xlarge', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnext_xxlarge(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], norm_eps=kwargs.pop('norm_eps', 1e-5))
#     model = _create_convnext('convnext_xxlarge', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_atto(pretrained=False, **kwargs):
#     # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
#     model_args = dict(
#         depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), use_grn=True, ls_init_value=None, conv_mlp=True)
#     model = _create_convnext('convnextv2_atto', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_femto(pretrained=False, **kwargs):
#     # timm femto variant
#     model_args = dict(
#         depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), use_grn=True, ls_init_value=None, conv_mlp=True)
#     model = _create_convnext('convnextv2_femto', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_pico(pretrained=False, **kwargs):
#     # timm pico variant
#     model_args = dict(
#         depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), use_grn=True, ls_init_value=None, conv_mlp=True)
#     model = _create_convnext('convnextv2_pico', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_nano(pretrained=False, **kwargs):
#     # timm nano variant with standard stem and head
#     model_args = dict(
#         depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), use_grn=True, ls_init_value=None, conv_mlp=True)
#     model = _create_convnext('convnextv2_nano', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_tiny(pretrained=False, **kwargs):
#     model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), use_grn=True, ls_init_value=None)
#     model = _create_convnext('convnextv2_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_small(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], use_grn=True, ls_init_value=None)
#     model = _create_convnext('convnextv2_small', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_base(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=True, ls_init_value=None)
#     model = _create_convnext('convnextv2_base', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_large(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=True, ls_init_value=None)
#     model = _create_convnext('convnextv2_large', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def convnextv2_huge(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, ls_init_value=None)
#     model = _create_convnext('convnextv2_huge', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# register_model_deprecations(__name__, {
#     'convnext_tiny_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k',
#     'convnext_small_in22ft1k': 'convnext_small.fb_in22k_ft_in1k',
#     'convnext_base_in22ft1k': 'convnext_base.fb_in22k_ft_in1k',
#     'convnext_large_in22ft1k': 'convnext_large.fb_in22k_ft_in1k',
#     'convnext_xlarge_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k',
#     'convnext_tiny_384_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k_384',
#     'convnext_small_384_in22ft1k': 'convnext_small.fb_in22k_ft_in1k_384',
#     'convnext_base_384_in22ft1k': 'convnext_base.fb_in22k_ft_in1k_384',
#     'convnext_large_384_in22ft1k': 'convnext_large.fb_in22k_ft_in1k_384',
#     'convnext_xlarge_384_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k_384',
#     'convnext_tiny_in22k': 'convnext_tiny.fb_in22k',
#     'convnext_small_in22k': 'convnext_small.fb_in22k',
#     'convnext_base_in22k': 'convnext_base.fb_in22k',
#     'convnext_large_in22k': 'convnext_large.fb_in22k',
#     'convnext_xlarge_in22k': 'convnext_xlarge.fb_in22k',
# })