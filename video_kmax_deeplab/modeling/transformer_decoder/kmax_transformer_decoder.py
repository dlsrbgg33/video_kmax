# Reference: https://github.com/google-research/deeplab2/blob/main/model/transformer_decoder/kmax.py
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
# Modified by Qihang Yu

from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
# from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from timm.models.layers import trunc_normal_

from detectron2.config import configurable
from detectron2.utils.registry import Registry

from ..pixel_decoder.kmax_pixel_decoder import get_norm, ConvBN

from typing import Optional
import math


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""

import numpy as np



# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/maskformer_transformer_decoder.py#L22
def build_transformer_decoder(cfg, input_shape_from_backbone):
    """
    Build a instance embedding branch from `cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME`.
    """
    name = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, input_shape_from_backbone)


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/decoder/max_deeplab.py#L60
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    # query_class_logits: B x N x C
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L41
class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        self._batch_norm_similarity = get_norm('syncbn', num_heads)
        self._batch_norm_retrieved_value = get_norm('syncbn', channels_v)

    def forward(self, query, key, value):
        # query/key/value: N x num_heads x C x L
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)
        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits, dim=-1)
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        # N x C x L
        return retrieved_value


# https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py#L32
class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133+1, num_frames=1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='syncbn', act=None)

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')

        self._pixel_space_mask_batch_norm = get_norm('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)
        self.num_frames = num_frames

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # cluster_centers: B x C x N
        # pixel feature: B x C x H x W
        # if num_frames_off:
        #     self.num_frames = num_frames_off
            
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        B,C,H,W = pixel_space_feature.shape
        _, _, num_mask = mask_embeddings.shape

        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)
        
        pixel_space_normalized_feature = pixel_space_normalized_feature.reshape(-1, self.num_frames, C, H, W)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous() # BxCxN->BxNxC, to align with mask2former format
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)

        mask_logits = torch.einsum('btchw,bcn->btnhw',
          pixel_space_normalized_feature, cluster_mask_kernel)
        pixel_space_normalized_feature = pixel_space_normalized_feature.reshape(-1, C, H, W)
        mask_logits = mask_logits.reshape(-1, num_mask, H, W)
        

        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)


        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature,
            'cluster_feat': cluster_mask_kernel}


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class kMaXTransformerLayer(nn.Module):
    def __init__(
        self,
        num_classes=133,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0, # drop prob instead of keep prob in tf2
        advanced_kmax=False,
        skip_conn_init_value=0.0,
        num_frames=1
    ):
        super().__init__()

        self._num_classes = num_classes
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        self.advanced_kmax = advanced_kmax

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        
        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu')
                                          
        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')

        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None)

        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
            in_channel_query=self._bottleneck_channels, num_classes=num_classes, num_frames=num_frames)
        self._kmeans_query_batch_norm_retrieved_value = get_norm('syncbn', self._total_value_depth)

        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self.num_frames = num_frames


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, pixel_feature, query_feature):

        # if num_frames_off:
        #     self.num_frames = num_frames_off

        N, C, H, W = pixel_feature.shape
        N = N // self.num_frames
        _, D, L = query_feature.shape
        
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # N//num_frames, C, H*W*num_frames
        query_space = self._query_conv1_bn_act(query_feature) # N x C x L

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space) # N C H W
        pixel_value = pixel_value.reshape(-1, self._total_value_depth, H*W)
        pixel_value = torch.cat(torch.split(pixel_value, 1, dim=0), dim=-1)
        
        # k-means assignment.
        pixel_space_2d = pixel_space.reshape(-1, self._bottleneck_channels, H, W)
        prediction_result = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space_2d)
        clustering_result = prediction_result['mask_logits'] # N L HW
        
        with torch.no_grad():
            clustering_result = clustering_result.reshape(N*self.num_frames, clustering_result.shape[1], -1).detach()
            clustering_result = torch.cat(torch.split(clustering_result, 1, dim=0), dim=-1)

            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        # k-means update.
        kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) # N x C x L
        if self.advanced_kmax:
            pixel_num = clustering_result.sum(-1).unsqueeze(1) # N x 1 x L
            kmeans_update = kmeans_update / torch.clamp(pixel_num.float(), min=1.0)
        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class CrossAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0, # drop prob instead of keep prob in tf2
        skip_conn_init_value=0.0,
        num_frames=1
    ):
        super().__init__()

        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 

        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        
        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')

        self._pixel_kv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None)

        self._query_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=skip_conn_init_value)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=skip_conn_init_value)
        self.num_frames = num_frames

    def forward(self, pixel_feature, query_feature):
        N, C, H, W = pixel_feature.shape
        N = N // self.num_frames
        _, D, L = query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # N C H W
        query_space = self._query_conv1_bn_act(query_feature) # N x C x L

        pixel_kv = self._pixel_kv_conv_bn(pixel_space) # N C H W
        pixel_k, pixel_v = torch.split(pixel_kv, [self._total_key_depth, self._total_value_depth], dim=1)
        pixel_k = pixel_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, H*W*self.num_frames)
        pixel_v = pixel_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, H*W*self.num_frames)

        # query self&cross-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, L)
        attn_update = self._query_attention(query_q, torch.cat([query_k, pixel_k], dim=-1), torch.cat([query_v, pixel_v], dim=-1))
        attn_update = self._query_conv3_bn(attn_update)
        query_feature = query_feature + self.drop_path_attn(attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature
    

class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, atrous_rates):
        super().__init__()

        self._aspp_conv0 = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                  norm='syncbn', act='gelu')

        rate1, rate2, rate3 = atrous_rates
        self._aspp_conv1 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate1, padding=rate1, bias=False,
                                  norm='syncbn', act='gelu')

        self._aspp_conv2 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate2, padding=rate2, bias=False,
                                  norm='syncbn', act='gelu')

        self._aspp_conv3 = ConvBN(in_channels, output_channels, kernel_size=3, dilation=rate3, padding=rate3, bias=False,
                                  norm='syncbn', act='gelu')

        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._aspp_pool = ConvBN(in_channels, output_channels, kernel_size=1, bias=False,
                                 norm='syncbn', act='gelu')

        self._proj_conv_bn_act = ConvBN(output_channels * 5, output_channels, kernel_size=1, bias=False,
                                 norm='syncbn', act='gelu')
        # https://github.com/google-research/deeplab2/blob/main/model/decoder/aspp.py#L249
        self._proj_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        results = []
        results.append(self._aspp_conv0(x))
        results.append(self._aspp_conv1(x))
        results.append(self._aspp_conv2(x))
        results.append(self._aspp_conv3(x))
        align_corners = (x.shape[-1] % 2 == 1)
        results.append(F.interpolate(self._aspp_pool(self._avg_pool(x)), size=x.shape[-2:], mode='bilinear', align_corners=align_corners))

        x = torch.cat(results, dim=1)
        x = self._proj_conv_bn_act(x)
        x = self._proj_drop(x)

        return x


class SemanticPredictor(nn.Module):
    def __init__(self, in_channels, os8_channels, os4_channels, num_classes):
        super().__init__()

        # Below is PanopticDeepLabSingleDecoder
        self._aspp = ASPP(
            in_channels=in_channels,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L35
            output_channels=256,
            # https://github.com/google-research/deeplab2/blob/main/configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto#L36
            atrous_rates=[6,12,18])
        
        self._low_level_projection_os8 = ConvBN(os8_channels, 64, kernel_size=1, bias=False,
                                                norm='syncbn', act='gelu')

        self._low_level_fusion_os8_conv0_bn_act = ConvBN(256 + 64, 256 + 64, groups=256 + 64, kernel_size=5, padding=2, bias=False,
                                                         norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._low_level_fusion_os8_conv1_bn_act = ConvBN(256 + 64, 256, kernel_size=1,bias=False,
                                                         norm='syncbn', act='gelu')

        self._low_level_projection_os4 = ConvBN(os4_channels, 32, kernel_size=1, bias=False,
                                                norm='syncbn', act='gelu')

        self._low_level_fusion_os4_conv0_bn_act = ConvBN(256 + 32, 256 + 32, groups=256 + 32, kernel_size=5, padding=2, bias=False,
                                                         norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._low_level_fusion_os4_conv1_bn_act = ConvBN(256 + 32, 256, kernel_size=1,bias=False,
                                                         norm='syncbn', act='gelu')

        # Below is PanopticDeepLabSingleHead
        self.conv_block_0 = ConvBN(256, 256, groups=256, kernel_size=5, padding=2, bias=False,
                                   norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self.conv_block_1 = ConvBN(256, 256, kernel_size=1,bias=False,
                                   norm='syncbn', act='gelu')
        self.final_conv = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None)

        trunc_normal_(self.final_conv.conv.weight, std=0.01)


    def forward(self, x, low_features_os8, low_features_os4):
        x = self._aspp(x)
        align_corners = (x.shape[-1] % 2 == 1)
        low_features_os8 = self._low_level_projection_os8(low_features_os8)
        x = F.interpolate(x, size=low_features_os8.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.cat([x, low_features_os8], dim=1)
        x = self._low_level_fusion_os8_conv0_bn_act(x)
        x = self._low_level_fusion_os8_conv1_bn_act(x)

        low_features_os4 = self._low_level_projection_os4(low_features_os4)
        x = F.interpolate(x, size=low_features_os4.shape[-2:], mode='bilinear', align_corners=align_corners)
        x = torch.cat([x, low_features_os4], dim=1)
        x = self._low_level_fusion_os4_conv0_bn_act(x)
        x = self._low_level_fusion_os4_conv1_bn_act(x)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.final_conv(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class VideokMaXTransformerDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        dec_layers: List[int],
        in_channels: List[int],
        num_classes: int,
        num_queries: int,
        drop_path_prob: float,
        add_aux_semantic_pred: bool,
        advanced_kmax: bool,
        skip_conn_init_value: float,
        input_shape_from_backbone,
        num_frames
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            num_frames: int, number of frames
        """
        super().__init__()
        
        # define Transformer decoder here
        self._kmax_transformer_layers = nn.ModuleList()
        self._num_blocks = dec_layers
        self._advanced_kmax = advanced_kmax
        os2channels = {32: in_channels[0], 16: in_channels[1], 8: in_channels[2]}

        if self._advanced_kmax:
            self._cross_attention_layers = nn.ModuleList()
            for _ in range(3):
                self._cross_attention_layers.append(
                    CrossAttentionTransformerLayer(
                    in_channel_pixel=os2channels[32],
                    in_channel_query=256,
                    base_filters=128,
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2,
                    drop_path_prob=0.0,
                    skip_conn_init_value=skip_conn_init_value,
                    num_frames=num_frames)
                )

        for index, output_stride in enumerate([32, 16, 8]):
            for _ in range(self._num_blocks[index]):
                self._kmax_transformer_layers.append(
                    kMaXTransformerLayer(num_classes=num_classes+1,
                    in_channel_pixel=os2channels[output_stride],
                    in_channel_query=256,
                    base_filters=128,
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2,
                    drop_path_prob=drop_path_prob,
                    advanced_kmax=advanced_kmax,
                    skip_conn_init_value=skip_conn_init_value,
                    num_frames=num_frames)
                )

        self._num_queries = num_queries
        # learnable query features
        self._cluster_centers = nn.Embedding(256, num_queries)
        trunc_normal_(self._cluster_centers.weight, std=1.0)

        self._class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')

        self._mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')

        self._predcitor = kMaXPredictor(in_channel_pixel=256,
            in_channel_query=256, num_classes=num_classes+1, num_frames=num_frames)


        self._add_aux_semantic_pred = add_aux_semantic_pred
        if add_aux_semantic_pred:
            self._auxiliary_semantic_predictor = SemanticPredictor(
                in_channels=input_shape_from_backbone['res5'].channels,
                os8_channels=input_shape_from_backbone['res3'].channels,
                os4_channels=input_shape_from_backbone['res2'].channels,
                # +1 for void.
                num_classes=num_classes+1)

        self.num_frames = num_frames

    @classmethod
    def from_config(cls, cfg, input_shape_from_backbone):
        ret = {}
        ret["dec_layers"] = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DEC_LAYERS
        ret["in_channels"] = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.IN_CHANNELS   
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["num_queries"] = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES
        ret["drop_path_prob"] = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DROP_PATH_PROB
        ret["add_aux_semantic_pred"] = (cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_WEIGHT > 0)
        ret["advanced_kmax"] = cfg.MODEL.KMAX_DEEPLAB.ADVANCED_KMAX
        ret["skip_conn_init_value"] = cfg.MODEL.KMAX_DEEPLAB.SKIP_CONN_INIT_VALUE
        ret["input_shape_from_backbone"] = input_shape_from_backbone
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        return ret


    def forward(self, x, panoptic_features, semantic_features):
        
        B = x[0].shape[0] // self.num_frames
        
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L
        
        current_transformer_idx = 0

        predictions_class = []
        predictions_mask = []
        predictions_pixel_feature = []

        if self._advanced_kmax:
            for _ in range(3):
                cluster_centers = self._cross_attention_layers[_](
                    pixel_feature=x[0], query_feature=cluster_centers
                )
                
        for i, feat in enumerate(x):
            for _ in range(self._num_blocks[i]):
                
                cluster_centers, prediction_result = self._kmax_transformer_layers[current_transformer_idx](
                    pixel_feature=feat, query_feature=cluster_centers
                )
                predictions_class.append(prediction_result['class_logits'])
                predictions_mask.append(prediction_result['mask_logits'])
                predictions_pixel_feature.append(prediction_result['pixel_feature'])
                current_transformer_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)
        

        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=panoptic_features
        )
        
        predictions_class.append(prediction_result['class_logits'])
        predictions_mask.append(prediction_result['mask_logits'])
        predictions_pixel_feature.append(prediction_result['pixel_feature'])

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pixel_feature': predictions_pixel_feature[-1],
            'cluster_feature': mask_embeddings,
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, predictions_pixel_feature
            ),      
        }

        if self._add_aux_semantic_pred and self.training:
            semantic_features, low_features_os8, low_features_os4 = semantic_features
            aux_semantic_prediction = self._auxiliary_semantic_predictor(
                x=semantic_features, low_features_os8=low_features_os8, low_features_os4=low_features_os4)
            out.update({'aux_semantic_pred': aux_semantic_prediction,})
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_pixel_feature):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # Upsample all aux predictions to highest output res.
        target_size = outputs_seg_masks[-1].shape[-2:]
        align_corners = (target_size[0] % 2 == 1)
        return [
            {"pred_logits": a, "pred_masks": F.interpolate(b, size=target_size, mode="bilinear", align_corners=align_corners),
            "pixel_feature": F.interpolate(c, size=target_size, mode="bilinear", align_corners=align_corners),}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_pixel_feature[:-1])
        ]
