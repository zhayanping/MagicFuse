# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy

import torch
import torch.nn as nn
from pretrained.IKR.unet_utils import (
    BasicTransformerBlock, Downsample, ResBlock, SpatialTransformer,
    SpatialTransformerV2, Timestep, TimestepEmbedSequential,
    TransformerBlockV2, Upsample, conv_nd, linear, normalization,
    timestep_embedding, zero_module)

def exists(x):
    return x is not None

def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass

class DiffusionUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.init_params(cfg)
        self.construct_network()

    def init_params(self, cfg):
        self.in_channels = cfg.IN_CHANNELS
        self.model_channels = cfg.MODEL_CHANNELS
        self.out_channels = cfg.OUT_CHANNELS
        self.num_res_blocks = cfg.NUM_RES_BLOCKS
        self.attention_resolutions = cfg.ATTENTION_RESOLUTIONS

        self.num_heads = cfg.get('NUM_HEADS', -1)
        self.num_head_channels = cfg.get('NUM_HEADS_CHANNELS', -1)
        self.dropout = cfg.get('DROPOUT', 0)
        self.channel_mult = tuple(cfg.get('CHANNEL_MULT', [1, 2, 4, 4]))
        self.conv_resample = cfg.get('CONV_RESAMPLE', True)
        self.dims = cfg.get('DIMS', 2)

        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.use_scale_shift_norm = cfg.get('USE_SCALE_SHIFT_NORM', False)
        self.resblock_updown = cfg.get('RESBLOCK_UPDOWN', False)
        self.use_new_attention_order = cfg.get('USE_NEW_ATTENTION_ORDER', True)
        self.use_spatial_transformer = cfg.get('USE_SPATIAL_TRANSFORMER', True)
        self.transformer_depth = cfg.get('TRANSFORMER_DEPTH', 1)
        self.use_linear_in_transformer = cfg.get('USE_LINEAR_IN_TRANSFORMER',
                                                 False)
        self.disable_self_attentions = cfg.get('DISABLE_SELF_ATTENTIONS', None)
        self.disable_middle_self_attn = cfg.get('DISABLE_MIDDLE_SELF_ATTN',
                                                False)
        self.adm_in_channels = cfg.get('ADM_IN_CHANNELS', None)
        self.ignore_keys = cfg.get('IGNORE_KEYS', [])

        assert (self.num_heads > 0 or self.num_head_channels > 0) and \
               (self.num_heads == -1 or self.num_head_channels == -1)

        if isinstance(self.num_res_blocks, int):
            self.num_res_blocks = len(
                self.channel_mult) * [self.num_res_blocks]
        elif len(self.num_res_blocks) != len(self.channel_mult):
            raise ValueError(
                'provide num_res_blocks either as an int (globally constant) or '
                'as a list/tuple (per-level) with the same length as channel_mult'
            )

    def construct_network(self):
        in_channels = self.in_channels
        model_channels = self.model_channels
        out_channels = self.out_channels
        attention_resolutions = self.attention_resolutions
        channel_mult = self.channel_mult

        num_heads = self.num_heads
        num_head_channels = self.num_head_channels
        dims = self.dims
        dropout = self.dropout
        use_checkpoint = self.use_checkpoint
        use_scale_shift_norm = self.use_scale_shift_norm
        disable_self_attentions = self.disable_self_attentions
        disable_middle_self_attn = self.disable_middle_self_attn
        transformer_depth = self.transformer_depth

        use_linear_in_transformer = self.use_linear_in_transformer
        resblock_updown = self.resblock_updown
        conv_resample = self.conv_resample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        input_down_flag = [False]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    #print(f'==> attention resolution={ds}.')
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                input_down_flag.append(False)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                input_down_flag.append(True)
                ds *= 2
                self._feature_size += ch
        self._input_block_chans = copy.deepcopy(input_block_chans)
        self._input_down_flag = input_down_flag

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch,
                               num_heads,
                               dim_head,
                               depth=transformer_depth,
                               disable_self_attn=disable_middle_self_attn,
                               use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self._middle_block_chans = [ch]

        self._output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        self.lsc_identity = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.lsc_identity.append(nn.Identity())
                self._feature_size += ch
                self._output_block_chans.append(ch)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def _forward_origin(self, x, emb, context=None, hint=None, **kwargs):
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        #print(h.shape)
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            if 'tuner_scale' in kwargs and kwargs[
                    'tuner_scale'] is not None and kwargs['tuner_scale'] < 1.0:
                tuner_scale = kwargs['tuner_scale']
                tuner_h = self.lsc_identity[m_id](skip_h) - skip_h
                h = torch.cat([h, skip_h + tuner_scale * tuner_h], dim=1)
            else:
                h = torch.cat([h, self.lsc_identity[m_id](skip_h)], dim=1)
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size)
        out = self.out(h)
        return out

    def forward(self, x, t=None, **kwargs): #[gt raw]
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        context = None
        out = self._forward_origin(x, emb, context, **kwargs)
        return out

