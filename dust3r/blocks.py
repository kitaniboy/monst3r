# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import torch
import torch.nn as nn 
from functools import partial
import torch.nn.functional as F

import itertools
import collections.abc

from einops import rearrange, repeat, einsum
from typing import Optional

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

import croco.utils.misc as misc  # noqa

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(itertools.repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
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
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        _q, _k, _v = [qkv[:,:,i] for i in range(3)]
        
        if self.rope is not None:
            _q = self.rope(_q, xpos)
            _k = self.rope(_k, xpos)

        x = F.scaled_dot_product_attention(_q, _k, _v)
        # x = flex_attention(_q, _k, _v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class CrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                qpos: torch.Tensor,
                kpos: torch.Tensor) -> torch.Tensor:
        B, Nq, C = query.shape
        _q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        Nk = key.shape[1]
        Nv = value.shape[1]
        _k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        _v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            _q = self.rope(_q, qpos)
            _k = self.rope(_k, kpos)
            
        # if score_mod == 'causal':
        #     def causal_mask(score, b, h, q_idx, kv_idx):
        #         return torch.where(q_idx >= kv_idx, score, -float("inf"))
        #     score_mod = causal_mask
        
        # if score_mod is not None:
        #     # x = flex_attention(_q, _k, torch.ones_like(_v), score_mod=score_mod)
        #     _dummy = torch.eye(4, device='cuda').reshape(1, 1, 4, 4).repeat(1*5*768, 12, 1, 1)

        #     x = flex_attention(_q, _k, _dummy, score_mod=score_mod)
        #     x = x.reshape(1, 5, 768, 12, 4, 4)
        #     breakpoint()
        #     abc = 1

        x = F.scaled_dot_product_attention(_q, _k, _v)
        # x = flex_attention(_q, _k, _v)
        
        
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class CrossAttentionWithMask(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                qpos: torch.Tensor,
                kpos: torch.Tensor,
                attn_mask: torch.Tensor
                # kv_cache: Optional[tuple[torch.Tensor]] = None
                ):
        B, Nq, C = query.shape
        
        _q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        # if kv_cache is not None:
        #     _k, _v = kv_cache
        # else:
        Nk = key.shape[1]
        Nv = value.shape[1]
        _k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        _v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            _q = self.rope(_q, qpos)
            _k = self.rope(_k, kpos)
        
        x = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=attn_mask)
        
        # _test = x.reshape(1, 5, -1, 12, 4, 64)
        # if misc.get_rank() == 7:
        #     breakpoint()
        #     abc = 1
        
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class EncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope_space=None, rope_time=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope_space, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos.contiguous()))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope_space=None, rope_time=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope_space, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.cross_attn = CrossAttention(dim, rope=rope_space, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

        self.ls = LayerScale(dim)
        self.norm_time = norm_layer(dim)
        self.attn_time = CrossAttentionWithMask(dim, rope=rope_time, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x: torch.Tensor, y: torch.Tensor, xpos: torch.Tensor, ypos: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T1, T2, N = x.shape[:4]
        
        x = rearrange(x, 'b t1 t2 n c -> (b t1 t2) n c')
        pos = rearrange(xpos, 'b t1 t2 n three -> (b t1 t2) n three').contiguous()
        
        x = x + self.drop_path(self.attn(self.norm1(x), pos))
                
        y = rearrange(y, 'b t1 t2 n c -> (b t1 t2) n c')
        
        qpos = rearrange(xpos, 'b t1 t2 n three -> (b t1 t2) n three').contiguous()
        kpos = rearrange(ypos, 'b t1 t2 n three -> (b t1 t2) n three').contiguous()
        
        _y = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), _y, _y, qpos=qpos, kpos=kpos))
        
        x = rearrange(x, '(b t1 t2) n c -> (b t1 n) t2 c', t1=T1, t2=T2)
        
        _x = self.norm_time(x)
        _pos = rearrange(ypos, 'b t1 t2 n three -> (b t1 n) t2 three').contiguous()

        x = x + self.drop_path(self.ls(self.attn_time(_x, _x, _x, qpos=_pos, kpos=_pos, attn_mask=attn_mask)))
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        x = rearrange(x, '(b t1 n) t2 c -> b t1 t2 n c', t1=T1, t2=T2, n=N)
        
        return x

DecoderBlockFixed = DecoderBlock

# class RepeatedAttention(nn.Module):

# def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#     super().__init__()
#     self.num_heads = num_heads
#     head_dim = dim // num_heads
#     self.scale = head_dim ** -0.5

#     self.projq = nn.Linear(dim, dim, bias=qkv_bias)
#     self.projk = nn.Linear(dim, dim, bias=qkv_bias)
#     self.projv = nn.Linear(dim, dim, bias=qkv_bias)
#     self.attn_drop = nn.Dropout(attn_drop)
#     self.proj = nn.Linear(dim, dim)
#     self.proj_drop = nn.Dropout(proj_drop)
    
#     self.rope = rope
    
# def forward(self, query, key, value, qpos, kpos, qtime, ktime):
#     Nq = query.shape[1]
#     Nk = key.shape[1]
    
#     _q = rearrange(self.projq(query), '(b tq) nq (h d) -> (b tq) nq h d', tq=qtime, h=self.num_heads)
#     _k = rearrange(self.projk(key), '(b tk) nk (h d) -> (b tk) nk h d', tk=ktime, h=self.num_heads)
#     _v = rearrange(self.projv(value), '(b tk) nk (h d) -> b tk h nk d', tk=ktime, h=self.num_heads)
    
#     if self.rope is not None:
#         _q = self.rope(_q, qpos)
#         _k = self.rope(_k, kpos)

#     _q = rearrange(_q, '(b tq) nq h d -> b tq h nq d', tq=qtime)
#     _k = rearrange(_k, '(b tk) nk h d -> b tk h nk d', tk=ktime)
    
#     attn = einsum(_q, _k, 'b tq h nq d, b tk h nk d -> b tq tk h nq nk') * self.scale
#     attn = attn.softmax(dim=-1)
#     attn = self.attn_drop(attn)

#     x = einsum(attn, _v, 'b tq tk h nq nk, b tk h nk d -> b tq tk h nq d')
#     x = rearrange(x, 'b tq tk h nq d -> (b tq) nq tk (h d)')
#     x = self.proj(x)
#     x = self.proj_drop(x)
    
#     return x

        # if self.cache is None:
        #     with torch.no_grad():
        #         _b_idx = torch.arange(B*T1*N, device='cuda').reshape(B*T1*N, 1, 1, 1).expand(-1, -1, T2, T2)
        #         _q_idx = torch.arange(T2, device='cuda').reshape(1, 1, T2, 1).expand(B*T1*N, -1, -1, T2)
        #         _kv_idx = torch.arange(T2, device='cuda').reshape(1, 1, 1, T2).expand(B*T1*N, -1, T2, -1)
                
        #         batch_mask = ((_kv_idx + 1) <= ((_b_idx % (T1 * N)) // N))
        #         qkv_mask = (_q_idx >= _kv_idx)
                
        #         self.cache = torch.where(torch.logical_or(batch_mask, qkv_mask), torch.zeros_like(_q_idx), torch.ones_like(_q_idx) * -float("inf"))

        # def causal_mask(score, b_idx, h, q_idx, kv_idx):
            
        #     batch_mask = ((kv_idx + 1) <= ((b_idx % (T1 * N)) // N))
        #     qkv_mask = (q_idx >= kv_idx)
            
        #     mask = torch.where(torch.logical_or(batch_mask, qkv_mask), torch.zeros_like(q_idx), torch.ones_like(q_idx) * -float("inf"))
        
        #     return score + mask
        
        # if self.time_attention is None:
        #     self.time_attention = torch.compile(F.scaled_dot_product_attention, fullgraph=True)
        
        # if self.cache.get((B, T1, N), None) is None:
        #     with torch.no_grad():
        #         _b_idx = torch.arange(B*T1*N, device='cuda').reshape(B*T1*N, 1, 1, 1).expand(-1, -1, T2, T2)
        #         _q_idx = torch.arange(T2, device='cuda').reshape(1, 1, T2, 1).expand(B*T1*N, -1, -1, T2)
        #         _kv_idx = torch.arange(T2, device='cuda').reshape(1, 1, 1, T2).expand(B*T1*N, -1, T2, -1)
                
        #         batch_mask = ((_kv_idx + 1) <= ((_b_idx % (T1 * N)) // N))
        #         qkv_mask = (_q_idx >= _kv_idx)
                
        #         mask = torch.where(torch.logical_or(batch_mask, qkv_mask), torch.zeros_like(_q_idx), torch.ones_like(_q_idx) * -float("inf"))
        #         mask.requires_grad = False
        #         self.cache[(B, T1, N)] = mask
                
        # attn_mask = self.cache[(B, T1, N)]
        # _attention_time = partial(F.scaled_dot_product_attention, attn_mask=attn_mask)
        # def causal_mask(score, b_idx, h, q_idx, kv_idx):
        #     batch_mask = ((kv_idx + 1) <= ((b_idx % (T1 * N)) // N))
        #     qkv_mask = (q_idx >= kv_idx)
            
        #     return torch.where(torch.logical_or(batch_mask, qkv_mask), score, -float("inf"))
        
        # _flex_attention_time = torch.compile(flex_attention)