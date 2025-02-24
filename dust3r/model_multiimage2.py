# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from packaging import version
import huggingface_hub
from functools import partial
from einops import rearrange, repeat, einsum

from .utils.misc import fill_default_args, freeze_all_params, token_rearrange_fn
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed

from dust3r.blocks import EncoderBlock, DecoderBlock, DecoderBlockFixed
from dust3r.curope3d import RoPE as RoPE3D

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

class AsymmetricCroCo3DMultiImage2 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 decoder_block_cls='DecoderBlock',
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        del self.rope
        self.rope_space = RoPE3D(freq=100.0, F0=1.0, x_dim=32, y_dim=32, z_dim=0)
        self.rope_time = RoPE3D(freq=100.0, F0=1.0, x_dim=16, y_dim=16, z_dim=32)

        del self.enc_blocks
        self.enc_embed_dim = self.croco_args['enc_embed_dim']
        self.enc_depth = self.croco_args['enc_depth']
        self.enc_num_heads = self.croco_args['enc_num_heads']
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(self.enc_embed_dim, self.enc_num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_space=self.rope_space, rope_time=self.rope_time)
            for i in range(self.enc_depth)])

        del self.dec_blocks
        self.dec_embed_dim = self.croco_args['dec_embed_dim']
        self.dec_depth = self.croco_args['dec_depth']
        self.dec_num_heads = self.croco_args['dec_num_heads']
        
        self.dec_blocks = nn.ModuleList([
            eval(decoder_block_cls)(self.dec_embed_dim, self.dec_num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_mem=True, rope_space=self.rope_space, rope_time=self.rope_time)
            for i in range(self.dec_depth)])
     
        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.hooks_idx = set(self.downstream_head1.dpt.hooks)
        self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return
    
    def log_parameters(self):
        outputs = {}
        for block_idx in range(self.dec_depth):
            outputs[f'dec1.blk{block_idx:02d}'] = self.dec_blocks[block_idx].ls.gamma.data
            outputs[f'dec2.blk{block_idx:02d}'] = self.dec_blocks2[block_idx].ls.gamma.data
            
        return outputs

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        # self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        # self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
            
    def _patchfy(self, x, time, is_landscape, landscape_all, portrait_all):
        B = is_landscape.size(0)
        
        if landscape_all:
            x, pos = self.patch_embed(x, landscape=True)
            
        elif portrait_all:
            x, pos = self.patch_embed(x, landscape=False)
            
        else:
            is_portrait = ~is_landscape
            
            l_x, l_pos = self.patch_embed(x[is_landscape], landscape=True)
            p_x, p_pos = self.patch_embed(x[is_portrait], landscape=False)
            
            x = l_x.new_empty((B, *l_x.shape[1:]))
            pos = l_pos.new_empty((B, *l_pos.shape[1:]))
            
            x[is_landscape] = l_x
            x[is_portrait] = p_x
            pos[is_landscape] = l_pos
            pos[is_portrait] = p_pos
            
            del l_x, l_pos, p_x, p_pos
        
        N = x.size(1)
        
        time = repeat(time, 'b 1 -> b n 1', n=N)
        pos = torch.cat([pos, time], dim=-1)
        
        return x, pos
    
    def _encode_path(self, x, pos):
        for blk in self.enc_blocks:
            x = blk(x, pos)
            
        x = self.enc_norm(x)
        x_embed = self.decoder_embed(x)
        
        return x, x_embed

    def _decode_path(self, x, x_embed, pos, rearrange_fn, attn_mask):
        f2 = f1 = rearrange_fn.repeat_tokens(x)
        pos2 = pos1 = rearrange_fn.repeat_tokens(pos)
        pos2_swap, pos1_swap = rearrange_fn.swap_tokens(pos2), rearrange_fn.swap_tokens(pos1)
        final_output = [(f1, f2)]
        
        f2 = f1 = rearrange_fn.repeat_tokens(x_embed)
        final_output.append((f1, f2))
                
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            x, y = final_output[-1][::+1]
            # img1 side
            f1 = blk1(x, rearrange_fn.swap_tokens(y), pos1, pos2_swap, attn_mask)
            # img2 side
            f2 = blk2(y, rearrange_fn.swap_tokens(x), pos2, pos1_swap, attn_mask)
            
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        dec1, dec2 = zip(*final_output)
        
        return dec1, dec2

    def _head_path(self, tok1, tok2, tensor_shape, is_landscape, landscape_all, portrait_all):
        (H, W) = tensor_shape
        def transposed(dic): return {k: v.swapaxes(1, 2) for k, v in dic.items()}
        if landscape_all:
            res1 = self.downstream_head1(tok1, (H, W))
            res2 = self.downstream_head2(tok2, (H, W))

        elif portrait_all:
            res1 = transposed(self.downstream_head1(tok1, (W, H)))
            res2 = transposed(self.downstream_head2(tok2, (W, H)))

        else:
            is_portrait = ~is_landscape
            
            l_tok1 = [tok[is_landscape] if _idx in self.hooks_idx else None for _idx, tok in enumerate(tok1)]
            l_tok2 = [tok[is_landscape] if _idx in self.hooks_idx else None for _idx, tok in enumerate(tok2)]
            p_tok1 = [tok[is_portrait] if _idx in self.hooks_idx else None for _idx, tok in enumerate(tok1)]
            p_tok2 = [tok[is_portrait] if _idx in self.hooks_idx else None for _idx, tok in enumerate(tok2)]
            
            l_res1 = self.downstream_head1(l_tok1, (H, W))
            l_res2 = self.downstream_head2(l_tok2, (H, W))
            del l_tok1, l_tok2
            
            p_res1 = transposed(self.downstream_head1(p_tok1, (W, H)))
            p_res2 = transposed(self.downstream_head2(p_tok2, (W, H)))
            del p_tok1, p_tok2

            # allocate full result
            res1 = {}
            for k in l_res1 | p_res1:
                x = l_res1[k].new(is_landscape.size(0), *l_res1[k].shape[1:])
                
                x[is_landscape] = l_res1[k]
                x[is_portrait] = p_res1[k]
                res1[k] = x
            del l_res1, p_res1
            
            res2 = {}
            for k in l_res2 | p_res2:
                x = l_res2[k].new(is_landscape.size(0), *l_res2[k].shape[1:])
                
                x[is_landscape] = l_res2[k]
                x[is_portrait] = p_res2[k]
                res2[k] = x
            del l_res2, p_res2
        
        return res1, res2
    
    def _organize_output(self, res1, res2, idx, rearrange_fn):
        idxi, idxj = rearrange_fn.triu_tokens(idx), rearrange_fn.tril_tokens(idx)
        
        pts3d = rearrange_fn.unflatten_tokens(res1.pop('pts3d'))
        conf = rearrange_fn.unflatten_tokens(res1.pop('conf'))
        pts3d_ij, pts3d_ji = rearrange_fn.triu_tokens(pts3d), rearrange_fn.tril_tokens(pts3d)
        conf_ij, conf_ji = rearrange_fn.triu_tokens(conf), rearrange_fn.tril_tokens(conf)
        
        res1['pts3d_ij'] = pts3d_ij # pts3d of i in i's coordinate system (i does cross-attention to j)
        res1['pts3d_ji'] = pts3d_ji # pts3d of j in j's coordinate system (j does cross-attention to i)
        res1['conf_ij'] = conf_ij
        res1['conf_ji'] = conf_ji
        
        pts3d_in_other_view = rearrange_fn.unflatten_tokens(res2.pop('pts3d'))
        conf2 = rearrange_fn.unflatten_tokens(res2.pop('conf'))
        pts3d_in_other_view_ij, pts3d_in_other_view_ji = rearrange_fn.triu_tokens(pts3d_in_other_view), rearrange_fn.tril_tokens(pts3d_in_other_view)
        conf2_ij, conf2_ji = rearrange_fn.triu_tokens(conf2), rearrange_fn.tril_tokens(conf2)
        
        res2['pts3d_in_other_view_ij'] = pts3d_in_other_view_ij # pts3d of i in j's coordinate system (i does cross-attention to j)
        res2['pts3d_in_other_view_ji'] = pts3d_in_other_view_ji # pts3d of j in i's coordinate system (j does cross-attention to i)
        res2['conf_ij'] = conf2_ij
        res2['conf_ji'] = conf2_ji
        
        return res1, res2, idxi, idxj
    
    def forward(self, frames, time_embedding, is_landscape, landscape_all=False, portrait_all=False):
        # 'Exactly one of landscape_all or portrait_all must be True or both False'
        assert not(portrait_all and landscape_all), 'Exactly one of landscape_all or portrait_all must be True or both False'
        
        B, T, C, H, W = frames.shape
        device = frames.device
        
        rearrange_fn = token_rearrange_fn(frames)

        x = rearrange(frames, 'b t c h w -> (b t) c h w')
        time_embedding = rearrange(time_embedding, 'b t -> (b t) 1')
        is_landscape = rearrange(is_landscape, 'b t -> (b t)')
        idx = rearrange_fn.repeat_tokens(torch.arange(T, device=device).repeat(B))
        
        x, pos = self._patchfy(x, time_embedding, is_landscape, landscape_all, portrait_all)
        N = x.size(1)

        # _b_idx = torch.arange(B*T*N, device='cuda').reshape(B*T*N, 1, 1, 1).expand(-1, -1, T-1, T-1)
        # _q_idx = torch.arange(T-1, device='cuda').reshape(1, 1, T-1, 1).expand(B*T*N, -1, -1, T-1)
        # _kv_idx = torch.arange(T-1, device='cuda').reshape(1, 1, 1, T-1).expand(B*T*N, -1, T-1, -1)
        
        # batch_mask = ((_kv_idx + 1) <= ((_b_idx % (T * N)) // N))
        # qkv_mask = (_q_idx >= _kv_idx)
        
        # attn_mask = torch.where(torch.logical_or(batch_mask, qkv_mask), torch.zeros_like(_q_idx), torch.ones_like(_q_idx) * -float("inf"))
        
        _b_idx = torch.arange(T*N, device=device).reshape(T*N, 1, 1, 1).expand(-1, -1, T-1, T-1)
        _q_idx = torch.arange(T-1, device=device).reshape(1, 1, T-1, 1).expand(T*N, -1, -1, T-1)
        _kv_idx = torch.arange(T-1, device=device).reshape(1, 1, 1, T-1).expand(T*N, -1, T-1, -1)
        
        batch_mask = (_kv_idx + 1) <= (_b_idx // N)
        qkv_mask = (_q_idx >= _kv_idx)
        attn_mask = torch.logical_or(batch_mask, qkv_mask).repeat(B, 1, 1, 1)
        
        x, x_embed = self._encode_path(x, pos)
        dec1, dec2 = self._decode_path(x, x_embed, pos, rearrange_fn, attn_mask)
        is_landscape = rearrange_fn.flatten_tokens(rearrange_fn.repeat_tokens(is_landscape))
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            tok1 = [rearrange_fn.flatten_tokens(tok).float() if _idx in self.hooks_idx else None for _idx, tok in enumerate(dec1)]
            tok2 = [rearrange_fn.flatten_tokens(tok).float() if _idx in self.hooks_idx else None for _idx, tok in enumerate(dec2)]
            res1, res2 = self._head_path(tok1, tok2, (H, W), is_landscape, landscape_all, portrait_all)
            
        outputs = self._organize_output(res1, res2, idx, rearrange_fn)
        del rearrange_fn
        
        return outputs