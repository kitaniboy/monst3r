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

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed

from dust3r.blocks import EncoderBlock, DecoderBlock, DecoderBlockFixed
from dust3r.curope3d import RoPE as RoPE3D

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

class AsymmetricCroCo3DStereo (
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
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def _encode_patch(self, x, time, true_shape):
        x, pos = self.patch_embed(x, true_shape=true_shape)
        N = x.size(1)
        
        time = repeat(time, 'b 1 -> b n 1', n=N)
        pos = torch.cat([pos, time], dim=-1)
        
        return x, pos

    def forward(self, frames, time_embedding, true_shape, idx=None):
        B, T = frames.shape[:2]
        if idx is None:
            idx = torch.arange(T, device=frames.device)[None].expand(B, -1)
            
        x = rearrange(frames, 'b t c h w -> (b t) c h w')
        time_embedding = rearrange(time_embedding, 'b t -> (b t) 1')
        true_shape = rearrange(true_shape, 'b t two -> (b t) two')
        idx = rearrange(idx, 'b t -> (b t)')
        
        x, pos = self._encode_patch(x, time_embedding, true_shape)
        N = x.size(1)

        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        x_embed = self.decoder_embed(x)
        
        _repeat_index = repeat(torch.arange(T, device=x.device), 't -> t tm1', tm1=T-1)
        def repeat_tokens(tokens):
            tokens = rearrange(tokens, '(b t) ... -> b t ...', t=T)
            return tokens[:, _repeat_index] # b t tm1 n d

        _row_index_swap = repeat(torch.arange(T, device=x.device), 't1 -> (t2 t1)', t1=T, t2=T)[1:].view(T-1, T+1)[:,:-1].reshape(T, T-1)
        _col_index_swap = repeat(torch.arange(T-1, device=x.device), 'tm1 -> tm1 t', tm1=T-1, t=T).reshape(T, T-1)
        def swap_tokens(tokens):
            return tokens[:, _row_index_swap, _col_index_swap]
        
        # time_embedding2 = time_embedding1 = repeat_tokens(time_embedding)
        # time_embedding1 = time_embedding1
        # time_embedding2 = swap_tokens(time_embedding2)
        # test_time_embedding = torch.stack([time_embedding1, time_embedding2], dim=-1)
        
        f2 = f1 = repeat_tokens(x)
        pos2 = pos1 = repeat_tokens(pos)
        pos2_swap, pos1_swap = swap_tokens(pos2), swap_tokens(pos1)
        final_output = [(f1, f2)]
        
        f2 = f1 = repeat_tokens(x_embed)
        final_output.append((f1, f2))
        
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            x, y = final_output[-1][::+1]
            # img1 side
            f1 = blk1(x, swap_tokens(y), pos1, pos2_swap)
            # img2 side
            f2 = blk2(y, swap_tokens(x), pos2, pos1_swap)
            
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        dec1, dec2 = zip(*final_output)
        
        shape = repeat_tokens(true_shape)
        
        def flatten_tokens(tokens):
            return rearrange(tokens, 'b t tm1 ... -> (b t tm1) ...', t=T, tm1=T-1)
        
        def unflatten_tokens(tokens):
            return rearrange(tokens, '(b t tm1) ... -> b t tm1 ...', b=B, t=T, tm1=T-1)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            res1 = self._downstream_head(1, [flatten_tokens(tok).float() for tok in dec1], flatten_tokens(shape))
            res2 = self._downstream_head(2, [flatten_tokens(tok).float() for tok in dec2], flatten_tokens(shape))

        _row_index_triu, _col_index_triu = torch.triu_indices(T, T-1, device=x.device).unbind(0)
        def triu_tokens(tokens):
            return tokens[:, _row_index_triu, _col_index_triu]

        _row_index_tril, _col_index_tril = _col_index_triu+1, _row_index_triu.clone()
        def tril_tokens(tokens):
            return tokens[:, _row_index_tril, _col_index_tril]
        
        idx = repeat_tokens(idx)
        idxi, idxj = triu_tokens(idx), tril_tokens(idx)
        
        pts3d = unflatten_tokens(res1.pop('pts3d'))
        conf = unflatten_tokens(res1.pop('conf'))
        pts3d_ij, pts3d_ji = triu_tokens(pts3d), tril_tokens(pts3d)
        conf_ij, conf_ji = triu_tokens(conf), tril_tokens(conf)
        
        res1['pts3d_ij'] = pts3d_ij # pts3d of i in i's coordinate system (i does cross-attention to j)
        res1['pts3d_ji'] = pts3d_ji # pts3d of j in j's coordinate system (j does cross-attention to i)
        res1['conf_ij'] = conf_ij
        res1['conf_ji'] = conf_ji
        
        pts3d_in_other_view = unflatten_tokens(res2.pop('pts3d'))
        conf2 = unflatten_tokens(res2.pop('conf'))
        pts3d_in_other_view_ij, pts3d_in_other_view_ji = triu_tokens(pts3d_in_other_view), tril_tokens(pts3d_in_other_view)
        conf2_ij, conf2_ji = triu_tokens(conf2), tril_tokens(conf2)
        
        res2['pts3d_in_other_view_ij'] = pts3d_in_other_view_ij # pts3d of i in j's coordinate system (i does cross-attention to j)
        res2['pts3d_in_other_view_ji'] = pts3d_in_other_view_ji # pts3d of j in i's coordinate system (j does cross-attention to i)
        res2['conf_ij'] = conf2_ij
        res2['conf_ji'] = conf2_ji
        
        return res1, res2, idxi, idxj