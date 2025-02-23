# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# PatchEmbed implementation for DUST3R,
# in particular ManyAR_PatchEmbed that Handle images with non-square aspect ratio
# --------------------------------------------------------
import torch
import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import PatchEmbed  # noqa


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim):
    assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, 3, enc_embed_dim)
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, init='xavier'):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten, init)

    def forward(self, img, landscape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W
        
        if landscape:
            x = self.proj(img).permute(0, 2, 3, 1).flatten(1, 2).float()
            pos = self.position_getter(B, H, W, x.device)
        
        else:
            x = self.proj(img.swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()
            pos = self.position_getter(B, W, H, x.device)
        
        x = self.norm(x)
        
        return x, pos
