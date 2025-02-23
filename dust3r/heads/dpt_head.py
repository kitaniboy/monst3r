# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.heads.dpt_block import DPTOutputAdapter  # noqa


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768, upsample_ratio=4):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]
        
        if upsample_ratio == 2:
            act_1_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.layer_dims[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=2, stride=2, padding=0,
                    bias=True, dilation=1, groups=1,
                )
            )

            act_2_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[1],
                    out_channels=self.layer_dims[1],
                    kernel_size=1, stride=1, padding=0,
                )
            )

            act_3_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[2],
                    out_channels=self.layer_dims[2],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.Conv2d(
                    in_channels=self.layer_dims[2],
                    out_channels=self.layer_dims[2],
                    kernel_size=3, stride=2, padding=1,
                )
            )

            act_4_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.Conv2d(
                    in_channels=self.layer_dims[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=3, stride=4, padding=1,
                )
            )
            
        elif upsample_ratio == 4:
            act_1_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.layer_dims[0],
                    out_channels=self.layer_dims[0],
                    kernel_size=4, stride=4, padding=0,
                    bias=True, dilation=1, groups=1,
                )
            )

            act_2_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[1],
                    out_channels=self.layer_dims[1],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.layer_dims[1],
                    out_channels=self.layer_dims[1],
                    kernel_size=2, stride=2, padding=0,
                    bias=True, dilation=1, groups=1,
                )
            )

            act_3_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[2],
                    out_channels=self.layer_dims[2],
                    kernel_size=1, stride=1, padding=0,
                )
            )

            act_4_postprocess = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dim_tokens_enc[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=1, stride=1, padding=0,
                ),
                nn.Conv2d(
                    in_channels=self.layer_dims[3],
                    out_channels=self.layer_dims[3],
                    kernel_size=3, stride=2, padding=1,
                )
            )

        self.act_postprocess = nn.ModuleList([
            act_1_postprocess,
            act_2_postprocess,
            act_3_postprocess,
            act_4_postprocess
        ])
        
    def forward(self, encoder_tokens: List[torch.Tensor], image_size):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        # image_size = self.image_size if image_size is None else image_size
        
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]
        
        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
        
        # # Project layers to chosen feature dim
        # layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # # Fuse layers using refinement stages
        # path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        # path_3 = self.scratch.refinenet3(path_4, layers[2])
        # path_2 = self.scratch.refinenet2(path_3, layers[1])
        # path_1 = self.scratch.refinenet1(path_2, layers[0])
        scratch_layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        penultimate = self.scratch(scratch_layers)
        # Output head
        out = self.head(penultimate)
        
        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        
        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)

        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        
        if dpt_args.get('patch_size', 16) == 8:
            dpt_init_args['upsample_ratio'] = 2
        elif dpt_args.get('patch_size', 16) == 16:
            dpt_init_args['upsample_ratio'] = 4
        
        self.dpt.init(**dpt_init_args)

    @torch.compiler.disable()
    def forward(self, x, img_info):
        out = self.dpt(x, image_size=img_info)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')

# def create_gaussian_head(net, has_conf=False):
#     """
#     return PixelwiseTaskWithDPT for given net params
#     """
#     assert net.dec_depth > 9
#     l2 = net.dec_depth
#     feature_dim = 256
#     last_dim = feature_dim//2
    
#     sh_degree = 4
#     d_sh = ((sh_degree + 1) ** 2) * 3
#     d_scales = 3
#     d_rotations = 4
#     d_density = 1
#     d_distance = 1

#     # d_dynamic = 1

#     out_nchan = d_sh + d_scales + d_rotations + d_density + d_distance
#     # out_nchan = last_dim//2
    
#     ed = net.enc_embed_dim
#     dd = net.dec_embed_dim

#     # print("hooks_idx")
#     # print([0, l2*2//4, l2*3//4, l2])
#     # print("hooks_idx")
#     # print([0, l2*2//4, l2*3//4, l2])
#     # print("hooks_idx")
#     # print([0, l2*2//4, l2*3//4, l2])
#     # print("hooks_idx")
#     # print([0, l2*2//4, l2*3//4, l2])

#     return PixelwiseTaskWithDPT(patch_size=8,
#                                 num_channels=out_nchan + has_conf,
#                                 feature_dim=feature_dim,
#                                 last_dim=last_dim,
#                                 hooks_idx=[0, l2*2//4, l2*3//4, l2],
#                                 dim_tokens=[dd, dd, dd, dd],
#                                 postprocess=lambda x, depth_mode, conf_mode: {'feat': x.permute(0, 2, 3, 1)},
#                                 depth_mode=None,
#                                 conf_mode=None,
#                                 head_type='regression')