# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale

# from lpips import LPIPS

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean', dim=-1):
        super().__init__()
        self.reduction = reduction
        self.dim = dim

class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape, f'Bad shape = {a.shape} {b.shape}' #  and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=self.dim)  # normalized L2 distance

class MSELoss (LLoss):
    """ Mean Squared Error """

    def distance(self, a, b):
        return (a - b).pow(2).mean(dim=self.dim)  # mean squared error


L21 = L21Loss()
# MSE = MSELoss()

class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha
        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details

class ColorMSE (MultiLoss):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss(dim=1)
        
    def compute_loss(self, batch, pred):
        pred_color = pred['color']
        target = batch['rgb_target']
        
        mse = self.mse(pred_color, target.detach()).mean()
        
        return mse

    def get_name(self):
        return f'ColorMSE'

class LPIPSLoss (MultiLoss):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        
    def compute_loss(self, batch, pred):
        pred_color = pred['color']
        target = batch['rgb_target']
        
        lpips = self.lpips.forward(pred_color, target.detach(), normalize=True).mean()

        return lpips

    def get_name(self):
        return f'LPIPSLoss'

class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1, point clouds at world coord_frame
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1), I is identical matrix
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        with torch.no_grad():
            camera_pose1 = gt1['camera_pose']
            R1 = camera_pose1[:, :, :3, :3].reshape(-1, 3, 3)
            T1 = camera_pose1[:, :, :3, 3].reshape(-1, 3)
            
            pts1 = gt1['pts3d']
            H, W = pts1.shape[2:4]
            pts1 = pts1.reshape(-1, H, W, 3)
            
            pts1_is_nan = pts1.isnan().any(dim=-1)

            gt_pts1 = einsum(R1.transpose(-1, -2), (pts1 - T1[:, None, None]), 'b i j, b h w j -> b h w i')
            
            camera_pose2 = gt2['camera_pose']
            R2 = camera_pose2[:, :, :3, :3].reshape(-1, 3, 3)
            T2 = camera_pose2[:, :, :3, 3].reshape(-1, 3)

            pts2 = gt2['pts3d']
            pts2 = pts2.reshape(-1, H, W, 3)

            pts2_is_nan = pts2.isnan().any(dim=-1)

            gt_pts2 = einsum(R2.transpose(-1, -2), (pts2 - T2[:, None, None]), 'b i j, b h w j -> b h w i')
        
            valid1 = gt1['valid_mask'].reshape(-1, H, W).clone()
            valid2 = gt2['valid_mask'].reshape(-1, H, W).clone()

            if dist_clip is not None:
                # points that are too far-away == invalid
                dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
                dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
                valid1 = valid1 & (dis1 <= dist_clip)
                valid2 = valid2 & (dis2 <= dist_clip)

        # pr_pts1 = pred1['pts3d_context'].reshape(-1, H, W, 3)
        # pr_pts2 = pred2['pts3d_target_in_context_view'].reshape(-1, H, W, 3)
        pr_pts1 = pred1['pts3d'].reshape(-1, H, W, 3)
        pr_pts2 = pred2['pts3d_in_other_view'].reshape(-1, H, W, 3)
        
        pr_conf1 = pred1['conf'].reshape(-1, H, W)
        pr_conf2 = pred2['conf'].reshape(-1, H, W)
        
        # pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        # pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
            pr_pts1[~valid1] = 0.0 
            pr_pts2[~valid2] = 0.0
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)
            gt_pts1[~valid1] = 0.0
            gt_pts2[~valid2] = 0.0
        
        monitoring = {}

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, pr_conf1, pr_conf2, monitoring

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, pred_conf1, pred_conf2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1,
                            gt_pts1)
        # loss on gt2 side
        l2 = self.criterion(pred_pts2,
                            gt_pts2)

        self_name = type(self).__name__
        loss_names = [self_name + '_pts3d_1', self_name + '_pts3d_2']
        confidences = [pred_conf1, pred_conf2]
        masks = [mask1, mask2]
        losses = [l1, l2]

        with torch.no_grad():
            # only for printing
            mean_weight1 = mask1.float() / mask1.sum().clamp(min=1)
            mean_weight2 = mask2.float() / mask2.sum().clamp(min=1)
            losses_mean = [mean_weight1.flatten() @ l1.flatten(),
                           mean_weight2.flatten() @ l2.flatten()
            ]

        details = {}
        for loss_name, loss in zip(loss_names, losses_mean):
            details[loss_name] = loss
        
        details['loss_names'] = loss_names
        details['confidences'] = confidences

        return Sum(*zip(losses, masks)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        # ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        losses_and_masks, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)

        loss_names = details.pop('loss_names')
        confidences = details.pop('confidences')
        
        total_loss = 0.0
        for (loss, mask), name, conf in zip(losses_and_masks, loss_names, confidences):
            float_mask = mask.float()
            sum_mask = mask.sum()
            if conf is not None:
                conf, log_conf = self.get_conf_log(conf)

                loss = loss * conf - self.alpha * log_conf
                
                mean_weight = float_mask / sum_mask.clamp(min=1)
                loss = mean_weight.flatten() @ loss.flatten()
                
                with torch.no_grad():
                    conf_mean = mean_weight.flatten() @ conf.flatten()
                    details[f'conf_{name}'] = conf_mean
            else:
                mean_weight = float_mask / sum_mask.clamp(min=1)
                loss = mean_weight.flatten() @ loss.flatten()
            
            if sum_mask.item() == 0:
                print(f'NO VALID POINTS in {name}', force=True)

            total_loss = total_loss + loss
        details['total_loss'] = total_loss
        
        return total_loss, details

class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, pred_conf1, pred_conf2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, pred_conf1, pred_conf2, monitoring

class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, pred_conf1, pred_conf2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, pred_conf1, pred_conf2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass
    
class TestLoss (MultiLoss):
    def __init__(self, pixel_loss):
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'{self.pixel_loss}'
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        # ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        losses_and_masks, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)

        loss_names = details.pop('loss_names')
        confidences = details.pop('confidences')
        
        total_loss = 0.0
        for (loss, mask), name in zip(losses_and_masks, loss_names):
            float_mask = mask.float()
            sum_mask = mask.sum()

            mean_weight = float_mask / sum_mask.clamp(min=1)
            loss = mean_weight.flatten() @ loss.flatten()

            if sum_mask.item() == 0:
                print(f'NO VALID POINTS in {name}', force=True)

            total_loss = total_loss + loss
        details['total_loss'] = total_loss
        
        return total_loss, details
    

        # ###########################
        # gt_img1 = pred1['color_target']
        # b, c, h, w = gt_img1.shape
        # gt_img1 = gt_img1.permute(0, 2, 3, 1)

        # pred_color1 = pred1['color'].permute(0, 2, 3, 1)
        # color_loss = self.criterion(pred_color1, gt_img1)
        
        # self_name = type(self).__name__
        # loss_names.append('color_loss')
        # confidences.append(None)
        # masks.append(None)
        # losses.append(color_loss)
        # losses_mean.append(float(color_loss.mean()))
        # ###########################
        
        # pred_color_in_other_view1 = pred1['color_in_other_view'].permute(0, 2, 3, 1)
        # reduction = self.criterion.reduction

        # self.criterion.reduction = 'none'
        # with torch.no_grad():
        #     color_other_view_loss = self.criterion(pred_color_in_other_view1, gt_img1)
        # self.criterion.reduction = reduction

        # dy_conf = rearrange(pred1['rigidity'], '(b t) (h w) 1 -> (b t) h w', t=2, h=h, w=w)

        # dy_conf_loss = self.criterion(dy_conf[..., None], color_other_view_loss[..., None].detach())


        # self_name = type(self).__name__
        # loss_names = (self_name + '_pts3d_1', self_name + '_pts3d_2', 'color_loss', 'dy_conf_loss')
        # confidences = (pred1['conf'], pred2['conf'], None, None)
        # masks = (mask1, mask2, None, None)
        # losses = (l1, l2, color_loss, dy_conf_loss)
        # losses_mean = (float(l1.mean()), float(l2.mean()), float(color_loss.mean()), float(dy_conf_loss.mean()))


        ###########################
        # pred_color_in_other_view1 = pred1['color_in_other_view'].permute(0, 2, 3, 1)
        # color_other_view_loss = self.criterion(pred_color_in_other_view1, gt_img1)

        # rigidity1 = rearrange(pred1['rigidity'], '(b t) (h w) 1 -> (b t) h w', t=2, h=h, w=w)
        # dy_conf = rigidity1
        

        # self_name = type(self).__name__
        # loss_names = (self_name + '_pts3d_1', self_name + '_pts3d_2', 'color_loss', 'color_other_view_loss')
        # confidences = (pred1['conf'], pred2['conf'], None, dy_conf)
        # masks = (mask1, mask2, None, None)
        # losses = (l1, l2, color_loss, color_other_view_loss)
        # losses_mean = (float(l1.mean()), float(l2.mean()), float(color_loss.mean()), float(color_other_view_loss.mean()))
        ###########################