import os
import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Sized, Any, Generator, Iterable, Literal, Optional, Union
import imageio.v3 as iio

from einops import rearrange, repeat

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from dust3r.datasets import get_data_loader  # noqa
import dust3r.utils.misc as misc

def _sanitize_color(color):
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)

def add_border(
    image,
    border=8,
    color=1,
):
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result

def prep_image(image):
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def _get_main_dim(main_axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]
    
def _compute_offset(base: int, overlay: int, align) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)
    
def overlay(
    base,
    overlay,
    main_axis,
    main_axis_alignment,
    cross_axis_alignment,
):
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result

def _get_cross_dim(main_axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]

def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item

def cat(
    main_axis,
    *images,
    align = "center",
    gap: int = 8,
    gap_color = 1,
):
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))

def vcat(
    *images,
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )

def hcat(
    *images,
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )
    
def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor) and value1.ndim == value2.ndim:
            value = torch.stack((value1, value2), dim=2).flatten(0, 2) # e.g., 'b v 2 c h w -> (b v 2) c h w'
            
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res

def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def save_on_master(*args, **kwargs):
    if misc.is_main_process():
        torch.save(*args, **kwargs)

def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None, best_pose_ate_sofar=None):
    output_dir = Path(args.output_dir)
    if fname is None: fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if best_so_far is not None: to_save['best_so_far'] = best_so_far
    if best_pose_ate_sofar is not None: to_save['best_pose_ate_sofar'] = best_pose_ate_sofar
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)

def resume_model(args, model_without_ddp, optimizer, loss_scaler):
    args.start_epoch = 0
    best_so_far = None
    best_pose_ate_sofar = None
    if args.resume is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Resume checkpoint %s" % args.resume)
        missing, unexpected = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
        if len(missing):
            print('... missing keys:', missing)
        
        if len(unexpected):
            print('... unexpected keys:', unexpected)
        
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        if 'best_pose_ate_sofar' in checkpoint:
            best_pose_ate_sofar = checkpoint['best_pose_ate_sofar']
            print(" & best_pose_ate_sofar={:g}".format(best_pose_ate_sofar))
        else:
            best_pose_ate_sofar = None
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
    
    return best_so_far, best_pose_ate_sofar

def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader