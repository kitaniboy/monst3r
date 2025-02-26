# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes  # noqa
from .blendedmvs import BlendedMVS  # noqa
from .co3d import Co3d  # noqa
from .habitat import Habitat  # noqa
from .megadepth import MegaDepth  # noqa
from .scannetpp import ScanNetpp  # noqa
from .staticthings3d import StaticThings3D  # noqa
from .waymo import Waymo  # noqa
from .wildrgbd import WildRGBD  # noqa
from .pointodyssey import PointOdysseyDUSt3R  # noqa
from .sintel import SintelDUSt3R  # noqa
from .tartanair import TarTanAirDUSt3R  # noqa
from .spring_dataset import SpringDUSt3R  # noqa
from .dynamic_replica import DynamicReplicaDUSt3R  # noqa
from .hypernerf import Hypernerf  # noqa
from .iphone import Iphone  # noqa
from .re10k import Re10kDUSt3R  # noqa

from .pointodyssey_lmdb import PointOdysseyLMDB
from .tartanair_lmdb import TarTanAirLMDB
from .spring_dataset_lmdb import SpringLMDB

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset) 
        # dataset: "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)"
        # eval(dataset) returns Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn,
    )

    return data_loader
