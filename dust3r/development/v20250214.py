import os
# os.environ['OMP_NUM_THREADS'] = '4' # will affect the performance of pairwise prediction
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
import torchvision.transforms.v2.functional as tvf
from functools import partial

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True

from dust3r.model_multiimage2 import AsymmetricCroCo3DMultiImage2  # noqa: F401, needed when loading the model
from dust3r.model_singleimage2 import AsymmetricCroCo3DSingleImage2  # noqa: F401, needed when loading the model

from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.development.common import *
from dust3r.development.v20250214_pose_eval import eval_pose_estimation

def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pose_mode', default="GlobalAlignerMode.PointCloudOptimizerWithSolve", type=str)
    
    parser.add_argument('--croco_checkpoint', default="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth", type=str)
    parser.add_argument('--raft_checkpoint', default="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth", type=str)
    parser.add_argument('--sam2_checkpoint', default="third_party/sam2/checkpoints/sam2.1_hiera_large.pt", type=str)
    parser.add_argument('--sam2_model_cfg', default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    
    parser.add_argument('--train_criterion', default="", type=str, help="train criterion")
    parser.add_argument('--test_criterion', default="", type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default='', type=str, help="training set")
    parser.add_argument('--test_dataset', default='', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--compile", action='store_true', default=False,
                        help="compile model")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--empty_cache", action='store_true', default=False,
                        help="empty cache when input size is big")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')

    # others
    parser.add_argument('--multiprocessing_data', action='store_true', default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--dist_url', default='tcp://localhost:29831', help='url used to set up distributed training')
    parser.add_argument('--print_all_proc', type=int, default=1, choices=[0, 1], help="Print from all processes")
    
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=1, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')
    parser.add_argument('--experiment_name', type=str, help='wandb experiment group')
    parser.add_argument('--experiment_group', type=str, help='wandb experiment group')
    
    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str, help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False, help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float, help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False, help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False, help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')
    
    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False, help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--motion_mask_thre', default=0.35, type=float, help='motion mask threshold for pose optimization')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False, help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float, help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument('--eval_dataset', type=str, default='sintel', 
                    choices=['davis', 'kitti', 'bonn', 'scannet', 'tum', 'nyu', 'sintel'], 
                    help='choose dataset for pose evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False, help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser

def load_model(args, device):
    # model
    inf = float('inf')
    
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    model.to(device)
    model_without_ddp = model
    if args.croco_checkpoint and not args.resume:
        print('Loading pretrained: ', args.croco_checkpoint)
        ckpt = torch.load(args.croco_checkpoint, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        
        if len(missing):
            print('... missing keys:', missing)
        
        if len(unexpected):
            print('... unexpected keys:', unexpected)
            
        del ckpt  # in case it occupies memory
        
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    return model, model_without_ddp

def train(args, input_queue, output_queue):
    misc.init_distributed_mode(args)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # if main process, init wandb
    if args.wandb and misc.is_main_process():
        wandb_dir = os.path.join(args.output_dir, 'wandb')
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb.init(group=args.experiment_group,
                   name=args.experiment_name, 
                   project='multi-image-2', 
                   config=args, 
                   dir=wandb_dir)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        if os.path.isfile(last_ckpt_fname) and (not args.eval_only): args.resume = last_ckpt_fname

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark
    model, model_without_ddp = load_model(args, device)
    
    if args.compile:
        print('Compiling model...')
        model = torch.compile(model)
    
    # training dataset and loader
    #  dataset and loader
    data_loader_train = None
    if len(args.train_dataset):
        print('Building train dataset {:s}'.format(args.train_dataset))
        data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    
    data_loader_test = {}
    if len(args.test_dataset):
        for dataset in args.test_dataset.split('+'):
            print('Building test dataset {:s}'.format(dataset))
            
            testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
            name_testset = dataset.split('(')[0]
            if getattr(testset.dataset.dataset, 'strides', None) is not None:
                name_testset += f'_stride{testset.dataset.dataset.strides}'
            data_loader_test[name_testset] = testset

    train_criterion = None
    if len(args.train_criterion):
        print(f'>> Creating train criterion = {args.train_criterion}')
        train_criterion = eval(args.train_criterion).to(device)
        
    test_criterion = None
    if len(args.test_criterion) or len(args.train_criterion):
        print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
        test_criterion = eval(args.test_criterion or args.train_criterion).to(device)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    args.print_freq = int(args.print_freq * args.accum_iter)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount(bool(args.amp))

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if args.wandb and misc.is_main_process():
                    wandb_step = int(epoch * (len(data_loader_train) // args.accum_iter) - 1)
                    # print(f"test wandb step: {wandb_step}")

                    log_dict = {f'{test_name}/' + k: v for k, v in test_stats[test_name].items()}
                    wandb.log(log_dict, step=wandb_step)

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb and misc.is_main_process():
                            wandb_step = int(epoch * (len(data_loader_train) // args.accum_iter) - 1)
                            # print(f"test wandb step: {wandb_step}")

                            log_dict = {f'{original_test_name}_stride_mean/' + f'{k}': v for k, v in stride_stats.items()}
                            wandb.log(log_dict, step=wandb_step)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    best_so_far, best_pose_ate_sofar = resume_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if best_pose_ate_sofar is None:
        best_pose_ate_sofar = float('inf')
        
    # if global_rank == 0 and args.output_dir is not None:
    #     log_writer = SummaryWriter(log_dir=args.output_dir)
    # else:
    #     log_writer = None
    log_writer = None # we only use wandb as logger

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = {}
    test_stats = {}
    
    save_model_fn = partial(save_model,
                            args=args,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler)
    for epoch in range(args.start_epoch, args.epochs + 1):
        # Test on multiple datasets
        new_best = False
        new_pose_best = False
        
        if (epoch > args.start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only:
            # if epoch > args.start_epoch:
            #     if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs and not already_saved:
            #         save_model(epoch - 1, 'before_eval')
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, model_without_ddp, test_criterion, testset, device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['total_loss/med'] < best_so_far:
                    best_so_far = stats['total_loss/med']
                    new_best = True

            # Ensure that eval_pose_estimation is only run on the main process
            if args.pose_eval_freq>0 and (epoch % args.pose_eval_freq==0 or args.eval_only):            
                if args.eval_only:
                    pose_save_dir = args.output_dir
                else:
                    pose_save_dir = f'{args.output_dir}/{epoch}'

                ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation(args, model, device, save_dir=pose_save_dir)
                print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')

                if args.wandb and misc.is_main_process():
                    wandb_dict = {
                        'pose/ATE': ate_mean,
                        'pose/RPE trans': rpe_trans_mean,
                        'pose/RPE rot': rpe_rot_mean,
                    }
                    if args.save_pose_qualitative:
                        for outfile in outfile_list:
                            wandb_dict[outfile.split('/')[-1]] = wandb.Object3D(open(outfile))
                    
                    wandb_step = int(epoch * (len(data_loader_train) // args.accum_iter) - 1)
                    wandb.log(wandb_dict, step=wandb_step)

                if ate_mean < best_pose_ate_sofar and not bug: # if the pose estimation is better, and w/o any error
                    best_pose_ate_sofar = ate_mean
                    new_pose_best = True

            # Synchronize all processes to ensure eval_pose_estimation is completed
            try:
                torch.distributed.barrier()
            except:
                pass

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if args.eval_only and args.epochs <= 1:
            exit(0)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model_fn(epoch=epoch-1, fname=str(epoch), best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)
                
            if new_best:
                save_model_fn(epoch=epoch-1, fname='best', best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)
                
            if new_pose_best and args.save_best_pose:
                save_model_fn(epoch=epoch-1, fname='best_pose', best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)
                

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model_fn(epoch=epoch-1, fname='last', best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)

        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        # print(train_criterion)
        train_stats = train_one_epoch(
            model, model_without_ddp, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)

def loss_of_one_batch(batch, labels, model, criterion, use_amp=False, step=0):    
    # print(f'forward start {step}', datetime.datetime.now())
    with torch.amp.autocast(device_type='cuda', enabled=bool(use_amp)):
        pred1, pred2, idxi, idxj = model(**batch)
    
    pts3d = labels['pts3d'] # assume world coordinate system
    valid_mask = labels['valid_mask']
    camera_pose = labels['camera_pose'] # means camera-to-world
    camera_intrinsics = labels['camera_intrinsics'] # means camera-to-pixel
    
    idx_this_view = torch.cat([idxi, idxj], dim=1)
    idx_other_view = torch.cat([idxj, idxi], dim=1)
    
    batch_size, num_views = pts3d.shape[:2]
    batch_idx = torch.arange(batch_size, device=pts3d.device)[:, None].repeat(1, idx_this_view.shape[1])
    
    pred1 = {
        'pts3d': torch.cat([pred1['pts3d_ij'], pred1['pts3d_ji']], dim=1),
        'conf': torch.cat([pred1['conf_ij'], pred1['conf_ji']], dim=1),
    }
    
    pred2 = {
        'pts3d_in_other_view': torch.cat([pred2['pts3d_in_other_view_ji'], pred2['pts3d_in_other_view_ij']], dim=1),
        'conf': torch.cat([pred2['conf_ji'], pred2['conf_ij']], dim=1),
    }
    
    with torch.no_grad():
        gt1 = {
            'pts3d': pts3d[batch_idx, idx_this_view],
            'valid_mask': valid_mask[batch_idx, idx_this_view],
            'camera_pose': camera_pose[batch_idx, idx_this_view],
            'camera_intrinsics': camera_intrinsics[batch_idx, idx_this_view]
        }
        
        gt2 = {
            'pts3d': pts3d[batch_idx, idx_other_view],
            'valid_mask': valid_mask[batch_idx, idx_other_view],
            'camera_pose': camera_pose[batch_idx, idx_this_view],
            'camera_intrinsics': camera_intrinsics[batch_idx, idx_this_view]
        }
    
    # print(f'loss start {step}', datetime.datetime.now())
    with torch.amp.autocast(device_type='cuda', enabled=False):
        loss = criterion(gt1, gt2, pred1, pred2) if criterion is not None else None
    
    result = dict(gt1=gt1, gt2=gt2, pred1=pred1, pred2=pred2, loss=loss)
    
    return result
    
def train_one_epoch(croco: torch.nn.Module, model_without_ddp: torch.nn.Module, criterion: torch.nn.Module, data_loader: Sized, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, args, log_writer=None):
    
    croco.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('epoch', misc.SmoothedValue(fmt='{value:.2f}'))
        
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    if args.empty_cache: torch.cuda.empty_cache()
    
    if len(data_loader) % args.accum_iter != 0:
        print('Warning: the number of iterations is not divisible by the accumulation factor')
    
    for data_iter_step, (views, extra_info) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        wandb_step = int(epoch * (len(data_loader) // accum_iter) + data_iter_step // accum_iter)
        
        frames_context = torch.stack([v['frames_context'] for v in views]).to(device, non_blocking=True) 
        time_embedding_context = torch.stack([v['time_embedding_context'] for v in views]).to(device, non_blocking=True) 

        pts3d_context = torch.stack([v['pts3d_context'] for v in views]).to(device, non_blocking=True) 
        valid_mask_context = torch.stack([v['valid_mask_context'] for v in views]).to(device, non_blocking=True) 
        camera_pose_context = torch.stack([v['camera_pose_context'] for v in views]).to(device, non_blocking=True) 
        camera_intrinsics_context = torch.stack([v['camera_intrinsics_context'] for v in views]).to(device, non_blocking=True) 

        batch_size, num_views, C, H, W = frames_context.shape
        
        if ((H * W / 256) > 680) and args.empty_cache:
            torch.cuda.empty_cache()
        
        true_shape = torch.stack([v['true_shape_context'] for v in views])
        
        is_landscape = (true_shape[..., 1] >= true_shape[..., 0])
        is_portrait = ~is_landscape
        
        landscape_all = False
        portrait_all = False
        if is_landscape.all():
            landscape_all = True
        elif is_portrait.all():
            portrait_all = True
        else:
            print('mixed landscape and portrait')
        
        is_landscape = is_landscape.to(device, non_blocking=True)
        
        batch = {
            'frames': frames_context,
            'time_embedding': time_embedding_context,
            'landscape_all': landscape_all,
            'portrait_all': portrait_all,
            'is_landscape': is_landscape
        }
        
        labels = {
            'pts3d': pts3d_context,
            'valid_mask': valid_mask_context,
            'camera_pose': camera_pose_context,
            'camera_intrinsics': camera_intrinsics_context,
        }
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        
    
        # def slice_time(batch, start, end):
        #     return {k: v[:, start:end] for k, v in batch.items()}
        
        batch_result = loss_of_one_batch(batch, labels, croco, criterion, use_amp=bool(args.amp), step=data_iter_step)
    
        loss, loss_details = batch_result['loss']  # criterion returns two values
        loss_value = float(loss)

        loss /= accum_iter
        
        # print(f'backward start {data_iter_step}', datetime.datetime.now())
        loss_scaler(loss, optimizer, parameters=croco.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)
        
        metric_logger.update(epoch=epoch_f)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)                
        metric_logger.update(**loss_details)
        
        # send logs to wandb
        with torch.no_grad():
            if args.wandb and ((data_iter_step + 1) % args.print_freq == 0):
                loss_value_reduce = {
                    k: misc.all_reduce_mean(v) for k, v in loss_details.items()
                }
                
                if misc.is_main_process():
                    wandb_scalars = {
                        'info/lr': lr,
                        'info/epoch': epoch_f,
                    }

                    for k, v in loss_value_reduce.items():
                        wandb_scalars[f'train/{k}'] = v
                    
                    parameters = model_without_ddp.log_parameters()
                    for k, v in parameters.items():
                        vmax = v.max().item()
                        vmin = v.min().item()
                        vabsmed = v.abs().median().item()
                        wandb_scalars[f'params-max/{k}'] = vmax
                        wandb_scalars[f'params-min/{k}'] = vmin
                        wandb_scalars[f'params-absmed/{k}'] = vabsmed
                        
                    # print(f"wandb step: {wandb_step}")
                    wandb.log(wandb_scalars, step=wandb_step)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    results = {f'{k}/med': getattr(meter, 'median') for k, meter in metric_logger.meters.items()}

    return results

def test_one_epoch(croco: torch.nn.Module, model_without_ddp: torch.nn.Module, criterion: torch.nn.Module, data_loader: Sized, device: torch.device, epoch: int, args, log_writer=None, prefix='test'):

    croco.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch) if not args.fixed_eval_set else data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) if not args.fixed_eval_set else data_loader.sampler.set_epoch(0)

    for data_iter_step, (views, extra_info) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        frames_context = torch.stack([v['frames_context'] for v in views]).to(device, non_blocking=True) 
        time_embedding_context = torch.stack([v['time_embedding_context'] for v in views]).to(device, non_blocking=True) 

        pts3d_context = torch.stack([v['pts3d_context'] for v in views]).to(device, non_blocking=True) 
        valid_mask_context = torch.stack([v['valid_mask_context'] for v in views]).to(device, non_blocking=True) 
        camera_pose_context = torch.stack([v['camera_pose_context'] for v in views]).to(device, non_blocking=True) 
        camera_intrinsics_context = torch.stack([v['camera_intrinsics_context'] for v in views]).to(device, non_blocking=True) 
        
        batch_size, num_views = frames_context.shape[:2]
        
        true_shape = torch.stack([v['true_shape_context'] for v in views])
        
        is_landscape = (true_shape[..., 1] >= true_shape[..., 0])
        is_portrait = ~is_landscape
        
        landscape_all = False
        portrait_all = False
        if is_landscape.all():
            landscape_all = True
        elif is_portrait.all():
            portrait_all = True
        else:
            print('mixed landscape and portrait')
        
        is_landscape = is_landscape.to(device, non_blocking=True)
        
        batch = {
            'frames': frames_context,
            'time_embedding': time_embedding_context,
            'landscape_all': landscape_all,
            'portrait_all': portrait_all,
            'is_landscape': is_landscape
        }
        
        labels = {
            'pts3d': pts3d_context,
            'valid_mask': valid_mask_context,
            'camera_pose': camera_pose_context,
            'camera_intrinsics': camera_intrinsics_context,
        }
        
        def slice_time(batch, start, end):
            return {k: v[:, start:end] for k, v in batch.items()}
        
        with torch.no_grad():        
            batch_result = loss_of_one_batch(batch, labels, croco, criterion, use_amp=bool(args.amp), step=data_iter_step)
            
            loss, loss_details = batch_result['loss']
            metric_logger.update(**loss_details)

        # if args.num_save_visual>0 and (data_iter_step % max((len(data_loader) // args.num_save_visual), 1) == 0) and misc.is_main_process() : # save visualizations
        #     save_dir = f'{args.output_dir}/{epoch}/eval'
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     # save_visualizations(merged_batch, batch_result, views, save_dir, data_iter_step)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}/{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    return results

def draw_heatmap(heatmap):
    max_val = heatmap.max()
    min_val = heatmap.min()
    distance = max_val - min_val
    
    heatmap = repeat((heatmap - min_val) / distance, '... h w -> ... c h w', c=3)
    # heatmap_image = hcat(*heatmap)
    
    return heatmap

def save_visualizations(gpu_batch, result, views, save_dir, data_iter_step):
    batch_size = len(views)
    
    color_inputs = gpu_batch['rgb_context']
    gt_color = gpu_batch['rgb_target']

    pred = result['pred']
    
    pred_color = pred['color']
    pred_density = pred['density']
    pred_distance = pred['distance']
    
    comparison = hcat(
        vcat(*gt_color),
        vcat(*pred_color),
    )
    image = prep_image(add_border(comparison))
    
    # save_name = f'{extra_info[0]["dataset"]}_{extra_info[0]["label"]}_pv_{extra_info[0]["primary_view"]}_tv_{"_".join(extra_info[0]["target_view"])}_image'
    image_save_path = f'{save_dir}/{data_iter_step:06d}_rendered_image.jpg'
    iio.imwrite(image_save_path, image)
    
    density_image = []
    for b_idx in range(batch_size):
        pred_density_i = pred_density[b_idx]
        density_image_i = hcat(*draw_heatmap(pred_density_i))
        density_image.append(density_image_i)
    
    density_image = vcat(*density_image)
    density_image = prep_image(add_border(density_image))
    
    image_save_path = f'{save_dir}/{data_iter_step:06d}_density.jpg'
    iio.imwrite(image_save_path, density_image)
    
    distance_image = []
    for b_idx in range(batch_size):
        pred_distance_i = pred_distance[b_idx]
        distance_image_i = hcat(*draw_heatmap(pred_distance_i))
        distance_image.append(distance_image_i)
    
    distance_image = vcat(*distance_image)
    distance_image = prep_image(add_border(distance_image))
    
    image_save_path = f'{save_dir}/{data_iter_step:06d}_distance.jpg'
    iio.imwrite(image_save_path, distance_image)
    
    informations = []
    for b_idx in range(batch_size):
        dataset = views[b_idx]['dataset']
        label = views[b_idx]['label']
        instance = views[b_idx]['instance']
        tensor_index_reference = views[b_idx]['tensor_index_reference']
        tensor_index_target = views[b_idx]['tensor_index_target']
        
        information_dict = {
            'dataset': dataset,
            'label': label,
            'instance': instance,
            'tensor_index_reference': tensor_index_reference,
            'tensor_index_target': tensor_index_target
        }
        informations.append(information_dict)
    json_save_path = f'{save_dir}/{data_iter_step:06d}_info.json'
    with open(json_save_path, 'w') as f:
        json.dump(informations, f, indent=4)