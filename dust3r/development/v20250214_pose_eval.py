import os
os.environ['OMP_NUM_THREADS'] = '4' # will affect the performance of pairwise prediction

import math
import cv2
import numpy as np
import torch
from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
import croco.utils.misc as misc
import torch.distributed as dist
from tqdm import tqdm
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
# from dust3r.demo import get_3D_model_from_scene
import dust3r.eval_metadata
from dust3r.eval_metadata import dataset_metadata
from dust3r.utils.device import to_cpu, collate_with_cat

from third_party.raft import load_RAFT
from sam2.build_sam import build_sam2_video_predictor

from tqdm import tqdm, trange
import pickle as pkl

def eval_pose_estimation(args, model, device, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset, dataset_metadata['sintel'])
    img_path = metadata['img_path']
    mask_path = metadata['mask_path']

    ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation_dist(
        args, model, device, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug

def eval_pose_estimation_dist(args, model, device, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.eval_dataset, dataset_metadata['sintel'])
    anno_path = metadata.get('anno_path', None)

    silent = args.silent
    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get('full_seq', False):
            args.full_seq = True
        else:
            seq_list = metadata.get('seq_list', [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    # Split seq_list across processes
    if misc.is_dist_avail_and_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    total_seqs = len(seq_list)
    seqs_per_proc = (total_seqs + world_size - 1) // world_size  # Ceiling division

    start_idx = rank * seqs_per_proc
    end_idx = min(start_idx + seqs_per_proc, total_seqs)

    seq_list = seq_list[start_idx:end_idx]

    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []
    outfile_list = []
    load_img_size = 512

    error_log_path = f'{save_dir}/_error_log_{rank}.txt'  # Unique log file per process
    bug = False

    flow_net = load_RAFT(args.raft_checkpoint).to(device)
    flow_net.eval()
    
    sam2 = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    
    for seq in tqdm(seq_list):
        try:
            dir_path = metadata['dir_path_func'](img_path, seq)

            # Handle skip_condition
            skip_condition = metadata.get('skip_condition', None)
            if skip_condition is not None and skip_condition(save_dir, seq):
                continue

            mask_path_seq_func = metadata.get('mask_path_seq_func', lambda mask_path, seq: None)
            mask_path_seq = mask_path_seq_func(mask_path, seq)

            filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            filelist.sort()
            filelist = filelist[::args.pose_eval_stride]
            max_winsize = max(1, math.ceil((len(filelist)-1)/2))
            scene_graph_type = args.scene_graph_type
            if int(scene_graph_type.split('-')[1]) > max_winsize:
                scene_graph_type = f'{args.scene_graph_type.split("-")[0]}-{max_winsize}'
                if len(scene_graph_type.split("-")) > 2:
                    scene_graph_type += f'-{args.scene_graph_type.split("-")[2]}'
            imgs = load_images(
                filelist, size=load_img_size, verbose=False,
                dynamic_mask_root=mask_path_seq, crop=not args.no_crop
            )
            if args.eval_dataset == 'davis' and len(imgs) > 95:
                # use swinstride-4
                scene_graph_type = scene_graph_type.replace('5', '4')
                
            # _pairs, _pairsid = make_pairs(
            #     imgs, scene_graph=scene_graph_type, prefilter=None, symmetrize=True
            # )
            
            # output = inference(pairs, model, device, batch_size=1, verbose=not silent)

            stride = 2
            winsize = int(scene_graph_type.split('-')[1])
            start = 1
                        
            outputs = dict()
            
            list_chunk = []
            for i in trange(len(imgs)):
                chunk = [i]
                for j in range(start, stride*winsize + start, stride):
                    idx = (i + j)
                    if idx >= len(imgs):
                        continue
                    
                    chunk.append(idx)
                
                if len(chunk) < 2:
                    continue
                
                list_chunk.append(chunk)
                
            
            list_frames = []
            list_time_embedding = []
            list_true_shape = []
            list_idx = []
            for chunk in list_chunk:
                list_frames.append(torch.cat([imgs[v]['img'] for v in chunk], dim=0))
                list_time_embedding.append(torch.cat([torch.tensor([v], dtype=torch.long) for v in chunk], dim=0))
                list_true_shape.append(torch.cat([torch.tensor(imgs[v]['true_shape'], dtype=torch.long) for v in chunk], dim=0))
                list_idx.append(torch.cat([torch.tensor([v], dtype=torch.long) for v in chunk], dim=0))
            
            def stack_batch(list_tensor, offset, batch_size):
                return torch.stack(list_tensor[offset:offset+batch_size], dim=0)
            
            def flatten_outputs(res):
                if isinstance(res, dict):
                    return {k: flatten_outputs(v) for k, v in res.items()}

                if isinstance(res, (tuple, list)):
                    return type(res)([flatten_outputs(x) for x in res])

                x = res
                if x is not None:
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    if torch.is_tensor(x):
                        x = x.flatten(0, 1)
                return x
            
            list_res1 = []
            list_res2 = [] 
            list_idx1 = []
            list_idx2 = []
            
            batch_size = 3
            offset = 0
            while offset < len(list_frames):
                try:
                    _frames = stack_batch(list_frames, offset, batch_size).to('cuda', non_blocking=True)
                    _time_embedding = stack_batch(list_time_embedding, offset, batch_size).to('cuda', non_blocking=True)
                    _true_shape = stack_batch(list_true_shape, offset, batch_size).to('cuda', non_blocking=True)
                    _idx = stack_batch(list_idx, offset, batch_size).to('cuda', non_blocking=True)
                    
                    offset += batch_size
                    batch_size = 3
                    
                except:
                    batch_size = 1
                    continue
                
                with torch.no_grad():
                    res1, res2, idx1, idx2 = model(_frames, _time_embedding, _true_shape, _idx)
                    
                res1 = flatten_outputs(to_cpu(res1))
                res2 = flatten_outputs(to_cpu(res2))
                idx1 = flatten_outputs(to_cpu(idx1))
                idx2 = flatten_outputs(to_cpu(idx2))
                
                list_res1.append(res1)
                list_res2.append(res2)
                list_idx1.append(idx1)
                list_idx2.append(idx2)
                
            idx1 = collate_with_cat(list_idx1)
            idx2 = collate_with_cat(list_idx2)
            res1 = collate_with_cat(list_res1)
            res2 = collate_with_cat(list_res2)
                        
            def index_outputs(res, index):
                if isinstance(res, dict):
                    return {k: index_outputs(v, index) for k, v in res.items()}

                if isinstance(res, (tuple, list)):
                    return type(res)([index_outputs(x, index) for x in res])

                x = res
                if x is not None:
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    if torch.is_tensor(x):
                        x = x[index]
                return x
            
            exsiting_edge = set()
            unique_idx = []
            for _idx, (i, j) in enumerate(zip(idx1.tolist(), idx2.tolist())):
                if (i, j) in exsiting_edge:
                    continue
                
                exsiting_edge.add((i, j))
                exsiting_edge.add((j, i))
                
                unique_idx.append(_idx)
            
            idx1 = index_outputs(idx1, unique_idx)
            idx2 = index_outputs(idx2, unique_idx)
            res1 = index_outputs(res1, unique_idx)
            res2 = index_outputs(res2, unique_idx)
            
            view1_idx = idx1.tolist() + idx2.tolist()
            view1 = {
                'img': torch.cat([imgs[v]['img'] for v in view1_idx], dim=0),
                'idx': view1_idx
            }
            
            view2_idx = idx2.tolist() + idx1.tolist()
            view2 = {
                'img': torch.cat([imgs[v]['img'] for v in view2_idx], dim=0),
                'idx': view2_idx
            }
            
            pred1 = {
                'pts3d': torch.cat([res1['pts3d_ij'], res1['pts3d_ji']], dim=0),
                'conf': torch.cat([res1['conf_ij'], res1['conf_ji']], dim=0)
            }
            
            pred2 = {
                'pts3d_in_other_view': torch.cat([res2['pts3d_in_other_view_ji'], res2['pts3d_in_other_view_ij']], dim=0),
                'conf': torch.cat([res2['conf_ji'], res2['conf_ij']], dim=0)
            }
            outputs = {
                'view1': view1,
                'view2': view2,
                'pred1': pred1,
                'pred2': pred2
            }
               
            with torch.enable_grad():
                if len(imgs) > 2:
                    mode = eval(args.pose_mode)
                    scene = global_aligner(
                        outputs, device=device, mode=mode, verbose=not silent,
                        shared_focal=not args.not_shared_focal and not args.use_gt_focal,
                        flow_loss_weight=args.flow_loss_weight, flow_loss_fn=args.flow_loss_fn,
                        depth_regularize_weight=args.depth_regularize_weight,
                        num_total_iter=args.n_iter, temporal_smoothing_weight=args.temporal_smoothing_weight, motion_mask_thre=args.motion_mask_thre,
                        flow_loss_start_epoch=args.flow_loss_start_epoch, flow_loss_thre=args.flow_loss_thre, translation_weight=args.translation_weight,
                        sintel_ckpt=args.eval_dataset == 'sintel', use_self_mask=not args.use_gt_mask, sam2_mask_refine=args.sam2_mask_refine,
                        empty_cache=len(imgs) >= 80 and len(outputs['view1']['idx']) > 600, pxl_thre=args.pxl_thresh, # empty cache to make it run on 48GB GPU
                        flow_net=flow_net, sam2=sam2
                    )
                    if args.use_gt_focal:
                        focal_path = os.path.join(
                            img_path.replace('final', 'camdata_left'), seq, 'focal.txt'
                        )
                        focals = np.loadtxt(focal_path)
                        focals = focals[::args.pose_eval_stride]
                        original_img_size = cv2.imread(filelist[0]).shape[:2]
                        resized_img_size = tuple(imgs[0]['img'].shape[-2:])
                        focals = focals * max(
                            (resized_img_size[0] / original_img_size[0]),
                            (resized_img_size[1] / original_img_size[1])
                        )
                        scene.preset_focal(focals, requires_grad=False)  # TODO: requires_grad=False
                    lr = 0.01
                    loss = scene.compute_global_alignment(
                        init='mst', niter=args.n_iter, schedule=args.pose_schedule, lr=lr,
                    )
                else:
                    mode = GlobalAlignerMode.PairViewer
                    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

            if args.save_pose_qualitative:
                outfile = get_3D_model_from_scene(
                    outdir=save_dir, silent=silent, scene=scene, min_conf_thr=2, as_pointcloud=True, mask_sky=False,
                    clean_depth=True, transparent_cams=False, cam_size=0.01, save_name=seq
                )
            else:
                outfile = None
            pred_traj = scene.get_tum_poses()

            os.makedirs(f'{save_dir}/{seq}', exist_ok=True)
            scene.clean_pointcloud()
            scene.save_tum_poses(f'{save_dir}/{seq}/pred_traj.txt')
            scene.save_focals(f'{save_dir}/{seq}/pred_focal.txt')
            scene.save_intrinsics(f'{save_dir}/{seq}/pred_intrinsics.txt')
            scene.save_depth_maps(f'{save_dir}/{seq}')
            scene.save_dynamic_masks(f'{save_dir}/{seq}')
            scene.save_conf_maps(f'{save_dir}/{seq}')
            scene.save_init_conf_maps(f'{save_dir}/{seq}')
            scene.save_rgb_imgs(f'{save_dir}/{seq}')
            enlarge_seg_masks(f'{save_dir}/{seq}', kernel_size=5 if args.use_gt_mask else 3)

            gt_traj_file = metadata['gt_traj_func'](img_path, anno_path, seq)
            traj_format = metadata.get('traj_format', None)

            if args.eval_dataset == 'sintel':
                gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=args.pose_eval_stride)
            elif traj_format is not None:
                gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format)
            else:
                gt_traj = None

            if gt_traj is not None:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj, seq=seq, filename=f'{save_dir}/{seq}_eval_metric.txt'
                )
                plot_trajectory(
                    pred_traj, gt_traj, title=seq, filename=f'{save_dir}/{seq}.png'
                )
            else:
                ate, rpe_trans, rpe_rot = 0, 0, 0
                outfile = None
                bug = True

            ate_list.append(ate)
            rpe_trans_list.append(rpe_trans)
            rpe_rot_list.append(rpe_rot)
            outfile_list.append(outfile)

            # Write to error log after each sequence
            with open(error_log_path, 'a') as f:
                f.write(f'{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n')
                f.write(f'{ate:.5f}\n')
                f.write(f'{rpe_trans:.5f}\n')
                f.write(f'{rpe_rot:.5f}\n')

        except Exception as e:
            if 'out of memory' in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, 'a') as f:
                    f.write(f'OOM error in sequence {seq}, skipping this sequence.\n')
                print(f'OOM error in sequence {seq}, skipping...')
            elif 'Degenerate covariance rank' in str(e) or 'Eigenvalues did not converge' in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, 'a') as f:
                    f.write(f'Exception in sequence {seq}: {str(e)}\n')
                print(f'Traj evaluation error in sequence {seq}, skipping.')
            else:
                raise e  # Rethrow if it's not an expected exception
            
    # Aggregate results across all processes
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    bug_tensor = torch.tensor(int(bug), device=device)

    bug = bool(bug_tensor.item())

    # Handle outfile_list
    outfile_list_all = [None for _ in range(world_size)]

    outfile_list_combined = []
    for sublist in outfile_list_all:
        if sublist is not None:
            outfile_list_combined.extend(sublist)

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if rank == 0:
        with open(f'{save_dir}/_error_log.txt', 'a') as f:
            # Copy the error log from each process to the main error log
            for i in range(world_size):
                with open(f'{save_dir}/_error_log_{i}.txt', 'r') as f_sub:
                    f.write(f_sub.read())
            f.write(f'Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n')

    return avg_ate, avg_rpe_trans, avg_rpe_rot, outfile_list_combined, bug
