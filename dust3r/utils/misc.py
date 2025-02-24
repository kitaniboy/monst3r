# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import builtins
import datetime
import os
import time
import math
import json
from collections import defaultdict, deque

import torch
import cv2
import numpy as np
from dust3r.utils.vo_eval import save_trajectory_tum_format
from PIL import Image

import torch.distributed as dist
from torch import inf
from einops import rearrange, repeat, einsum

def init_distributed_mode(args):
    if (not hasattr(args, 'world_size')) or (args.world_size < 2):
        setup_for_distributed(True)
        args.distributed = False
        return

    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank,
                                         timeout=datetime.timedelta(seconds=3600),
                                         device_id=torch.device(f"cuda:{args.gpu}"))
    torch.distributed.barrier()
    do_print = True if bool(args.print_all_proc) else args.rank == 0
    setup_for_distributed(do_print)

def setup_for_distributed(do_print):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if do_print or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        is_tensor = isinstance(x, torch.Tensor)
        if not is_tensor:
            x = torch.tensor(x)
        
        device = x.device
        dist.all_reduce(x.cuda(), op=dist.ReduceOp.AVG)

        x = x.to(device)
        if not is_tensor:
            x = x.item()

        return x
    else:
        return x

def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """ flip so that tensor[0::2] <=> tensor[1::2] """
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res, feat = head(decout, (H, W))
        return res, feat

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            res, feat = head(decout, (H, W))
            return transposed(res), feat.swapaxes(1, 2)

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result, l_feat = head(selout(is_landscape), (H, W))
        p_result, p_feat = head(selout(is_portrait), (W, H))
        p_result, p_feat = transposed(p_result), p_feat.swapaxes(1, 2)

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        feat = l_feat.new(B, *l_feat.shape[1:])

        return result

    return wrapper_yes if activate else wrapper_no

def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res, feat = head(decout, (H, W))
        return res, feat

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            res, feat = head(decout, (H, W))
            return transposed(res), feat.swapaxes(1, 2)

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result, l_feat = head(selout(is_landscape), (H, W))
        p_result, p_feat = head(selout(is_portrait), (W, H))
        p_result, p_feat = transposed(p_result), p_feat.swapaxes(1, 2)

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        feat = l_feat.new(B, *l_feat.shape[1:])

        return result

    return wrapper_yes if activate else wrapper_no

def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}

def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr

def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

def save_tum_poses(traj, path):
    # traj = self.get_tum_poses()
    save_trajectory_tum_format(traj, path)
    return traj[0] # return the poses

def save_focals(focals, path):
    # convert focal to txt
    # focals = self.get_focals()
    np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
    return focals

def save_intrinsics(K_raw, path):
    # K_raw = self.get_intrinsics()
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
    return K_raw

def save_conf_maps(conf, path):
    # conf = self.get_conf()
    for i, c in enumerate(conf):
        np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
    return conf

def save_rgb_imgs(imgs, path):
    # imgs = self.imgs
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        img = img[..., ::-1]
        cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
    return imgs

def save_dynamic_masks(dynamic_masks, path):
    # dynamic_masks = self.dynamic_masks
    for i, dynamic_mask in enumerate(dynamic_masks):
        cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))
    return dynamic_masks

def save_depth_maps(depth_maps, path):
    images = []
    for i, depth_map in enumerate(depth_maps):
        depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        img_path = f'{path}/frame_{(i):04d}.png'
        cv2.imwrite(img_path, depth_map_colored)
        images.append(Image.open(img_path))
        # Save npy file
        np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
    
    # Save gif using Pillow
    images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    return depth_maps

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return [to_cpu(xx) for xx in x]
    
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.GradScaler('cuda', enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith('decoder_embed') or var_name.startswith('enc_norm'): # part of the last black
        return enc_depth
    elif var_name.startswith('dec_blocks'):
        layer_id = int(var_name.split('.')[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith('dec_norm'): # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ['head','prediction_head']):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)

def get_parameter_groups(model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values 
    assert layer_decay==1.0 or 0.<layer_decay<1.
    if layer_decay<1.:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, 'dec_blocks') else 0
        num_layers = enc_depth+dec_depth
        layer_decay_values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay<1.:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
        
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr

class token_rearrange_fn():
    def __init__(self, x):
        B, T = x.shape[:2]
        self.B = B
        self.T = T
        self._repeat_index = repeat(torch.arange(T, device=x.device), 't -> t tm1', tm1=T-1)

        self._row_index_swap = repeat(torch.arange(T, device=x.device), 't1 -> (t2 t1)', t1=T, t2=T)[1:].view(T-1, T+1)[:,:-1].reshape(T, T-1)
        self._col_index_swap = repeat(torch.arange(T-1, device=x.device), 'tm1 -> tm1 t', tm1=T-1, t=T).reshape(T, T-1)
        self._row_index_triu, self._col_index_triu = torch.triu_indices(T, T-1, device=x.device).unbind(0)
        self._row_index_tril, self._col_index_tril = self._col_index_triu+1, self._row_index_triu.clone()
        
    def repeat_tokens(self, tokens):
        tokens = rearrange(tokens, '(b t) ... -> b t ...', t=self.T)
        return tokens[:, self._repeat_index] # b t tm1 n d
    
    def swap_tokens(self, tokens):
        return tokens[:, self._row_index_swap, self._col_index_swap]
    
    def flatten_tokens(self, tokens):
        return rearrange(tokens, 'b t tm1 ... -> (b t tm1) ...', t=self.T, tm1=self.T-1)
    
    def unflatten_tokens(self, tokens):
        return rearrange(tokens, '(b t tm1) ... -> b t tm1 ...', b=self.B, t=self.T, tm1=self.T-1)
        
    def triu_tokens(self, tokens):
        return tokens[:, self._row_index_triu, self._col_index_triu]

    def tril_tokens(self, tokens):
        return tokens[:, self._row_index_tril, self._col_index_tril]
    
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        if name not in self.meters:
            self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ':' + str(len(str(len_iterable))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it,obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iterable))