import os
import json
import torch
import torch.distributed as dist



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


def init_distributed_mode(world_size):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        distributed = False
        return

    distributed = True
    torch.cuda.set_device(gpu)
#    dist_backend = 'gloo'
    dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=dist_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return gpu

def on_master(func, *args, **kwargs):
    if is_main_process():
        return func(*args, **kwargs)


def save_dict(model_dir, _dict, name):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'dict', f'{name}.json')
    os.makedirs(os.path.join(model_dir, 'dict'), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_dict, f, indent=2, sort_keys=True)


def save_params(model_dir, params, name='params'):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict


def load_json(filename):
    with open(filename) as f:
        arr = json.load(f)
    return arr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr
    

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
               continue
            else:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def update_args(args, old_dict):
    for key, val in old_dict.items():
        args.__dict__[key] = val
        

def save_ckpt(model_dir, ckpt, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    torch.save(ckpt, os.path.join(model_dir, 'checkpoints', 
        'epoch{}.pt'.format(epoch)))


def get_grad(model):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
               continue
            else:
                param_norm = p.grad.detach()
                print(name, param_norm)
                print('weight', p.data)
