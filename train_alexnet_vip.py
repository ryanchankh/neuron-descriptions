import argparse
import random
import time
import glob
from tqdm import tqdm   
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from src.exemplars.models import load
from arch.fc1 import FullyConnectedQuerier

import ops.ip as ip
import utils
import wandb


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data', type=str, default='imagenet')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=1132)
    parser.add_argument('--max_queries_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='imagenet')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--n', type=int, default=1, help='total number of nodes we’re going to use')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus on each node')
    parser.add_argument('--nr', type=int, default=1, help='rank of the current node within all the nodes')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='/cis/project/vision_sequences/ImageNet-ImageFolderFormat/', help='data directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint directory')
    args = parser.parse_args()
    return args

def multiply_mask(layer, mask):
    """Hooks for modifying layer output during forward pass."""
    layer_mask_idx = {
        'conv1': torch.arange(0, 64),
        'conv2': torch.arange(64, 256),
        'conv3': torch.arange(256, 640),
        'conv4': torch.arange(640, 896),
        'conv5': torch.arange(876, 1132)
    }
    layer_mask = torch.index_select(mask, 1, layer_mask_idx[layer])
    layer_mask = layer_mask[:, :, None, None]
    def hook(model, input, output):
        return output * layer_mask
    return hook

def add_hooks(model, mask, layer_hooks):
    for layer_idx, layer_name in zip([0, 3, 6, 8, 10], ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
        layer_hook = model.module[layer_idx].register_forward_hook(multiply_mask(layer_name, mask))
        layer_hooks.append(layer_hook)
    return layer_hooks

def remove_hooks(layer_hooks):
    for layer_hook in layer_hooks:
        layer_hook.remove()
        
def update_history(history, query):
    return history + query

def main(args):
    
    ## CUDA
    world_size = 3
    if args.distributed:
        gpu = utils.init_distributed_mode(world_size)
        rank = gpu
        print('distributed, rank', rank)
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    
    # Randomness
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## Directory
    if utils.is_main_process():
        run = wandb.init(entity='jhuvisionlab', project='IP-dissect', name=args.name, mode=args.mode)
        model_dir = os.path.join(args.save_dir, args.data, f'{run.id}')
        os.makedirs(model_dir, exist_ok=True)
        wandb.config.update(args)
        utils.save_params(model_dir, vars(args))
        print(model_dir)

    ## Data
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    trainset = datasets.ImageFolder(f'{args.data_dir}/train/', transform=train_transforms)
    train_sampler = DistributedSampler(trainset, world_size, rank, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    testset = datasets.ImageFolder(f'{args.data_dir}/val/', transform=test_transforms)
    testsampler = DistributedSampler(testset, world_size, rank, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, sampler=testsampler)
    MAX_QUERIES = 1132
    
    ## Architectures
    classifier, _, _ = load('alexnet/imagenet')
    classifier_embed = classifier[:16]
    classifier = classifier.half().to(device)
    classifier = DistributedDataParallel(classifier, device_ids=[gpu])
    classifier.eval()
    print(classifier)
    
    classifier_embed = classifier_embed.half().to(device)
    classifier_embed = DistributedDataParallel(classifier_embed, device_ids=[gpu])
    classifier_embed.eval()
    
    querier = FullyConnectedQuerier(input_dim=9216, n_queries=MAX_QUERIES)
    querier = querier.half().to(device)
    querier = DistributedDataParallel(querier, device_ids=[gpu])
    querier.train()
    print(querier)

    ## Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(querier.parameters(), amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)
    

    ## Training
    scheduler_tau = torch.linspace(args.tau_start, args.tau_end, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        querier.train()
        tau = scheduler_tau[epoch]
        querier.module.update_tau(tau)
        trainloader.sampler.set_epoch(epoch)
        torch.cuda.synchronize()
        for train_images, train_labels in tqdm(trainloader):
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            train_bs = train_images.shape[0]
            optimizer.zero_grad()
            hooks = []
            
            # inference
            with torch.cuda.amp.autocast():
                # random sampling history
                random_mask = ip.sample_random_history(train_bs, MAX_QUERIES, args.max_queries).to(device)
                hooks = add_hooks(classifier_embed, random_mask, hooks)
                
                # query and update history
                train_embed = classifier_embed(train_images)
                train_query = querier(train_embed, random_mask)
                updated_mask = random_mask + train_query
                hooks = add_hooks(classifier, updated_mask, hooks)
                
                # predict with updated history
                train_logits = classifier(train_images)

            # backprop
            loss = criterion(train_logits, train_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # remove all hooks
            remove_hooks(hooks)

            # logging
            utils.on_master(
                wandb.log,
                {'train_epoch': epoch, 
                 'train_lr': utils.get_lr(optimizer),
                 'train_querier_grad_norm': utils.get_grad_norm(querier),
                 'train_classifier_grad_norm': utils.get_grad_norm(classifier),
                 'train_tau': tau,
                 'train_loss': loss.item()}
                )
            torch.cuda.synchronize()
        scheduler.step()


if __name__ == '__main__':
    args = parseargs()
    main(args)
