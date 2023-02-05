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
import ops.evaluate as evaluate
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

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    testset = datasets.ImageFolder(
        f'{args.data_dir}/val/',
        transform=test_transforms
    )
    testsampler = DistributedSampler(testset, world_size, rank, shuffle=False)
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        sampler=testsampler,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )
    # MAX_QUERIES = 1132
    
    ## Architectures
    classifier, _, _ = load('alexnet/imagenet')
    classifier_embed = classifier[:16]
    classifier_fc = classifier[16:]
    classifier_fc = classifier_fc.to(device)
    classifier_fc = DistributedDataParallel(classifier_fc, device_ids=[gpu])
    
    classifier_embed = classifier_embed.to(device)
    classifier_embed = DistributedDataParallel(classifier_embed, device_ids=[gpu])

    ## Optimization
    criterion = nn.CrossEntropyLoss()

    classifier_embed.eval()
    classifier_fc.eval()
    torch.distributed.barrier()

    y_test_pred_all, y_test = [], []
    test_loss = 0. 
    total_test = 0.
    for test_batch_i, (test_images, test_labels) in enumerate(testloader):
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_bs = test_labels.size(0)

        # inference
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                batch_logits_test_all = classifier_fc(classifier_embed(test_images))
        batch_logits_test_all = batch_logits_test_all.float()
        batch_y_pred_all = batch_logits_test_all.argmax(dim=1)
        y_test_pred_all.append(batch_y_pred_all.cpu())
        print(batch_logits_test_all.sum(1))
        print(batch_y_pred_all[:25])
        print(test_labels[:25])
        print((batch_y_pred_all==test_labels).sum() / test_bs)
        
        y_test.append(test_labels.cpu())
        total_test += test_bs
        
        test_loss = criterion(batch_logits_test_all, test_labels)
        
        batch_acc = evaluate.compute_accuracy(batch_y_pred_all, test_labels)
        print(f'{test_batch_i} | loss: {test_loss} | batch_acc: {batch_acc}')
    y_test = torch.hstack(y_test)
    y_test_pred_all = torch.hstack(y_test_pred_all)
    acc_all = evaluate.compute_accuracy(y_test_pred_all, y_test)
    print(f'{test_batch_i} | loss: {test_loss} | all acc: {acc_all}')


if __name__ == '__main__':
    args = parseargs()
    main(args)
