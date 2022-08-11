import imageio
import os
import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

import random
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms.functional import InterpolationMode
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

import torch.nn.init as init
import copy
from datetime import datetime 

import argparse
from collections import defaultdict
from tqdm.autonotebook import tqdm
from datetime import datetime 
import pickle
import urllib.request
import zipfile


class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")



class ImageNet(Dataset):   
    
    def __init__(self, root, imagenet_size=1000, train=True, transforms=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.loader = default_loader
        self.train = train


        self.imagenet_size = imagenet_size
        self.open_image = True
        self.suffix = ""
        self.metadata_path = None
        self.split = "train" if train else "val"
        
        self.data_path = "."
        self.download = download
        
        self.base_dataset(self.data_path)

    def set_custom_transforms(self):
        if not self.transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)
            
    def __len__(self):
        return len(self.data)     
            

    def base_dataset(self, data_path):
        if self.download:
            warnings.warn(
                "ImageNet incremental dataset cannot download itself,"
                " please see the instructions in the README."
            )
            
        print("Loading metadata of ImageNet_{} ({} split).".format(self.imagenet_size, self.split))
        metadata_path = os.path.join(
            data_path if self.metadata_path is None else self.metadata_path,
            "{}_{}{}.txt".format(self.split, self.imagenet_size, self.suffix)
        )

        self.data, self.targets = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")
                if not self.train:
                    path = path[0:4] + path[14:]
                
                self.data.append(os.path.join(data_path, path))
                self.targets.append(int(target))

        self.data = np.array(self.data)

        return self
    
        
    def __getitem__(self, idx):
        filepath = self.data[idx]
        path = os.path.join(filepath)
        target = self.targets[idx] 
        img = self.loader(path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target 
          



#class ImageNet1000(ImageNet100):
#    imagenet_size = 1000
    

def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'cub-200':
        input_shape, num_classes = (3, 224,224), 200    
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.Resize((224,224), interpolation=3))    
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def load_dataset(dataset, batch_size=64, train=True, workers=2, length=None):
    # Dataset
    if dataset == 'mnist':
        #mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        dataset = datasets.MNIST('Data', train=train, download=True, transform=transform)
    if dataset == 'cifar10':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR10('Data', train=train, download=True, transform=transform) 
    if dataset == 'cifar100':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100('Data', train=train, download=True, transform=transform)
    if dataset == 'tiny-imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        dataset = TINYIMAGENET('Data', train=train, download=True, transform=transform)
    if dataset == 'cub-200':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=63 / 255),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),                
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        dataset = Cub2011('.', train=train, download=True, transform=transform)
        
    if 'imagenet' in dataset:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        if dataset == 'imagenet100':        
            dataset = ImageNet('.', imagenet_size=100, train=train, download=False, transforms=transform)
        elif dataset == 'imagenet1000':  
            dataset = ImageNet('.', imagenet_size=1000, train=train, download=False, transforms=transform)
    
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    '''
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             **kwargs)
    '''
    return dataset


def get_loaders(train_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               shuffle=True, 
                                               num_workers=2) 

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)   

    return train_loader, test_loader 


def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return torch.tensor((list(map(lambda x: list(order).index(x), y))))

def task_construction(task_labels, dataset_name, order):
    train_dataset = load_dataset(dataset_name, train=True)
    test_dataset = load_dataset(dataset_name, train=False)
       
    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)
       
    train_dataset.targets = _map_new_class_index(train_dataset.targets, order)
    test_dataset.targets = _map_new_class_index(test_dataset.targets, order)
    
    if 'cub' in dataset_name:
        train_dataset.data.target = train_dataset.targets.clone() 
        test_dataset.data.target = test_dataset.targets.clone() 
    
    train_dataset=split_dataset_by_labels(train_dataset, task_labels)
    test_dataset=split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset, test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx=np.in1d(dataset.targets, labels)
        splited_dataset=copy.deepcopy(dataset)
        splited_dataset.targets = torch.tensor(splited_dataset.targets)[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)

    return datasets

def create_labels(num_classes, num_tasks, num_classes_per_task):
        
    tasks_order = np.arange(num_classes)
    labels = tasks_order.reshape((num_tasks, num_classes_per_task))
    return labels


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )
        
class AffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBN, self).__init__(dim, affine=True)           

class MultitaskNonAffineBN(nn.Module):
    def __init__(self, dim):
        super(MultitaskNonAffineBN, self).__init__()
        self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(args.num_tasks)])

    def forward(self, x, task_id):
        return self.bns[task_id](x)
    
class MultitaskBN(nn.Module):
    def __init__(self, dim, args, affine=True, only_fc=False):
        super(MultitaskBN, self).__init__()
        
        self.only_fc = only_fc        
        
        if affine:
            self.bns = nn.ModuleList([AffineBN(dim) for _ in range(args.num_tasks)])
        else:    
            self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(args.num_tasks)])

    def forward(self, x, task_id):
        return self.bns[task_id](x) 
    
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)        
    elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _masks_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)        

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, args, device, stride=1, only_fc=True, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MultitaskBN(planes, args)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MultitaskBN(planes, args)

        self.task_id = 0

        self.downsample = nn.Sequential()
        self.planes = planes
        self.device = device
        self.in_planes = in_planes
        self.stride = stride


        self.block_masks = self._make_masks(in_planes, planes, stride)
        
        self.tasks_masks = []

        self.planes = planes
        self.in_planes = in_planes
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                         MultitaskBN(self.expansion * planes, args)
                    )

    def add_mask(self):
        self.tasks_masks.append(copy.deepcopy(self.block_masks))       

    def _make_masks(self, in_planes, planes, stride):
        if stride != 1 or in_planes != self.expansion*planes:
            mask = [torch.ones(planes, in_planes, 3, 3), torch.ones(planes, planes, 3, 3), torch.ones(self.expansion*planes, in_planes, 1, 1)]  
        else:
            mask = [torch.ones(planes, in_planes, 3, 3), torch.ones(planes, planes, 3, 3), torch.ones(planes)] 
        
        return mask   
    def forward(self, x):
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(self.device)
        out = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)       
        out = F.relu(self.bn1(out, self.task_id))
        
        active_conv = self.conv2.weight*self.tasks_masks[self.task_id][1].to(self.device)
        out = F.conv2d(out, weight=active_conv, bias=None, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)       
        out = self.bn2(out, self.task_id)        
               
        if self.stride != 1 or self.in_planes != self.planes:
            downsample = list(self.downsample.children())[0]
            active_downsample = downsample.weight*self.tasks_masks[self.task_id][-1].to(self.device)
            
            bn = list(self.downsample.children())[1]
            downsample = F.conv2d(x, weight=active_downsample, bias=None, stride=downsample.stride, padding=downsample.padding, groups=downsample.groups)
            downsample = bn(downsample, self.task_id)
            out += downsample
        else:
            downsample = self.downsample(x)
            out += downsample*(self.tasks_masks[self.task_id][-1].reshape((-1, 1, 1)).expand((downsample.size(-3), downsample.size(-2), downsample.size(-1))).to(self.device))
        
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, device, only_fc=True, task_id=0):
        super(ResNet, self).__init__()
        
        _outputs = [64, 128, 256, 512]
        #_outputs = [21, 42, 85, 170]
        self.in_planes = _outputs[0]

        self.num_blocks = num_blocks

        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.num_tasks = 0
        
        self.only_fc = only_fc
        self.args = args
        self.device = device

        self.task_id = task_id

        self.conv1 = nn.Conv2d(3,  _outputs[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_masks = torch.ones(_outputs[0], 3, 7, 7)
        
        self.bn1 = MultitaskBN(_outputs[0], self.args)
        
        self.layer1, self.layer1_masks  = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2, self.layer2_masks = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3, self.layer3_masks = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.layer4, self.layer4_masks = self._make_layer(block, _outputs[3], num_blocks[3], stride=2)
        
        self.layers_masks = [self.layer1_masks, self.layer2_masks, self.layer3_masks, self.layer4_masks]

        self.fc = nn.Linear(_outputs[3]*block.expansion, self.num_classes)
        self.fc_masks = [torch.ones(self.num_classes, _outputs[3]*block.expansion), torch.ones(self.num_classes)]

        self.apply(_weights_init)

        self.tasks_masks = []

        self._add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

    def _add_mask(self, task_id):
        self.num_tasks += 1
        
        network_mask = [copy.deepcopy(self.conv1_masks), copy.deepcopy(self.layers_masks), copy.deepcopy(self.fc_masks)]
       
        self.tasks_masks.append(copy.deepcopy(network_mask) )
        
        for layer in range(len(network_mask[1])):                              # layer x block x 0/1
            for block in range(len(network_mask[1][layer])):
                Block = list(list(self.children())[layer+2])[block]
                Block.add_mask()
                for conv in range(2):
                    self.tasks_masks[task_id][1][layer][block][conv] = Block.tasks_masks[task_id][conv]

                self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[task_id][-1]    

      
        index = self.num_classes_per_task*task_id
        if index+self.num_classes_per_task < self.num_classes-1:
            self.tasks_masks[-1][-1][-1][(index+self.num_classes_per_task):] = 0
            self.tasks_masks[-1][-1][-2][(index+self.num_classes_per_task):, :] = 0
            
        if task_id > 0:
            self.tasks_masks[-1][-1][-1][:index] = 0
            self.tasks_masks[-1][-1][-2][:index, :] = 0

    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        
        for id in range(1, self.num_tasks):
            if not self.only_fc:
                self.masks_union[0] = 1*torch.logical_or(self.masks_union[0], self.tasks_masks[id][0])
                for layer in range(len(self.masks_union[1])):
                    for block in range(0, len(self.masks_union[1][layer])):
                        for conv in range(2):
                            self.masks_union[1][layer][block][conv] = 1*torch.logical_or(self.masks_union[1][layer][block][conv], self.tasks_masks[id][1][layer][block][conv])

                        self.masks_union[1][layer][block][-1] = 1*torch.logical_or(self.masks_union[1][layer][block][-1], self.tasks_masks[id][1][layer][block][-1])    

            self.masks_union[-1][0] = 1*torch.logical_or(self.masks_union[-1][0], self.tasks_masks[id][-1][0])
            self.masks_union[-1][1] = 1*torch.logical_or(self.masks_union[-1][1], self.tasks_masks[id][-1][1])  

    def set_masks_intersection(self):
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        
        for id in range(1, self.num_tasks):
            if not self.only_fc:
                self.masks_intersection[0] = 1*torch.logical_and(self.masks_intersection[0], self.tasks_masks[id][0])
                for layer in range(len(self.masks_intersection[1])):
                    for block in range(0, len(self.masks_intersection[1][layer])):
                        for conv in range(2):
                            self.masks_intersection[1][layer][block][conv] = 1*torch.logical_and(self.masks_intersection[1][layer][block][conv], self.tasks_masks[id][1][layer][block][conv])

                        self.masks_intersection[1][layer][block][-1] = 1*torch.logical_and(self.masks_intersection[1][layer][block][-1], self.tasks_masks[id][1][layer][block][-1])    

            self.masks_intersection[-1][0] = 1*torch.logical_and(self.masks_intersection[-1][0], self.tasks_masks[id][-1][0])
            self.masks_intersection[-1][1] = 1*torch.logical_and(self.masks_intersection[-1][1], self.tasks_masks[id][-1][1])        

    def set_trainable_masks(self, task_id):
        if task_id > 0:
            if not self.only_fc:
                self.trainable_mask[0] = copy.deepcopy( 1*((self.tasks_masks[task_id][0] - self.masks_union[0]) > 0) )
                for layer in range(len(self.trainable_mask[1])):                              # layer x block x 0/1
                    for block in range(len(self.trainable_mask[1][layer])):
                        Block = list(list(self.children())[layer+2])[block]
                        for conv in range(2):
                            self.trainable_mask[1][layer][block][conv] = copy.deepcopy( 1*((self.tasks_masks[task_id][1][layer][block][conv] - self.masks_union[1][layer][block][conv]) > 0) )

                        if Block.stride != 1 or Block.in_planes != Block.planes: 
                            self.trainable_mask[1][layer][block][-1] = copy.deepcopy( 1*((self.tasks_masks[task_id][1][layer][block][-1] - self.masks_union[1][layer][block][-1]) > 0) )    

            self.trainable_mask[-1][0] = copy.deepcopy( 1*((self.tasks_masks[task_id][-1][0] - self.masks_union[-1][0]) > 0) )
            self.trainable_mask[-1][1] = copy.deepcopy( 1*((self.tasks_masks[task_id][-1][1] - self.masks_union[-1][1]) > 0) )
        else:    
            self.trainable_mask = copy.deepcopy(self.tasks_masks[0])  


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        layers_masks = []
        for stride in strides:
            basicblock = block(self.in_planes, planes, self.args, self.device, stride, self.only_fc)
            layers.append(basicblock)
            layers_masks.append(basicblock.block_masks)
            self.in_planes = planes * block.expansion       

        return nn.Sequential(*layers), layers_masks

    def features(self, x):
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(self.device)
        out = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)       
        out = F.relu(self.bn1(out, self.task_id))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1) 
        return out     

    def forward(self, x):
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(self.device)
        out = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)       
        out = F.relu(self.bn1(out, self.task_id))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        active_weight = self.fc.weight*self.tasks_masks[self.task_id][-1][0].to(self.device)
        active_bias = self.fc.bias*self.tasks_masks[self.task_id][-1][1].to(self.device)
        out = F.linear(out, weight=active_weight, bias=active_bias)
        
        return out

    def _save_masks(self, file_name='net_masks.pt'):
        masks_database = {}
        
        for task_id in range(self.num_tasks):
            masks_database['conv1.mask.task{}'.format(task_id)] = self.tasks_masks[task_id][0]

            for layer in range(len(self.num_blocks)):
                for block in range(self.num_blocks[layer]): 
                    for conv_num in range(2):
                        name = 'layer{}.{}.conv{}.mask.task{}'.format(layer+1, block, conv_num+1, task_id)
                        masks_database[name] = self.tasks_masks[task_id][1][layer][block][conv_num]

                    name = 'layer{}.{}.downsample.mask.task{}'.format(layer+1, block, task_id)
                    masks_database[name] = self.tasks_masks[task_id][1][layer][block][-1]

            masks_database['fc.weight.mask.task{}'.format(task_id)] = self.tasks_masks[task_id][-1][0]
            masks_database['fc.bias.mask.task{}'.format(task_id)] = self.tasks_masks[task_id][-1][1]  

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name='net_masks.pt', num_tasks=1):
        #self.num_tasks = 0
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            self.tasks_masks[task_id][0] = masks_database['conv1.mask.task{}'.format(task_id)]
            
            for layer in range(len(self.num_blocks)):                              # layer x block x 0/1
                for block in range(self.num_blocks[layer]):
                    Block = list(list(self.children())[layer+2])[block]
                    #Block.add_mask(task_id)
                    for conv in range(2):
                        name = 'layer{}.{}.conv{}.mask.task{}'.format(layer+1, block, conv+1, task_id)
                        Block.tasks_masks[task_id][conv] = masks_database[name]
                        self.tasks_masks[task_id][1][layer][block][conv] = Block.tasks_masks[task_id][conv]

                    name = 'layer{}.{}.downsample.mask.task{}'.format(layer+1, block, task_id)
                    Block.tasks_masks[task_id][-1] = masks_database[name]    
                    self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[task_id][-1]
            
            self.tasks_masks[task_id][-1][0] = masks_database['fc.weight.mask.task{}'.format(task_id)]
            self.tasks_masks[task_id][-1][1] = masks_database['fc.bias.mask.task{}'.format(task_id)]
            
            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)
                
        self.set_masks_union()
        self.set_masks_intersection()



def resnet18(args, only_fc, device):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, device, only_fc)
                            
def resnet20(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, num_classes_per_task, device)


def resnet44(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes):
    return ResNet(BasicBlock, [18, 18, 18])    


def init_model(args, device, only_fc=True):
    model = resnet18(args, only_fc, device)
    model = model.to(device)

    return model

def rewrite_parameters(net, old_params, device):
    for (name, param), (old_name, old_param) in zip(net.named_parameters(), old_params()):
        if (name == 'conv1.weight' and not net.only_fc):
            param.data = old_param.data*(1-net.trainable_mask[0]).to(device) + param.data*net.trainable_mask[0].to(device)
        elif ('fc' in name):
            if ('weight' in name):
                param.data = old_param.data*(1-net.trainable_mask[-1][0]).to(device) + param.data*net.trainable_mask[-1][0].to(device)
            else:
                param.data = old_param.data*(1-net.trainable_mask[-1][1]).to(device) + param.data*net.trainable_mask[-1][1].to(device)
        elif not net.only_fc:
            for layer_num in range(len(net.num_blocks)):
                for block_num in range(net.num_blocks[layer_num]):
                    if (name == 'layer{}.{}.conv1.weight'.format(layer_num + 1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][0]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][0].to(device)
                    elif (name == 'layer{}.{}.conv2.weight'.format(layer_num + 1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][1]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][1].to(device)
                    elif (name == 'layer{}.{}.downsample.0.weight'.format(layer_num+1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][-1]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][-1].to(device) 
                        
    for (name, param), (old_name, old_param) in zip(net.named_parameters(), old_params()):
        for task_id in range(0, net.task_id):
            if 'bns.{}'.format(task_id) in name:
                param.data = 1*old_param.data
            
            
def set_task(model, task_id):
    model.task_id = task_id
    for layer in range(len(model.num_blocks)):
            for block in range(model.num_blocks[layer]):
                Block = list(model.children())[layer+2][block]
                Block.task_id = task_id 
                
def resnet_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def resnet_total_params_mask(model, task_id=0):
    total_number_conv = 0
    total_number_fc = 0

    for name, param in list(model.named_parameters()):
        if (name == 'conv1'):
            total_number_conv += model.tasks_masks[task_id][0].sum()
        elif ('fc' in name):
            if ('weight' in name):
                total_number_fc += model.tasks_masks[task_id][-1][0].sum()
            else:
                total_number_fc += model.tasks_masks[task_id][-1][1].sum()
        else:
            for layer in range(len(model.num_blocks)):
                for block in range(model.num_blocks[layer]):
                    if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                        total_number_conv += model.tasks_masks[task_id][1][layer][block][0].sum()
                    elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                        total_number_conv += model.tasks_masks[task_id][1][layer][block][1].sum()

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def resnet_get_architecture(model):
    arch = []
    convs = []
    fc = []

    convs.append(torch.sum(model.conv1_masks.sum(dim=(1, 2, 3)) > 0).item())
    for block, num_block in enumerate(model.num_blocks):
        block_masks = []
        for i in range(num_block):
            block_conv_masks = []
            for conv_num in range(2):
                block_conv_masks.append(torch.sum(model.layers_masks[block][i][conv_num].sum(dim=(1, 2, 3)) > 0).item())

            block_masks.append(block_conv_masks)

        convs.append(block_masks)

    arch.append(convs)

    fc.append(torch.sum(model.fc_masks[0].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def resnet_compute_flops(model):
    arch_conv, arch_fc = resnet_get_architecture(model)
    flops = 0
    k = 3
    h = 32
    w = 32
    in_channels = 3

    flops += (2 * h * w * (in_channels * k ** 2) - 1) * arch_conv[0]

    in_channels = arch_conv[0]

    for block, num_block in enumerate(model.num_blocks):
        for i in range(num_block):
            for conv_num in range(2):
                if (conv_num == 1):
                    out_channels = torch.sum(torch.logical_or(model.layers_masks[block][i][1].sum(dim=(1, 2, 3)),
                                                              model.layers_masks[block][i][-1]) > 0).item()
                else:
                    out_channels = arch_conv[1 + block][i][conv_num]

                flops += (2 * h * w * (in_channels * k ** 2) - 1) * out_channels

            in_channels = out_channels

            flops += h * w * model.layers_masks[block][i][-1].sum().item()

        h /= 2
        w /= 2

    flops += (2 * arch_fc[0] - 1) * 10

    return flops
    
    
def correct_preds_batch(output, targets, topk=1):
    """Computes the precision@k for the specified values of k"""
    output, targets = torch.tensor(output), torch.tensor(targets)


    #nb_classes = len(np.unique(targets))
    #topk = min(topk, nb_classes)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0).item()
        
    #print(correct_k, output.size(0))    
    return torch.tensor(correct_k)  



def accuracy(model, data_loader, device, topk=5):
    correct_preds = 0 
    n = 0
    
    offset_a = model.task_id*model.num_classes_per_task
    offset_b = (model.task_id+1)*model.num_classes_per_task
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device) 

            y_preds = model(X)
                        
            n += y_true.size(0)
            correct_preds += correct_preds_batch(y_preds.cpu()[:, offset_a:offset_b], y_true.cpu() - offset_a, topk=topk)

    #print(correct_preds)
    
    return (correct_preds/n).item()
    

def plot_losses(train_losses, valid_losses):
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    
def train_resnet(train_loader, model, criterion, optimizer, old_params, device, task_id):
   
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass
        y_hat = model(X) 
        offset_a = task_id*model.num_classes_per_task
        offset_b = (task_id+1)*model.num_classes_per_task
               
        loss = criterion(y_hat[:, offset_a:offset_b], y_true-offset_a) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()

        '''
        with torch.no_grad():
            for name, param in list(model.named_parameters()):
                if (name == 'conv1' and not net.only_fc):
                    param.grad.data = param.grad.data*(model.trainable_mask[0]).to(device)
                elif ('fc' in name):
                    if ('weight' in name):
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][0]).to(device)
                    else:
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][1]).to(device)
                elif not model.only_fc:
                    for layer in range(len(model.num_blocks)):
                        for block in range(model.num_blocks[layer]):
                            if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][0]).to(device)
                            elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][1]).to(device)
                            elif (name == 'layer{}.{}.downsample.0.weight'.format(layer+1, block)):
                                param.data = param.data*(model.layers_masks[layer][block][-1]).to(device) 
        '''

        optimizer.step()
        
        with torch.no_grad():
            rewrite_parameters(model, old_params, device)       
        
       
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, task_id, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
        offset_a = task_id*model.num_classes_per_task
        offset_b = (task_id+1)*model.num_classes_per_task
        
        loss = criterion(y_hat[:, offset_a:offset_b], y_true-offset_a) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop(model, criterion, optimizer, scheduler,
                  train_loader, valid_loader, epochs, task_id, device, model_name, file_name='VGG_curr.pth', print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    for epoch in range(0, epochs):
        # training
        
        model, optimizer, train_loss = train_resnet(train_loader, model, criterion, optimizer, old_params, device, task_id)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, task_id, device)
            valid_losses.append(valid_loss)
            scheduler.step()

        #train_acc = accuracy(model, train_loader, device=device)
        train_acc = 0
        valid_acc = accuracy(model, valid_loader, device=device)    

        
        if (valid_acc >= best_acc):
            if (valid_acc == best_acc) and (valid_loss < best_loss):
                torch.save(model.state_dict(), file_name)             
            if (valid_acc > best_acc):            
                torch.save(model.state_dict(), file_name)
                best_acc = valid_acc
            
            best_loss = valid_loss

        if epoch % print_every == (print_every - 1):
                            
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch+1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    #plot_losses(train_losses, valid_losses)
    
    return model, (train_losses, valid_losses)


def train(args, model, train_loader, test_loader, device, task_id=0):
    
    loss = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.decay_epochs_train,
                                                     gamma=args.gamma)
    net, _ = training_loop(model=model,
                           criterion=loss,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           train_loader=train_loader,
                           valid_loader=test_loader,
                           epochs=args.train_epochs,
                           task_id=task_id,
                           model_name=args.model_name,
                           device=device,
                           file_name=args.path_pretrained_model)
    
    net.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))
   
    return net

def get_imagenet_weights(model, num_tasks=10):
    pretrained_model = models.resnet18(pretrained=True)
    
    names = list(model.state_dict())
    pretrained_names = list(pretrained_model.state_dict())
    state_dict = {}
    
    for name in names:
        if 'fc' not in name:
            if 'bns' not in name:
                #print(name)
                state_dict[name] = pretrained_model.state_dict()[name]
            else:
                for task_id in range(num_tasks):
                    if 'bns.{}.'.format(task_id) in name:
                        pretrained_name = name.replace('bns.{}.'.format(task_id), '')
                        state_dict[name] = pretrained_model.state_dict()[pretrained_name]
        else:
            state_dict[name] = model.state_dict()[name]
                        
        
    del pretrained_model 
    
    model.load_state_dict(state_dict)
    
    return model

def resnet_conv_block_pruning(net, layer_num, block_num, conv_num, alpha, x_batch, task_id, residual=0):
    if layer_num == 0:
        conv = net.conv1
        bn = net.bn1
        active_conv = conv.weight*net.tasks_masks[task_id][0].to(net.device)

        name = 'conv1.weight'
        name_bn = 'bn1.bns.{}'.format(task_id)
    else:
        Block = list(net.named_children())[layer_num+1][1][block_num]
        if (conv_num >= 0):
            conv = list(list(net.named_children())[layer_num + 1][1][block_num].named_children())[2*conv_num][1]
            active_conv = conv.weight*net.tasks_masks[task_id][1][layer_num-1][block_num][conv_num].to(net.device) 

            bn = list(list(net.named_children())[layer_num + 1][1][block_num].named_children())[2*conv_num + 1][1]
            name = 'layer{}.{}.conv{}.weight'.format(layer_num, block_num, conv_num + 1)
            name_bn = 'layer{}.{}.bn{}.bns.{}'.format(layer_num, block_num, conv_num + 1, task_id)
        else:
            conv = list(Block.named_children())[-1][1][0]
            active_conv = conv.weight*net.tasks_masks[task_id][1][layer_num-1][block_num][-1].to(net.device) 
            
            bn = list(Block.named_children())[-1][1][1]
            name = 'layer{}.{}.downsample.0.weight'.format(layer_num, block_num)
            name_bn = 'layer{}.{}.downsample.1.bns.{}'.format(layer_num, block_num, task_id)    
                        

    bn_out = bn(F.conv2d(x_batch, weight=active_conv, stride=conv.stride, padding=conv.padding), task_id)
    
    if conv_num == 1:
        block_out = F.relu(bn_out + residual)
    else:
        if conv_num >= 0:
            block_out = F.relu(bn_out)
        else:
            block_out = bn_out 

    if layer_num == 0:
        block_out = F.max_pool2d(block_out, kernel_size=3, stride=2, padding=1)
    
    block_out_mean = block_out.mean(dim=0)

    padding = conv.padding
    stride = conv.stride
    kernel_size = conv.kernel_size
    zero_kernel = torch.zeros(kernel_size)

    filters = net.state_dict()[name]

    p2d = (padding[0],) * 2 + (padding[1],) * 2
    n = x_batch.size(3)
    m = x_batch.size(2)

    x_batch = F.pad(x_batch, p2d, "constant", 0)

    for k in range(filters.size(0)):

        if (block_out_mean[k]).norm(dim=(0, 1)) == 0:
            if layer_num == 0:
                net.tasks_masks[task_id][0][k] = zero_kernel
            else:
                net.tasks_masks[task_id][1][layer_num-1][block_num][conv_num][k] = zero_kernel
                if conv_num == 1:
                    if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
                        net.tasks_masks[task_id][1][layer_num-1][block_num][-1][k] = torch.zeros((1, 1))

                        downsample_name_bn = 'layer{}.{}.downsample.1.bns.{}'.format(layer_num, block_num, task_id)

                        net.state_dict()[downsample_name_bn+'.weight'][k] = 0
                        net.state_dict()[downsample_name_bn+'.bias'][k] = 0
                        net.state_dict()[downsample_name_bn+'.running_mean'][k] = 0
                        net.state_dict()[downsample_name_bn+'.running_var'][k] = 0 
                    else:
                        net.tasks_masks[task_id][1][layer_num-1][block_num][-1][k] = 0

            net.state_dict()[name_bn + '.weight'][k] = 0
            net.state_dict()[name_bn + '.bias'][k] = 0
            net.state_dict()[name_bn + '.running_mean'][k] = 0
            net.state_dict()[name_bn + '.running_var'][k] = 0
        else:
            importances = torch.zeros(filters.size(1), ((n+2* padding[0] - kernel_size[0])//stride[0]+1), ((m+2*padding[1]-kernel_size[1])//stride[1]+1))

            for i in range(kernel_size[0]//2, (n+2*padding[0])-kernel_size[0]//2, stride[0]):
                for j in range(kernel_size[1]//2, (m+2*padding[1])-kernel_size[1]//2, stride[1]):
                    input = x_batch[:, :, (i-kernel_size[0]//2):(i+kernel_size[0]//2+1),
                            (j-kernel_size[1]//2):(j+kernel_size[1]//2+1)].abs().mean(dim=0)

                    importances[:, (i-kernel_size[0]//2)//stride[0], (j-kernel_size[1]//2)//stride[1]] = torch.sum(torch.abs(input*filters[k]), dim=(1, 2))

            importances = torch.norm(importances, dim=(1, 2))
            sorted_importances, sorted_indices = torch.sort(importances, dim=0, descending=True)

            pivot = torch.sum(sorted_importances.cumsum(dim=0) < alpha*importances.sum())
            if pivot < importances.size(0) - 1:
                pivot += 1
            else:
                pivot = importances.size(0) - 1

            # delete all connectons that are less important than the pivot
            thresh = sorted_importances[pivot]
            kernel_zero_idx = torch.nonzero(importances <= thresh).reshape(1, -1).squeeze(0)

            if layer_num == 0:
                net.tasks_masks[task_id][0][k][kernel_zero_idx] = zero_kernel
            else:
                net.tasks_masks[task_id][1][layer_num-1][block_num][conv_num][k][kernel_zero_idx] = zero_kernel

    if conv_num == 1:
        pruned_channels = torch.nonzero(
            bn_out.abs().mean(dim=0).norm(dim=(1, 2))/residual.abs().mean(dim=0).norm(dim=(1, 2)) < (1-alpha)/alpha).reshape(1, -1).squeeze(0)
        net.tasks_masks[task_id][1][layer_num-1][block_num][conv_num][pruned_channels] = zero_kernel
        
        net.state_dict()[name_bn + '.weight'][pruned_channels] = 0
        net.state_dict()[name_bn + '.bias'][pruned_channels] = 0
        net.state_dict()[name_bn+'.running_mean'][pruned_channels] = 0
        net.state_dict()[name_bn+'.running_var'][pruned_channels] = 0 
        '''
        pruned_channels = torch.nonzero(
            residual.abs().mean(dim=0).norm(dim=(1, 2))/bn_out.abs().mean(dim=0).norm(dim=(1, 2)) < (1-alpha)/alpha).reshape(1, -1).squeeze(0)
        
        if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
            net.tasks_masks[task_id][1][layer_num-1][block_num][-1][pruned_channels] = torch.zeros((1, 1))

            shortcut_name_bn = 'layer{}.{}.shortcut.1.bns.{}'.format(layer_num, block_num, task_id)

            #net.state_dict()[shortcut_name_bn+'.weight'][pruned_channels] = 0
            #net.state_dict()[shortcut_name_bn+'.bias'][pruned_channels] = 0
            net.state_dict()[shortcut_name_bn+'.running_mean'][pruned_channels] = 0
            net.state_dict()[shortcut_name_bn+'.running_var'][pruned_channels] = 0  
        else: 
            net.tasks_masks[task_id][1][layer_num-1][block_num][-1][pruned_channels] = 0
        '''
    #net._apply_mask(task_id)
    #print(name)
    return net, block_out    


def resnet_conv_pruning(net, alpha, x_batch, start_conv_prune, task_id, device):
    net.eval()
    named_params = list(net.named_parameters())

    for name, param in named_params:
        if (name == 'conv1.weight'):
            net, x_batch = resnet_conv_block_pruning(net, 0, 0, 0, alpha, x_batch, task_id)
        else:
            for layer in range(len(net.num_blocks)):
                for block in range(net.num_blocks[layer]):
                    Block = list(net.named_children())[layer+2][1][block]
                    for conv_num in range(2):
                        if (name == 'layer{}.{}.conv{}.weight'.format(layer+1, block, conv_num+1)):
                            if (conv_num == 0):
                                if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
                                    net, residual  = resnet_conv_block_pruning(net, layer+1, block, -1, alpha, x_batch, task_id, residual=0)
                                else:
                                    residual = list(Block.named_children())[-1][1](x_batch)
                                
                            net, x_batch = resnet_conv_block_pruning(net, layer+1, block, conv_num, alpha, x_batch, task_id, residual)
    return net


def resnet_backward_pruning(net, task_id):
    pruned_channels = torch.nonzero(net.tasks_masks[task_id][-1][0].sum(dim=0) == 0).reshape(1, -1).squeeze(0)

    kernel_size = list(list(net.layer3.named_children())[-1][1].named_children())[-3][1].kernel_size
    zero_kernel = torch.zeros(kernel_size)

    for name in reversed(list(net.state_dict())):
        for layer in range(len(net.num_blocks))[::-1]:
            for block in range(net.num_blocks[layer])[::-1]:
                Block = list(net.children())[layer+2][block]

                for conv_num in range(2)[::-1]:
                    if (('layer{}.{}.bn{}.bns.{}'.format(layer+1, block, conv_num+1, task_id) in name) and ('num_batches_tracked' not in name)):
                        net.state_dict()[name][pruned_channels] = 0
                    elif (name == 'layer{}.{}.conv{}.weight'.format(layer+1, block, conv_num+1)):
                        net.tasks_masks[task_id][1][layer][block][conv_num][pruned_channels] = zero_kernel

                        if (conv_num == 1):
                            if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
                                net.tasks_masks[task_id][1][layer][block][-1][pruned_channels] = torch.zeros((1, 1))
                                name_bn = 'layer{}.{}.downsample.1.bns.{}'.format(layer+1, block, task_id)

                                net.state_dict()[name_bn+'.weight'][pruned_channels] = 0
                                net.state_dict()[name_bn+'.bias'][pruned_channels] = 0
                                net.state_dict()[name_bn+'.running_mean'][pruned_channels] = 0
                                net.state_dict()[name_bn+'.running_var'][pruned_channels] = 0  
                            else:  
                                net.tasks_masks[task_id][1][layer][block][-1][pruned_channels] = 0   
                                
                            pruned_channels = torch.nonzero(
                                net.tasks_masks[task_id][1][layer][block][conv_num].sum(dim=(0, 2, 3)) == 0).reshape(1,-1).squeeze(0)
                        else:
                            if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
                                pruned_channels = torch.nonzero( (net.tasks_masks[task_id][1][layer][block][-1].sum(dim=(0,2,3))==0)*(net.tasks_masks[task_id][1][layer][block][conv_num].sum(dim=(0,2,3))==0)).reshape(1, -1).squeeze(0)
                            else:
                                pruned_channels = torch.nonzero( (1-net.tasks_masks[task_id][1][layer][block][-1])*(net.tasks_masks[task_id][1][layer][block][conv_num].sum(dim=(0,2,3))==0)).reshape(1, -1).squeeze(0)    


    net.tasks_masks[task_id][0][pruned_channels] = torch.zeros((7, 7))
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.weight'][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.bias'][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.running_mean'][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.running_var'][pruned_channels] = 0 

    return net   


def resnet_pruning(net, alpha_conv, alpha_fc, x_batch, task_id, device,
                  start_conv_prune=0, start_fc_prune=0):
    # do forward step fon convolutional layers
    # should be replaced by conv layers pruning and then the forward step
    if (start_conv_prune >= 0):
        net = resnet_conv_pruning(net, alpha_conv, x_batch, start_conv_prune, task_id, device)
    
    x_batch = net.features(x_batch)
    
    # pruner for fully-connected layers
    if (start_fc_prune >= 0):
        net = resnet_fc_pruning(net, alpha_fc, x_batch, task_id, device, start_fc_prune)

    #print('---Before backward: ', total_params_mask(net))
    net = resnet_backward_pruning(net, task_id)
    #print('---After backward: ', total_params_mask(net))  

    #net._apply_mask(task_id) 

    return net


def iterative_pruning(args, net, train_loader, test_loader, x_prune, task_id, device,
                      start_conv_prune=0, start_fc_prune=-1):
    cr = 1
    sparsity = 100
    columns = ['seed', 'iter', 'acc_pruned', 'acc_retrained', 'cr', 'remained_percentage']
    data = pd.DataFrame(columns=columns)
    acc = np.round(100 * accuracy(net, test_loader, device), 2)
    row = [args.seed, 0, acc, acc, cr, sparsity]
    data.loc[0] = row

    if args.model_name == 'lenet_mlp':
        x_prune = x_prune.reshape([-1, 28 * 28])

    init_masks_num = resnet_total_params_mask(net, task_id)

    for it in range(1, args.num_iters + 1):
        # before_params_num = total_params(net)
        before_masks_num = resnet_total_params_mask(net, task_id)
        net.eval()
        net = resnet_pruning(net=net,
                             alpha_conv=args.alpha_conv,
                             alpha_fc=args.alpha_fc,
                             x_batch=x_prune,
                             task_id=task_id,
                             device=device,
                             start_conv_prune=start_conv_prune,
                             start_fc_prune=start_fc_prune
                            )
        

        #net._apply_mask(task_id)
        net.set_trainable_masks(task_id)

        after_masks_num = resnet_total_params_mask(net, task_id)
        acc_before = np.round(100 * accuracy(net, test_loader, device), 2)
        # curr_arch = lenet_get_architecture(net)
        print('Accuracy before retraining: ', acc_before)
        print('Compression rate on iteration %i: ' %it, before_masks_num[0]/after_masks_num[0])
        print('Total compression rate: ', init_masks_num[0]/after_masks_num[0])
        print('The percentage of the remaining weights: ', 100*after_masks_num[0]/init_masks_num[0])
        # print('Architecture: ', curr_arch)

        cr = np.round(init_masks_num[0]/after_masks_num[0], 2)
        sparsity = np.round(100*after_masks_num[0]/init_masks_num[0], 2)

        if 'lenet' in args.model_name:
            if task_id == 0:
                init_params = args.path_init_params
            else:
                init_params = '{}_task{}_{}classes_{}_{}_it{}_seed{}.pth'.format(args.model_name, task_id+1, args.num_classes_per_task, 
                                                                                 args.optimizer, args.alpha_conv, args.num_iters, args.seed)

            net.load_state_dict(torch.load(init_params, map_location=device))
            #net._apply_mask(task_id)

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optimizer == 'radam':
            optimizer = RAdam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=args.decay_epochs_retrain,
                                                         gamma=args.gamma)
        loss = torch.nn.CrossEntropyLoss()

        net, _ = training_loop(
                               model=net,
                               criterion=loss,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               train_loader=train_loader,
                               valid_loader=test_loader,
                               epochs=args.retrain_epochs,
                               task_id=task_id,
                               model_name=args.model_name,
                               device=device,
                               file_name='resnet18.pth')

        net.load_state_dict(torch.load('resnet18.pth', map_location=device))
        #net._apply_mask(task_id)

        acc1_after = np.round(100*accuracy(net, test_loader, device, topk=1), 2)
        acc5_after = np.round(100*accuracy(net, test_loader, device, topk=5), 2)
        print('Accuracy after retraining: ', acc1_after, acc5_after)

       
        print('-------------------------------------------------')

    return net, data


def features_out(model, x, device):
    i=0
    bs=32
    out = []

    while i+bs < len(x):
        out.append(model.features(x[i:(i+bs)].to(device) ).cpu().detach())

        i += bs

    if (i < len(x) and i+bs >= len(x)):
        out.append(model.features(x[i:].to(device) ).cpu().detach())

    out = torch.cat(out)    
    
    return out
    
def compute_importances(model, signal, device, task_id=0):
    importances = []
    total_importance_per_neuron = []
   
    fc_weights = model.fc.weight.cpu().detach()*model.tasks_masks[task_id][-1][0]
    #scores = np.abs(signal.mean(dim=0))*fc_weights.abs()
    scores = signal.mean(dim=0)*fc_weights

    total_importance_per_neuron.append(scores.sum(axis=0))  
      
    
    importances.append(scores) 
    
    importances = torch.cat(importances) 
     
    return importances, total_importance_per_neuron

def joint_importance(model, signal, device, num_learned):
    importances = []
    total_importance_per_neuron = []
    #prototypes_mean, prototypes_std = get_prototypes(model)   
    fc_weights = model.fc.weight.cpu().detach()
    
    for task_id in range(num_learned):
        set_task(model, task_id)
        signal_task = features_out(model, signal, device)
        #signal_task = embeddings_augmentation(signal_task, num_emb=50)
        #signal_task = generate_embeddings(signal_task, task_id, num_emb=signal.size(0))
        
        #scores = torch.abs(signal_task).mean(dim=0)*fc_weights.abs()*model.tasks_masks[task_id][-1][0]
        scores = signal_task.mean(dim=0)*fc_weights*model.tasks_masks[task_id][-1][0]
        
        total_importance_per_neuron.append(scores.sum(axis=0))    

        importances.append(scores)
        del signal_task

    return importances, total_importance_per_neuron  


def compute_importance_train(model, train_dataset, device):
    importances_train = []
    total_importances_train = []
    for task_id in range(model.num_tasks):
        idx_est = np.random.permutation(train_dataset[task_id].data.shape[0])[:500]

        x = []
        for idx in idx_est:
            x.append(train_dataset[task_id][idx][0])
                    
        x = torch.stack(x, dim=0).to(device)
        
               
        set_task(model, task_id)
        x = features_out(model, x, device)
        
        importances, total_importance_per_neuron = compute_importances(model, x, device, task_id=task_id)
        
        importances_train.append(importances)
        total_importances_train.append(total_importance_per_neuron)
        
        del importances, total_importance_per_neuron, x
        
  
    return importances_train, total_importances_train 

def distance(Test, Train, mask):
    return torch.sum(torch.abs(Test-Train))/mask.sum()
    

def select_subnetwork(model, x, importances_train, device, num_layers=1):
    num_learned = len(importances_train)
    importance_x, total_importance_x = joint_importance(model, x, device, num_learned)
    dists = []
    for j in range(len(importance_x)):
        dist = 0
        
        for l in range(num_layers):
            dist += distance(importance_x[j], importances_train[j], model.tasks_masks[j][-1][0].cpu())
        
        dists.append(dist.item())
        
    j0 = np.argmin(dists)

    return j0     


def select_subnetwork_maxoutput(model, x, device, num_learned=10):
    max_out = []
    for task_id in range(num_learned):
        set_task(model, task_id)
        
        offset_a = task_id*model.num_classes_per_task
        offset_b = (task_id+1)*model.num_classes_per_task
        
        preds = model(x.to(device))[:, offset_a:offset_b]
        max_out.append(torch.max(preds, dim=1)[0].sum().cpu().detach())
        
    j0 = np.argmax(max_out)
    
    return j0

def eval(args, model, train_dataset, test_dataset, path_to_save, device, method='IS', batch_size=32, max_num_learned=10, topk=5, shuffle=False):
        
    model.load_state_dict(torch.load(path_to_save, map_location=device) )
    model.eval()

    if method == 'IS':
        importances_train, total_importances_train = compute_importance_train(model, train_dataset, device)
    elif method == 'nme':
        prototypes, _ = get_prototypes(model)

    total_acc = []
    total_acc_matrix = np.zeros((max_num_learned, max_num_learned))
    task_acc_matrix = np.zeros((max_num_learned, max_num_learned))
    
    test_loaders = []
    
    for i in range(max_num_learned):
        if shuffle:
            order = np.random.permutation(np.arange(len(test_dataset[i])))
            test_dataset[i].data = test_dataset[i].data[order]
            test_dataset[i].targets = test_dataset[i].targets[order]
        
        test_loaders.append(torch.utils.data.DataLoader(test_dataset[i], 
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=2)
                           )
    
    for num_learned in range(1, max_num_learned+1):
        total_correct_preds = 0
        total_size = 0
        for task_id in range(0, num_learned):
            dataset_size = len(test_dataset[task_id])
            total_size += dataset_size

            acc_task_classification = 0
            correct_preds = 0
            offset_a = task_id*model.num_classes_per_task
            offset_b = (task_id+1)*model.num_classes_per_task            
            
            for x, y_true in test_loaders[task_id]:

                x_tmp = torchvision.transforms.RandomHorizontalFlip(p=1)(x)

                x_tmp = torch.cat((x, x_tmp))
                
                if method == 'IS':
                    j0 = select_subnetwork(model, x_tmp, importances_train[:num_learned], device) 
                elif 'max' in method:    
                    j0 = select_subnetwork_maxoutput(model, x_tmp, device, num_learned)
                elif method == 'nme':    
                    j0 = select_subnetwork_icarl(model, x_tmp, prototypes, num_learned) 
                    
                del x_tmp
                

                if j0 == task_id:
                    acc_task_classification += x.size(0)

                    set_task(model, j0)

                    pred = model(x.to(device))
                    correct_preds += correct_preds_batch(pred.cpu()[:, offset_a:offset_b], y_true.cpu()-offset_a, topk=topk)

            total_correct_preds += correct_preds
            acc_task_classification /= dataset_size
            
            task_acc_matrix[task_id, num_learned-1] = acc_task_classification
            total_acc_matrix[task_id, num_learned-1] = correct_preds/dataset_size

        total_acc.append(100*(total_correct_preds/total_size).cpu())

        print(f'Accuracy after task {task_id+1}: {100*(total_correct_preds/total_size).item():.2f}%')
    
    acc = total_correct_preds/total_size
    print(f'Accuracy : {100*acc.item():.2f}%')

    return total_acc, task_acc_matrix, total_acc_matrix
    
    
def backward_transfer(acc_matrix, tasks_learned):
    bwt  = np.zeros(tasks_learned)
    
    for t in range(1, tasks_learned):
        for task_id in range(t):
            bwt[t] += 100*(acc_matrix[task_id, t] - acc_matrix[task_id, task_id])/t
    
    return list(-np.round(bwt, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='imagenet1000', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='path to save/load dataset')
    parser.add_argument('--download_data', type=bool, default=True, help='download dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='network architecture to use')
    parser.add_argument('--path_pretrained_model', type=str, default='pretrained_model.pth', help='path to pretrained parameters')
    parser.add_argument('--path_init_params', type=str, default='init_params.pth', help='path to initialization parameters')
    parser.add_argument('--alpha_conv', type=float, default=0.9, help='fraction of importance to keep in conv layers')
    parser.add_argument('--alpha_fc', type=float, default=1, help='fraction of importance to keep in fc layers')
    parser.add_argument('--num_tasks', type=int, default=10, help='number of tasks')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--num_classes_per_task', nargs='+', type=int, default=100, help='number of classes per task')
    parser.add_argument('--num_iters', type=int, default=1, help='number of pruning iterations')   # 3
    parser.add_argument('--prune_batch_size', type=int, default=1000, help='number of examples for pruning')
    parser.add_argument('--batch_size', type=int, default=128, help='number of examples per training batch')
    parser.add_argument('--test_batch_size', type=int, default=20, help='number of examples per test batch')
    parser.add_argument('--train_epochs', type=int, default=90, help='number training epochs')      # 90
    parser.add_argument('--retrain_epochs', type=int, default=30, help='number of retraining epochs after pruning') #30
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='multistep', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')                   # 1e-1 SGD; 1e-2 Adam
    parser.add_argument('--decay_epochs_train', nargs='+', type=int, default=[30, 60], help='epochs for multistep decay')  # [30, 60]
    parser.add_argument('--decay_epochs_retrain', nargs='+', type=int, default=[10, 20], help='epochs for multistep decay')   # [10, 20]
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')   # 0.1
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay during retraining')         
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN = True
    EVAL = True
    
    path_results = f"./results/{args.dataset_name}-{args.model_name}/{args.num_tasks}_tasks/"
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    
    
    print("STARTED")
    
    orders_imagenet100 = {
        '0' : [i for i in range(args.num_classes)],
        'icarl' : [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
    }
    
    orders_imagenet1000 = {
        '0' : [i for i in range(args.num_classes)],
        'icarl' : [
            54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419, 274, 108, 928, 856, 494, 836, 473, 650,
            85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289, 123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 
            850, 665, 990, 597, 722, 748, 14, 77, 437, 394, 859, 279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400, 471, 632, 275,
            730, 105, 523, 224, 186, 478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954,
            465, 533, 585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659, 360, 136, 578, 163, 427,
            70, 226, 925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39, 326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 348,
            220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957, 691, 155, 820, 584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410,
            374, 515, 484, 624, 409, 156, 455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248, 333,
            941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933, 306, 378, 76, 227, 426, 403, 322, 321, 808,
            393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917, 611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313,
            129, 978, 592, 573, 128, 243, 520, 887, 892, 696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 678, 745,
            845, 208, 188, 674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619, 217, 631, 934, 932, 568, 353, 863, 827, 425, 420, 99, 823, 113,
            974, 438, 874, 343, 118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407, 56, 927, 655, 809, 839, 640, 297, 34, 497,
            210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535, 139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 
            767, 666, 22, 525, 902, 233, 250, 825, 79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701,
            532, 586, 729, 253, 486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452, 245, 487, 706, 2,
            137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 826, 668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 
            733, 575, 781, 864, 617, 171, 795, 132, 145, 368, 147, 327, 713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246, 
            449, 492, 162, 97, 59, 357, 198, 519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793,
            151, 847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987, 801, 629, 491, 605, 112, 
            429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725, 480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 
            100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213, 196, 743, 817, 433, 328, 970, 969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 
            86, 119, 203, 268, 718, 317, 926, 269, 161, 209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564, 
            185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170, 679, 838, 819, 485, 430, 190,
            566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861, 692, 686, 277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 
            737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599, 187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 
            267, 867, 772, 604, 618, 346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 869, 955, 17, 506, 963, 
            786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717, 116, 488, 796, 983, 646, 
            499, 53, 1, 603, 45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19, 965, 143, 555, 687, 235, 790, 125, 173, 364, 882, 727, 
            728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998, 991, 469, 967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 
            501, 440, 82, 684, 862, 574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308, 
            881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178, 489, 37, 197, 789, 530, 111, 
            876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953, 270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 
            377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749, 916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 
            194, 152, 981, 938, 854, 483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 559, 756, 25, 211, 158, 
            723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 331, 901, 416, 873, 754, 900, 435, 762, 124, 304, 329, 349, 295, 95, 
            451, 285, 225, 945, 697, 417
        ]
    }    

    if args.dataset_name == 'imagenet100':
        orders = orders_imagenet100
    elif args.dataset_name == 'imagenet1000':
        orders = orders_imagenet1000
    
    order_name = 'icarl'
    tasks_order = np.array(orders[order_name])
    
    pretrained = False
    only_fc = False
    
    print(args.dataset_name)
        
    task_labels = create_labels(args.num_classes, args.num_tasks, args.num_classes_per_task)
    train_dataset, test_dataset = task_construction(task_labels, args.dataset_name, tasks_order)

    net = init_model(args, device, only_fc=only_fc)


    if pretrained:
        net = get_imagenet_weights(net)
        mode = 'pretrained'
    else:    
        mode = 'not_pretrained'    


    if net.only_fc:
        for name, param in net.named_parameters():
            if 'fc' not in name and 'bn' not in name:
                param.requires_grad = False

    torch.save(net.state_dict(), args.path_init_params)

    num_tasks = args.num_tasks

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
        
    start_task = 0
    if start_task > 0:
        path_to_save = path_results+'{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, start_task, 
                                                                                         args.num_classes_per_task, args.optimizer, 
                                                                                         args.alpha_conv, args.num_iters, order_name)
        net.load_state_dict(torch.load(path_to_save, map_location=device) )

        net._load_masks(file_name=path_results+'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, 
                                                                                                              start_task, args.num_classes_per_task, 
                                                                                                              args.optimizer, args.alpha_conv, args.num_iters, 
                                                                                                              order_name), 
                        num_tasks=start_task)

        if start_task < num_tasks:
            net._add_mask(task_id=start_task)



    if TRAIN:
        for task_id in range(start_task, num_tasks):
            path_to_save = path_results+'{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, start_task, 
                                                                                         args.num_classes_per_task, args.optimizer, 
                                                                                         args.alpha_conv, args.num_iters, order_name)
            set_task(net, task_id)
            print("CURREN TASK: ", net.task_id)
            train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)

            net.set_trainable_masks(task_id)

            net = train(args, net, train_loader, test_loader, device=device, task_id=task_id)

            net.eval()
            
            acc1_before = np.round(100*accuracy(net, test_loader, device, topk=1), 2)
            acc5_before = np.round(100*accuracy(net, test_loader, device, topk=5), 2)
            print('Accuracy after training: ', acc1_before, acc5_before)

            if not net.only_fc and args.num_iters > 0:
                random.seed(args.seed)
                np.random.seed(args.seed)
                prune_idx = np.random.permutation(train_dataset[task_id].data.shape[0])[:args.prune_batch_size]

                x_prune = []
                for idx in prune_idx:
                    x_prune.append(train_dataset[task_id][idx][0])
                    
                x_prune = torch.stack(x_prune, dim=0).to(device)

                net, stats = iterative_pruning(args=args,
                                               net=net, 
                                               train_loader=train_loader, 
                                               test_loader=test_loader,
                                               x_prune=x_prune,
                                               task_id=task_id,
                                               device=device
                                               )


            net.set_masks_intersection()
            net.set_masks_union()

            torch.save(net.state_dict(), path_to_save)
            net._save_masks(path_results+'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, 
                                                                                                    start_task, args.num_classes_per_task, 
                                                                                                    args.optimizer, args.alpha_conv, args.num_iters, 
                                                                                                    order_name))

            if task_id < num_tasks-1:
                net._add_mask(task_id=task_id+1)
                print('-------------------TASK {}------------------------------'.format(task_id+1))
                      
    
    if EVAL:
        NUM_LEARNED = args.num_tasks
        if not TRAIN:
            path_to_save = path_results+'{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, start_task, 
                                                                                         args.num_classes_per_task, args.optimizer, 
                                                                                         args.alpha_conv, args.num_iters, order_name)
            net.load_state_dict(torch.load(path_to_save, map_location=device) )

            net._load_masks(file_name=path_results+'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, 
                                                                                                                  start_task, args.num_classes_per_task, 
                                                                                                                  args.optimizer, args.alpha_conv, args.num_iters, 
                                                                                                                  order_name), 
                            num_tasks=start_task)
    
    
        accs1 = []
        accs5 = []
        
        avg_inc_acc1 = []
        avg_inc_acc5 = []
        
        net.eval()
        
        for task_id in range(NUM_LEARNED):
            set_task(net, task_id)
            train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)

            accs1.append(np.round(100*accuracy(net, test_loader, device, topk=1), 2))
            accs5.append(np.round(100*accuracy(net, test_loader, device, topk=5), 2))

            print('Task {} accuracy with task_id: '.format(task_id+1), accs1[task_id], accs5[task_id])
            
            avg_inc_acc1.append(np.array(accs1).mean())
            avg_inc_acc5.append(np.array(accs5).mean())
            
        print("Upper-bound Top-1: ", torch.FloatTensor(avg_inc_acc1))
        print("Upper-bound Top-5: ", torch.FloatTensor(avg_inc_acc5))
        
        
        NUM_RUNS = 3
        shuffle = True

        net.eval()

        batch_size = [args.test_batch_size]
        topk = 5
        method = 'max'

        for bs in batch_size:  
            total_accs = []
            task_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            total_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            print("BATCH SIZE: ", bs)
            for i in range(NUM_RUNS):
                print("RUN ", i+1)
                total_acc, task_acc_matrix, total_acc_matrix = eval(args, net, train_dataset, test_dataset, 
                                                                    path_to_save, device, method=method, batch_size=bs, 
                                                                    max_num_learned=NUM_LEARNED, topk=topk, shuffle=shuffle)
                total_accs.append(torch.stack(total_acc))
                task_acc_matrices += task_acc_matrix
                total_acc_matrices += total_acc_matrix
                print(total_accs[-1])
                
            task_acc_matrices /= NUM_RUNS
            total_acc_matrices /= NUM_RUNS
            
            print("####")
            print("INCREMENTAL ACCURACY: ", torch.stack(total_accs, dim=0).mean(dim=0) ) 

            print("####################################################################")            
                
            with open(path_results+'{}_{}_{}tasks_{}classes_{}_{}_avg_acc_{}_bs{}_order_{}.pickle'.format(args.dataset_name, args.model_name,
                                                                                                          args.dataset_name, args.model_name, args.num_tasks, 
                                                                                                          args.num_classes_per_task, args.optimizer, 
                                                                                                          args.alpha_conv, method, bs, order_name), 
                                                                                                          'wb') as handle:
                pickle.dump(total_acc_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                

            with open(path_results+'{}_{}_{}tasks_{}classes_{}_{}_task_select_{}_bs{}_order_{}.pickle'.format(args.dataset_name, args.model_name,
                                                                                                              args.dataset_name, args.model_name, 
                                                                                                              args.num_tasks, args.num_classes_per_task, 
                                                                                                              args.optimizer, args.alpha_conv, method, 
                                                                                                              bs, order_name), 'wb') as handle:
                pickle.dump(task_acc_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL) 
   

if __name__== "__main__":
    main()