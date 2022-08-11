import imageio
import os
import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import pickle

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
import urllib.request
import zipfile


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        
        self.data.target -= 1   # Targets start at 1 by default, so shift to 0
        self.targets = []
        for i in range(len(self.data)):
            sample = self.data.iloc[i]
            target = sample.target 
            self.targets.append(target)
        
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile
        
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        #download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target 
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    

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
                #transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
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
        
    if dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2,1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        folder = 'Data/imagenet_raw/{}'.format('train' if train else 'val')
        dataset = datasets.ImageFolder(folder, transform=transform)
    
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



def accuracy(model, data_loader, device, topk=1):
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
            #scheduler.step(valid_loss)
            scheduler.step()

        train_acc = accuracy(model, train_loader, device=device)
        valid_acc = accuracy(model, valid_loader, device=device)    

        if (valid_acc > best_acc):
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

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
    model, _ = training_loop(model=model,
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
    
    model.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))
   
    return model

def get_imagenet_weights(model, num_tasks):
    pretrained_model = models.resnet18(pretrained=True)
    
    names = list(model.state_dict())
    pretrained_names = list(pretrained_model.state_dict())
    state_dict = {}
    
    for name in names:
        if 'fc' not in name:
            if 'bns' not in name:
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
        print(name)
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

        acc_after = np.round(100 * accuracy(net, test_loader, device), 2)
        print('Accuracy after retraining: ', acc_after)

        row = [args.seed, it, acc_before, acc_after, cr, sparsity]
        data.loc[it] = row
       
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
    fc_weights = model.fc.weight.cpu().detach()
    
    for task_id in range(num_learned):
        set_task(model, task_id)
        signal_task = features_out(model, signal, device)
        
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

def eval(args, model, train_dataset, test_dataset, path_to_save, device, method='IS', batch_size=32, max_num_learned=10, topk=1, shuffle=False):
        
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
            test_dataset[i].data = test_dataset[i].data.iloc[order]
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
                    correct_preds += correct_preds_batch(pred.cpu()[:, offset_a:offset_b], y_true.cpu()-offset_a)

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
    parser.add_argument('--dataset_name', type=str, default='cub-200', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='path to save/load dataset')
    parser.add_argument('--download_data', type=bool, default=True, help='download dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='network architecture to use')
    parser.add_argument('--path_pretrained_model', type=str, default='pretrained_model.pth', help='path to pretrained parameters')
    parser.add_argument('--path_init_params', type=str, default='init_params.pth', help='path to initialization parameters')
    parser.add_argument('--alpha_conv', type=float, default=0.95, help='fraction of importance to keep in conv layers')
    parser.add_argument('--alpha_fc', type=float, default=1, help='fraction of importance to keep in fc layers')
    parser.add_argument('--num_tasks', type=int, default=4, help='number of tasks')
    parser.add_argument('--num_classes', type=int, default=200, help='number of classes')
    parser.add_argument('--num_classes_per_task', type=int, default=50, help='number of classes per task')
    parser.add_argument('--num_iters', type=int, default=1, help='number of pruning iterations')   # 3
    parser.add_argument('--prune_batch_size', type=int, default=100, help='number of examples for pruning')
    parser.add_argument('--batch_size', type=int, default=64, help='number of examples per training batch')
    parser.add_argument('--test_batch_size', type=int, default=20, help='number of examples per test batch')
    parser.add_argument('--train_epochs', type=int, default=40, help='number training epochs')      # 40
    parser.add_argument('--retrain_epochs', type=int, default=30, help='number of retraining epochs after pruning') #30
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='multistep', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')                   # 1e-2 SGD; 1e-3 Adam
    parser.add_argument('--decay_epochs_train', nargs='+', type=int, default=[20, 30], help='epochs for multistep decay')  # [20, 30]
    parser.add_argument('--decay_epochs_retrain', nargs='+', type=int, default=[10, 20], help='epochs for multistep decay')   # [8, 15]
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')   # 0.1
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay during retraining')         
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--order_name', type=str, default='default', help='name of class ordering. Options: defult, seed1993, seed1605')
    
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
    
    orders = {
        'default' : [i for i in range(args.num_classes)],
        'seed1993' : [
            168, 136,  51, 9, 183, 101, 171, 99, 42, 159, 191, 70, 16, 188, 27, 10, 175, 26, 68, 187, 98, 6, 85, 35, 112, 43,
            100, 0, 103, 181,  88,  59, 4, 2, 116, 174, 94, 80, 106, 1, 147, 17,141, 131, 72, 23, 173, 54, 197, 118, 87, 32,
            79, 104,  91,  19, 135, 107, 178,  36,  11, 199, 142,   8, 122, 3,  28,  57, 153, 172, 190,  56,  49,  44,  97,  62, 151, 169,
            194,  55, 192,  12, 189,  78,  66, 180,  15, 137, 109, 134,  92, 119, 126,  52, 170,  40, 148,  65, 144,  64, 138,  45,  77,  89,
            154,  90,  71, 193,  74,  30, 113, 143,  96,  84,  67,  50, 186, 156,  69,  21,  18, 111, 108,  58, 125, 157, 150, 110, 182, 129,
            166,  83,  81,  60,  13, 165,  14, 176,  63, 117,   5,  22, 145, 121,  38,  41,  82, 127, 114,  20,  31,  53,  37, 163, 196, 130,
            152, 162,  86,  76,  24,  34, 184, 149,  33, 128, 198, 155, 146, 167, 139, 120, 140, 102,  47,  25, 158, 123,  46, 164,  61,   7,
            115,  75, 133, 160, 105, 132, 179, 124,  48,  73,  93,  39,  95, 195,  29, 177, 185, 161
        ],
        'seed1605' : [
            96, 101,  73, 161, 163,  24, 107, 147, 126, 108, 116, 151, 185, 134,  56,  37,  91, 111, 120, 106, 164, 121, 131, 193,   8,  65,
            54, 159, 115,  94, 117, 145,  82, 182, 140, 142, 186, 194, 158, 97,  88,   1, 173, 197,  75,  14,  35,  48, 104, 162,  34,  31,
            93, 149,   3, 114, 175,  32, 199,  40, 112, 132, 125,   2, 176, 128,  85,  74,  18, 195, 139,  86,  92, 102,  11,  99,  44, 136,
            90,  76, 146,  29,  81,  53,  84,  57,   5, 130,   0, 157,  26, 83,  20, 168, 110, 198,  78, 127,  79, 135,  71, 183,  23,  15,
            122, 180,  19, 174, 129, 169,  89, 187, 154, 166,  64,  46,  33, 68, 138, 100, 188,  58,  49,  41, 184,  67, 167,  38,   7,   6,
            156,  61,  47, 170,  52,  21, 105, 192,  63, 191,  51, 119,  28, 165, 196, 148, 153, 150,  80,  95, 123,  12, 141, 160,  72,  36,
            27, 143,   4,  10, 171, 177,  16, 155,  30,   9, 179,  66, 189, 70,  60,  69, 133,  13, 118,  87,  25,  50,  17, 172, 103,  62,
            59,  98,  45, 124, 181, 178,  43,  55,  22, 152, 137,  39, 109, 190, 144,  77, 113,  42
        ]
    }
    
    
    
    order_name = args.order_name
    tasks_order = np.array(orders[order_name])
    
    pretrained = True
    only_fc = False
    
    task_labels = create_labels(args.num_classes, args.num_tasks, args.num_classes_per_task)
    train_dataset, test_dataset = task_construction(task_labels, args.dataset_name, tasks_order)

    net = init_model(args, device, only_fc=only_fc)


    if pretrained:
        net = get_imagenet_weights(net, args.num_tasks)
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
        acc = accuracy(net, test_loader, device)
        print('Accuracy: ', np.round(100*acc, 2))

        if not net.only_fc and args.num_iters > 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            prune_idx = np.random.permutation(train_dataset[task_id].data.shape[0])[:args.prune_batch_size]

            x_prune = []
            for idx in prune_idx:
                x_prune.append(torch.FloatTensor(train_dataset[task_id][idx][0]))

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
        
        avg_inc_acc1 = []
        
        net.eval()
        
        for task_id in range(NUM_LEARNED):
            set_task(net, task_id)
            train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)

            accs1.append(np.round(100*accuracy(net, test_loader, device), 2))

            print('Task {} accuracy with task_id: '.format(task_id+1), accs1[task_id])
            
            avg_inc_acc1.append(np.array(accs1).mean())
            
        print("Upper-bound Top-1: ", torch.FloatTensor(avg_inc_acc1))
        
        
        NUM_RUNS = 5
        shuffle = True

        net.eval()

        batch_size = [args.test_batch_size]
        method = "max"
               

        for bs in batch_size:  
            total_accs = []
            task_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            total_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            print("BATCH SIZE: ", bs)
            for i in range(NUM_RUNS):
                print("RUN ", i+1)
                total_acc, task_acc_matrix, total_acc_matrix = eval(args, net, train_dataset, test_dataset, 
                                                                    path_to_save, device, method=method, batch_size=bs, 
                                                                    max_num_learned=NUM_LEARNED, shuffle=shuffle)
                total_accs.append(torch.stack(total_acc))
                task_acc_matrices += task_acc_matrix
                total_acc_matrices += total_acc_matrix
                
            task_acc_matrices /= NUM_RUNS
            total_acc_matrices /= NUM_RUNS
            
            print("####")
            print("INCREMENTAL ACCURACY: ", torch.stack(total_accs, dim=0).mean(dim=0) ) 
            print('BWT: ', backward_transfer(total_acc_matrices, net.num_tasks) )


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