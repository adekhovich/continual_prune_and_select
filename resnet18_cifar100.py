import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import torch.autograd as autograd
from scipy.stats import entropy
from torch.utils.data.sampler import SubsetRandomSampler
import pickle

import matplotlib.pyplot as plt

import torch.nn.init as init
import copy
from datetime import datetime 

import argparse
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import torchvision.models as models

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from datetime import datetime 

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


import torchvision

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def load_dataset(dataset_name):
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])    
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./', 
                                                     train=True, 
                                                     transform=transforms_train, 
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./', 
                                                    train=False, 
                                                    transform=transforms_test, 
                                                    download=True)
    else:    
        train_dataset = torchvision.datasets.CIFAR100(root='./', 
                                                      train=True, 
                                                      transform=transforms_train, 
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='./', 
                                                     train=False, 
                                                     transform=transforms_test, 
                                                     download=True)
      

    return train_dataset, test_dataset 


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



def task_construction(task_labels, dataset_name, order=None):
    train_dataset, test_dataset=load_dataset(dataset_name)
    
    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)
    
    if order is not None:
        train_targets = -1*torch.tensor(np.ones(len(train_dataset.targets)), dtype=torch.long)
        test_targets = -1*torch.tensor(np.ones(len(test_dataset.targets)), dtype=torch.long)
        for i, label in enumerate(order):
            train_targets[train_dataset.targets == label] = i
            test_targets[test_dataset.targets == label] = i
        
        train_dataset.targets = train_targets.clone()
        test_dataset.targets = test_targets.clone()
    
    
    train_dataset=split_dataset_by_labels(train_dataset, task_labels)
    test_dataset=split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset,test_dataset

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
        
class AffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBN, self).__init__(dim, affine=True)        

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )

class MultitaskBN(nn.Module):
    def __init__(self, dim, num_tasks, affine=True):
        super(MultitaskBN, self).__init__()
        if affine:
            self.bns = nn.ModuleList([AffineBN(dim) for _ in range(num_tasks)])
        else:    
            self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(num_tasks)])

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

    def __init__(self, in_planes, planes, args, device, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MultitaskBN(planes, args.num_tasks)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MultitaskBN(planes, args.num_tasks)

        self.task_id = 0

        self.shortcut = nn.Sequential()
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
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     MultitaskBN(self.expansion * planes, args.num_tasks)
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
            shortcut = list(self.shortcut.children())[0]
            active_shortcut = shortcut.weight*self.tasks_masks[self.task_id][-1].to(self.device)
            
            bn = list(self.shortcut.children())[1]
            shortcut = F.conv2d(x, weight=active_shortcut, bias=None, stride=shortcut.stride, padding=shortcut.padding, groups=shortcut.groups)
            shortcut = bn(shortcut, self.task_id)
            out += shortcut
        else:
            shortcut = self.shortcut(x)
            out += shortcut*(self.tasks_masks[self.task_id][-1].reshape((-1, 1, 1)).expand((shortcut.size(-3), shortcut.size(-2), shortcut.size(-1))).to(self.device))
        
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, device, task_id=0):
        super(ResNet, self).__init__()
        
        #_outputs = [64, 128, 256, 512]
        _outputs = [21, 42, 85, 170]
        self.in_planes = _outputs[0]

        self.num_blocks = num_blocks

        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.num_tasks = 0
        self.args = args
        
        self.device = device

        self.task_id = task_id

        self.conv1 = nn.Conv2d(3,  _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_masks = torch.ones(_outputs[0], 3, 3, 3)
        self.bn1 = MultitaskBN(_outputs[0], args.num_tasks)
        
        self.layer1, self.layer1_masks  = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2, self.layer2_masks = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3, self.layer3_masks = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.layer4, self.layer4_masks = self._make_layer(block, _outputs[3], num_blocks[3], stride=2)
        
        self.layers_masks = [self.layer1_masks, self.layer2_masks, self.layer3_masks, self.layer4_masks]

        self.linear = nn.Linear(_outputs[3]*block.expansion, self.num_classes)
        self.linear_masks = [torch.ones(self.num_classes, _outputs[3]*block.expansion), torch.ones(self.num_classes)]

        self.apply(_weights_init)

        self.tasks_masks = []

        self._add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

    def _add_mask(self, task_id):
        self.num_tasks += 1
        network_mask = [copy.deepcopy(self.conv1_masks), copy.deepcopy(self.layers_masks), copy.deepcopy(self.linear_masks)] 

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

    def set_masks_union(self, num_learned=-1):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        
        if (num_learned < 0):
            num_learned = self.num_tasks

        for id in range(1, num_learned):
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
            basicblock = block(self.in_planes, planes, self.args, self.device, stride)
            layers.append(basicblock)
            layers_masks.append(basicblock.block_masks)
            self.in_planes = planes * block.expansion       

        return nn.Sequential(*layers), layers_masks

    def features(self, x):
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(self.device)
        out = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)       
        out = F.relu(self.bn1(out, self.task_id))

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
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        active_weight = self.linear.weight*self.tasks_masks[self.task_id][-1][0].to(self.device)
        active_bias = self.linear.bias*self.tasks_masks[self.task_id][-1][1].to(self.device)
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

                    name = 'layer{}.{}.shortcut.mask.task{}'.format(layer+1, block, task_id)
                    masks_database[name] = self.tasks_masks[task_id][1][layer][block][-1]

            masks_database['linear.weight.mask.task{}'.format(task_id)] = self.tasks_masks[task_id][-1][0]
            masks_database['linear.bias.mask.task{}'.format(task_id)] = self.tasks_masks[task_id][-1][1]  

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name='net_masks.pt', num_tasks=1):
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            self.tasks_masks[task_id][0] = masks_database['conv1.mask.task{}'.format(task_id)]
            
            for layer in range(len(self.num_blocks)):                              # layer x block x 0/1
                for block in range(self.num_blocks[layer]):
                    Block = list(list(self.children())[layer+2])[block]
                    for conv in range(2):
                        name = 'layer{}.{}.conv{}.mask.task{}'.format(layer+1, block, conv+1, task_id)
                        Block.tasks_masks[task_id][conv] = masks_database[name]
                        self.tasks_masks[task_id][1][layer][block][conv] = Block.tasks_masks[task_id][conv]

                    name = 'layer{}.{}.shortcut.mask.task{}'.format(layer+1, block, task_id)
                    Block.tasks_masks[task_id][-1] = masks_database[name]    
                    self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[task_id][-1]
            
            self.tasks_masks[task_id][-1][0] = masks_database['linear.weight.mask.task{}'.format(task_id)]
            self.tasks_masks[task_id][-1][1] = masks_database['linear.bias.mask.task{}'.format(task_id)]
            
            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)
                
        self.set_masks_union()
        self.set_masks_intersection()



def resnet18(args, device):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, device)
                            
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



def init_model(args, device):
    model = resnet18(args, device);
    model = model.to(device)
    return model



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
        elif ('linear' in name):
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
                    elif (name == 'layer{}.{}.shortcut.0.weight'.format(layer+1, block)):
                        total_number_conv += model.tasks_masks[task_id][1][layer][block][-1].sum()

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

    fc.append(torch.sum(model.linear_masks[0].sum(dim=0) > 0).item())

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



def accuracy(model, data_loader, device):
    correct_preds = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device) 
            y_preds = model(X)

            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()

    return (correct_preds / n).item()

def validate(valid_loader, model, criterion, task_id, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X) 
        offset_a = task_id*model.num_classes_per_task
        offset_b = (task_id+1)*model.num_classes_per_task
        
        loss = criterion(y_hat[:, offset_a:offset_b], y_true-offset_a) 

        
        running_loss += loss.item() * X.size(0)

        
    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss

def rewrite_parameters(net, old_params, device):
    for (name, param), (old_name, old_param) in zip(net.named_parameters(), old_params()):
        if (name == 'conv1.weight'):
            param.data = old_param.data*(1-net.trainable_mask[0]).to(device) + param.data*net.trainable_mask[0].to(device)
        elif ('linear' in name):
            if ('weight' in name):
                param.data = old_param.data*(1-net.trainable_mask[-1][0]).to(device) + param.data*net.trainable_mask[-1][0].to(device)
            else:
                param.data = old_param.data*(1-net.trainable_mask[-1][1]).to(device) + param.data*net.trainable_mask[-1][1].to(device)
        else:
            for layer_num in range(len(net.num_blocks)):
                for block_num in range(net.num_blocks[layer_num]):
                    if (name == 'layer{}.{}.conv1.weight'.format(layer_num + 1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][0]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][0].to(device)
                    elif (name == 'layer{}.{}.conv2.weight'.format(layer_num + 1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][1]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][1].to(device)
                    elif (name == 'layer{}.{}.shortcut.0.weight'.format(layer_num+1, block_num)):
                        param.data = old_param.data*(1-net.trainable_mask[1][layer_num][block_num][-1]).to(device) + param.data*net.trainable_mask[1][layer_num][block_num][-1].to(device) 
                        
    
    for (name, param), (old_name, old_param) in zip(net.named_parameters(), old_params()):
        for task_id in range(0, net.task_id):
            if 'bns.{}'.format(task_id) in name:
                param.data = 1*old_param.data
                
                        
    
def freeze_bn(net): 
    for name, param in net.named_parameters():
        if 'bns.{}'.format(net.task_id) in name:
            param.requires_grad = False
            net.bn1.bns[net.task_id].track_running_stats=False        
            print(name)
    
    return net

def set_task(model, task_id):
    model.task_id = task_id
    for layer in range(len(model.num_blocks)):
            for block in range(model.num_blocks[layer]):
                Block = list(model.children())[layer+2][block]
                Block.task_id = task_id

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
                if (name == 'conv1'):
                    param.grad.data = param.grad.data*(model.trainable_mask[0]).to(device)
                elif ('linear' in name):
                    if ('weight' in name):
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][0]).to(device)
                    else:
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][1]).to(device)
                else:
                    for layer in range(len(model.num_blocks)):
                        for block in range(model.num_blocks[layer]):
                            if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][0]).to(device)
                            elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][1]).to(device)
                            elif (name == 'layer{}.{}.shortcut.0.weight'.format(layer+1, block)):
                                param.data = param.data*(model.layers_masks[layer][block][-1]).to(device)     
        '''
       
        optimizer.step()
        
        with torch.no_grad():
            rewrite_parameters(model, old_params, device)
        

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss


def training_loop(model, criterion, optimizer, scheduler,
                  train_loader, valid_loader, epochs, task_id, model_name, 
                  device, file_name='model.pth', print_every=1):
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []

    if 'lenet' in model_name:
        train = train_lenet
    elif 'vgg' in model_name:
        train = train_vgg
    else:
        train = train_resnet

    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    print('TRAINING...')
    for epoch in range(0, epochs):
        # training

        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, old_params, device, task_id=task_id)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, task_id, device)
            valid_losses.append(valid_loss)
            scheduler.step()

        train_acc = accuracy(model, train_loader, device=device)
        valid_acc = accuracy(model, valid_loader, device=device)

        if valid_acc > best_acc:
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    return model, (train_losses, valid_losses)


def train(args, model, train_loader, test_loader, device, task_id=0):
    
    loss = torch.nn.CrossEntropyLoss()
    #loss = torch.nn.BCEWithLogitsLoss()
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=args.decay_epochs_retrain,
                                                         gamma=args.gamma)

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



def resnet_fc_pruning(net, alpha, x_batch, task_id, device, start_fc_prune = 0):
    layers = list(net.linear.state_dict())
    num_samples = x_batch.size()[0]

    fc_weight = net.linear.state_dict()[layers[0]]*net.tasks_masks[task_id][-1][0].to(device)
    fc_bias = net.linear.state_dict()[layers[1]]*net.tasks_masks[task_id][-1][1].to(device)

    #curr_layer = net.linear(x_batch)
    curr_layer = F.linear(x_batch, weight=fc_weight, bias=fc_bias)

    for i in range(curr_layer.size(1)):
        flow = torch.cat((x_batch * fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1)
        importances = torch.mean(torch.abs(flow), dim=0)

        sum_importance = torch.sum(importances)
        sorted_importances, sorted_indices = torch.sort(importances, descending=True)

        cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
        pivot = torch.sum(cumsum_importances < alpha * sum_importance)

        if pivot < importances.size(0)-1:
            pivot += 1
        else:
            pivot = importances.size(0)-1

        thresh = importances[sorted_indices][pivot]

        net.tasks_masks[task_id][-1][0][i][importances[:-1] <= thresh] = 0

        if importances[-1] <= thresh:
            net.tasks_masks[task_id][-1][1][i] = 0


    #net._apply_mask(task_id)
    
    return net


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
            name = 'layer{}.{}.shortcut.0.weight'.format(layer_num, block_num)
            name_bn = 'layer{}.{}.shortcut.1.bns.{}'.format(layer_num, block_num, task_id)    
            
            

    bn_out = bn(F.conv2d(x_batch, weight=active_conv, stride=conv.stride, padding=conv.padding), task_id)

    if conv_num == 1:
        block_out = F.relu(bn_out + residual)
    else:
        if conv_num >= 0:
            block_out = F.relu(bn_out)
        else:
            block_out = bn_out 

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

                        shortcut_name_bn = 'layer{}.{}.shortcut.1.bns.{}'.format(layer_num, block_num, task_id)

                        net.state_dict()[shortcut_name_bn+'.weight'][k] = 0
                        net.state_dict()[shortcut_name_bn+'.bias'][k] = 0
                        net.state_dict()[shortcut_name_bn+'.running_mean'][k] = 0
                        net.state_dict()[shortcut_name_bn+'.running_var'][k] = 0 
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
                                name_bn = 'layer{}.{}.shortcut.1.bns.{}'.format(layer+1, block, task_id)

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


    net.tasks_masks[task_id][0][pruned_channels] = zero_kernel
    net.state_dict()['bn1.bns.{}.weight'.format(task_id)][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}.bias'.format(task_id)][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.running_mean'][pruned_channels] = 0
    net.state_dict()['bn1.bns.{}'.format(task_id)+'.running_var'][pruned_channels] = 0 

    return net



def resnet_pruning(net, alpha_conv, x_batch, task_id, device,
                  start_conv_prune=0, start_fc_prune=0):
    # do forward step fon convolutional layers
    # should be replaced by conv layers pruning and then the forward step
    if (start_conv_prune >= 0):
        net = resnet_conv_pruning(net, alpha_conv, x_batch, start_conv_prune, task_id, device)
    
    x_batch = net.features(x_batch)
    
    #print('---Before backward: ', total_params_mask(net))
    net = resnet_backward_pruning(net, task_id)
    #print('---After backward: ', total_params_mask(net))  

    #net._apply_mask(task_id) 

    return net

def iterative_pruning(args, net, train_loader, test_loader, x_prune, task_id, device, path_to_save,
                      start_conv_prune=0, start_fc_prune=-1):
    cr = 1
    sparsity = 100
    acc = np.round(100 * accuracy(net, test_loader, device), 2)

   
    init_masks_num = resnet_total_params_mask(net, task_id)

    for it in range(1, args.num_iters + 1):
        # before_params_num = total_params(net)
        before_masks_num = resnet_total_params_mask(net, task_id)
        net.eval()
        net = resnet_pruning(net=net,
                             alpha_conv=args.alpha_conv,
                             x_batch=x_prune,
                             task_id=task_id,
                             device=device,
                             start_conv_prune=start_conv_prune,
                            )
        

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
                               file_name=path_to_save
                               )

        net.load_state_dict(torch.load(path_to_save, map_location=device))

        acc_after = np.round(100 * accuracy(net, test_loader, device), 2)
        print('Accuracy after retraining: ', acc_after)

        print('-------------------------------------------------')

    return net



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
   
    fc_weights = model.linear.weight.cpu().detach()*model.tasks_masks[task_id][-1][0]
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
    fc_weights = model.linear.weight.cpu().detach()
    
    for task_id in range(num_learned):
        set_task(model, task_id)
        signal_task = features_out(model, signal, device)
        
        #scores = torch.abs(signal_task).mean(dim=0)*fc_weights.abs()*model.tasks_masks[task_id][-1][0]
        scores = signal_task.mean(dim=0)*fc_weights*model.tasks_masks[task_id][-1][0]
        
        total_importance_per_neuron.append(scores.sum(axis=0))
          

        importances.append(scores)
        del signal_task

    return importances, total_importance_per_neuron

def distance(Test, Train, mask):
    #return torch.sum(torch.abs(Test[mask!=0]-Train[mask!=0])/Train[mask!=0])/mask.sum()
    return torch.sum(torch.abs(Test-Train))/mask.sum()

def compute_importance_train(model, train_dataset, device):
    importances_train = []
    total_importances_train = []
    for task_id in range(model.num_tasks):
        idx = np.random.permutation(np.arange( len(train_dataset[task_id]) ) )
        x = torch.FloatTensor(train_dataset[task_id].data)[idx]
        x = x.permute(0, 3, 1, 2)
        x = torchvision.transforms.Normalize(mean, std)(x.float()/255)
               
        set_task(model, task_id)
        x = features_out(model, x, device)
        
        importances, total_importance_per_neuron = compute_importances(model, x, device, task_id=task_id)
        
        importances_train.append(importances)
        total_importances_train.append(total_importance_per_neuron)
        
        del importances, total_importance_per_neuron, x
        
  
    return importances_train, total_importances_train

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

def get_prototypes(model):
    prototypes_mean = []
    prototypes_std = []
    for task_id in range(model.num_tasks):
        idx = np.random.permutation(np.arange( len(train_dataset[task_id]) ) )[:2000]
        x = torch.FloatTensor(train_dataset[task_id].data)[idx]
        x = x.permute(0, 3, 1, 2)
        x = torchvision.transforms.Normalize(mean, std)(x.float()/255)
               
        set_task(model, task_id)
        x = features_out(model, x)
        
        prototypes_mean.append(x.mean(dim=0))
        prototypes_std.append(x.std(dim=0))
        
    return prototypes_mean, prototypes_std

def select_subnetwork_icarl(model, x, prototypes, num_learned=10):
    dists = []
    #prototypes = get_prototypes(model)

    for task_id in range(num_learned):
        set_task(model, task_id)
        out = features_out(model, x)
        
        dists.append( ((out.mean(dim=0)-prototypes[task_id]).abs()).mean() )
        
    j0 = np.argmin(dists)   
    
    return j0

#prototypes_mean, prototypes_std = get_prototypes(net)

def select_subnetwork_maxoutput(model, x, num_learned, device):
    max_out = []
    for task_id in range(num_learned):
        set_task(model, task_id)
        preds = model(x.to(device))
        max_out.append(torch.max(preds[:, task_id*model.num_classes_per_task:((task_id+1)*model.num_classes_per_task)], dim=1)[0].sum().cpu().detach())
        
        
    j0 = np.argmax(max_out)
    
    return j0


def eval(args, model, train_dataset, test_dataset, path_to_save, device, method='IS', batch_size=32, max_num_learned=10, shuffle=False):
    columns = ['seed', 'batch_size', 'task', 'task_acc', 'num_samples', 'acc']
    data_GNu = pd.DataFrame(columns=columns)
    
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
            dataset_size = len(test_dataset[task_id].data)
            total_size += dataset_size

            acc_task_classification = 0
            correct_preds = 0
            for x, y_true in test_loaders[task_id]:

                x_tmp = torchvision.transforms.RandomHorizontalFlip(p=1)(x)

                x_tmp = torch.cat((x, x_tmp))
                
                if method == 'IS':
                    j0 = select_subnetwork(model, x_tmp, importances_train[:num_learned], device) 
                elif 'max' in method:    
                    j0 = select_subnetwork_maxoutput(model, x_tmp, num_learned, device)
                elif method == 'nme':    
                    j0 = select_subnetwork_icarl(model, x_tmp, prototypes, num_learned, device)
                
                    
                del x_tmp

                if j0 == task_id:
                    acc_task_classification += x.size(0)

                    set_task(model, j0)

                    pred = model(x.to(device))
                    correct_preds += (pred.argmax(dim=1) == y_true.to(device)).sum().float()

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
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='path to save/load dataset')
    parser.add_argument('--download_data', type=bool, default=True, help='download dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='network architecture to use')
    parser.add_argument('--path_pretrained_model', type=str, default='pretrained_model.pth', help='path to pretrained parameters')
    parser.add_argument('--path_init_params', type=str, default='init_params.pth', help='path to initialization parameters')
    parser.add_argument('--alpha_conv', type=float, default=0.9, help='fraction of importance to keep in conv layers')
    parser.add_argument('--num_tasks', type=int, default=10, help='number of tasks')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--num_classes_per_task', type=int, default=10, help='number of classes per task')
    parser.add_argument('--num_iters', type=int, default=3, help='number of pruning iterations')   # 3
    parser.add_argument('--prune_batch_size', type=int, default=1000, help='number of examples for pruning')
    parser.add_argument('--batch_size', type=int, default=128, help='number of examples per training batch')
    parser.add_argument('--test_batch_size', type=int, default=20, help='number of examples per test batch')
    parser.add_argument('--train_epochs', type=int, default=70, help='number training epochs')      # 70
    parser.add_argument('--retrain_epochs', type=int, default=50, help='number of retraining epochs after pruning') #30
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='multistep', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--decay_epochs_train', nargs='+', type=int, default=[20, 40, 60], help='epochs for multistep decay')  # [20, 40, 60]
    parser.add_argument('--decay_epochs_retrain', nargs='+', type=int, default=[15, 25, 40], help='epochs for multistep decay')   # [15, 25, 40]
    parser.add_argument('--gamma', type=float, default=0.2, help='multiplicative factor of learning rate decay')   # 0.1
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay during retraining')         
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--order_name', type=str, default='default', help='name of class ordering. Options: defult, seed1993, seed1605')
    parser.add_argument('--task_select_method', type=str, default='max', help='task selection method')
    parser.add_argument('--train', type=bool, default=True, help='train the model; if False inference mode only')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TRAIN = args.train
    EVAL = True
    
    path_results = f"./results/{args.dataset_name}-{args.model_name}/{args.num_tasks}_tasks/"
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    
    print("STARTED")
    print(args)

    orders = {
        'default' : [i for i in range(args.num_classes)],    
        'seed1993' : [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
        'seed1605' : [
            47, 23, 18, 97, 84, 49, 98, 26, 0, 14, 40, 85, 29, 72, 1, 48, 35, 52, 3, 2, 92, 7, 80, 32, 6, 19,
            79, 58, 11, 34, 89, 57, 21, 37, 38, 86, 73, 94, 28, 67, 63, 87, 51, 20, 54, 33, 64, 56, 31, 41,
            12, 46, 76, 99, 61, 8, 36, 75, 15, 4, 10, 83, 82, 78, 96, 27, 30, 93, 74, 66, 90, 70, 81, 69,
            5, 65, 13, 25, 88, 17, 71, 60, 44, 68, 95, 59, 45, 53, 50, 43, 55, 22, 24, 9, 39, 62, 16, 77, 91, 42
        ],
        'seed2022' : [
            79, 76, 83,  5, 35, 57, 22, 96, 67, 58, 93,  3, 69, 60, 39, 17, 54, 44, 61, 94, 32, 84, 70, 20, 50, 81, 47, 51,  4, 97, 30, 10,  1, 25,
            65,  7, 26, 31, 82,  6,  9, 28, 62, 63, 89, 34, 95, 66,  8, 40, 90, 59, 36,  0, 68, 77, 46, 43, 78, 73, 21, 74, 85, 29, 71, 64, 91, 42,
            52, 13, 80, 98, 12, 56, 37, 23, 15,  2, 87, 99, 14, 72, 38, 86, 48, 75, 19, 11, 27, 33, 41, 53, 16, 24, 18, 88, 55, 49, 45, 92
        ],
        'seed9999' : [
            83, 13, 57,  5, 31, 44, 50, 55, 89, 74, 73,  1, 35, 15, 23, 34,  7, 65, 93, 49,  0,  4, 66, 40, 97, 17, 18, 52, 98,  2, 80, 72, 82, 85,
            77, 78, 26, 42, 91, 81,  6, 10, 62, 87, 53, 27, 60, 84, 67, 29, 96, 58, 63, 94, 21, 48, 95, 64, 75, 70, 32, 88, 71, 56, 61, 99, 11,  3,
            41, 30, 19,  9, 43, 39, 14, 45, 68, 28,  8, 22, 12, 20, 59, 36, 38, 79, 90, 51, 69, 46, 33, 76, 16, 24, 25, 37, 92, 54, 47, 86
        ]
    }

    order_name = args.order_name
    tasks_order = np.array(orders[order_name])

    if TRAIN:
        task_labels = create_labels(args.num_classes, args.num_tasks, args.num_classes_per_task)
        train_dataset, test_dataset = task_construction(task_labels, args.dataset_name, tasks_order)

        net = init_model(args, device)

        #torch.save(net.state_dict(), args.path_init_params)

        num_tasks = args.num_tasks

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        start_task = 0
        if start_task > 0:
            path_to_save = path_results+'{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, start_task, args.num_classes_per_task, 
                                                                                             args.optimizer, args.alpha_conv, args.num_iters, order_name)
            net.load_state_dict(torch.load(path_to_save, map_location=device) )

            net._load_masks(file_name=path_results+'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, args.model_name, 
                                                                                                              start_task, args.num_classes_per_task, 
                                                                                                              args.optimizer, args.alpha_conv, args.num_iters, order_name), 
                            num_tasks=start_task)

            if start_task < num_tasks:
                net._add_mask(task_id=start_task)



        for task_id in range(start_task, num_tasks):
            path_to_save = path_results + '{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, task_id+1, args.num_classes_per_task, 
                                                                                               args.optimizer, args.alpha_conv, args.num_iters, order_name)
            set_task(net, task_id)
            train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)

            net.set_trainable_masks(task_id)

            net = train(args=args, model=net, train_loader=train_loader, test_loader=test_loader, device=device, task_id=task_id)

            net.eval()
            acc = accuracy(net, test_loader, device)
            print('Accuracy: ', np.round(100*acc, 2))

            random.seed(args.seed)
            np.random.seed(args.seed)
            prune_idx = np.random.permutation(train_dataset[task_id].data.shape[0])[:args.prune_batch_size]

            x_prune = torch.FloatTensor(train_dataset[task_id].data[prune_idx]).to(device)
            x_prune = x_prune.permute(0, 3, 1, 2)
            x_prune = torchvision.transforms.Normalize(mean, std)(x_prune.float()/255)

            net = iterative_pruning(args=args,
                                    net=net, 
                                    train_loader=train_loader, 
                                    test_loader=test_loader,
                                    x_prune=x_prune,
                                    task_id=task_id,
                                    device=device,
                                    path_to_save=path_to_save
                                    )


            net.set_masks_intersection()
            net.set_masks_union()

            torch.save(net.state_dict(), path_to_save)
            net._save_masks(path_results +'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.model_name, task_id+1, args.num_classes_per_task, 
                                                                                                     args.optimizer, args.alpha_conv, args.num_iters, order_name))

            if task_id < num_tasks-1:
                net._add_mask(task_id=task_id+1)
                print('-------------------TASK {}------------------------------'.format(task_id+1))
                
    if EVAL:
        NUM_LEARNED = args.num_tasks
        if not TRAIN:
            path_to_save = path_results+'{}_task{}_{}classes_{}_{}_it{}_order_{}.pth'.format(args.dataset_name, args.model_name, args.model_name, args.num_tasks, 
                                                                                             args.num_classes_per_task, args.optimizer, args.alpha_conv, 
                                                                                             args.num_iters, order_name)
            net.load_state_dict(torch.load(path_to_save, map_location=device) )

            net._load_masks(file_name=path_results+'{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth'.format(args.dataset_name, args.model_name, 
                                                                                                              args.model_name, args.num_tasks, 
                                                                                                              args.num_classes_per_task, args.optimizer, 
                                                                                                              args.alpha_conv, args.num_iters, order_name), 
                            num_tasks=args.num_tasks)
    
    
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
        method = args.task_select_method
               

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