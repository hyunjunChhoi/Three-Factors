# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet
from torch.utils.data import DataLoader, Subset
from models.resnet_con import ResNet18, ResNet34
if __package__ is None:
    import sys
    from os import path
    import random
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.tinyimages_300k import TinyImages
    from utils.validation_dataset import validation_split
    from utils.ImbalanceCIFAR import IMBALANCECIFAR10
    from utils.ImbalanceCIFAR import IMBALANCECIFAR100
parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv', help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/D_all/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/pretrained', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='OE', help='OE|energy')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--num_ood_samples', default=300000, type=int, help='Number of OOD samples to use.')
parser.add_argument('--imbalance_ratio', type=float, default=0.01, help='Imbalance ratio small is big imbal')


parser.add_argument('--trial', type=int, default=1, help='trial one, two three')

parser.add_argument('--loss_mode', type=int, default=1, help='loss_mode (1:Contra_base 2:Contra_base_ours 3:Contra_cider 4:Contra_cider_ours 5:Contra_balsc 6:Contra_balsc_ours')
parser.add_argument('--beta1', type=float, default=1.00, help='Contrastive loss weight')
parser.add_argument('--beta2', type=float, default=1.00, help='Our regularize loss weight')

parser.add_argument('--dstemp', type=float, default=4.0, help='distillation temperature scale factor')
parser.add_argument('--pool_size', type=int, default=30000, help='pooling sample size in total 300000 ')
parser.add_argument('--step_size', type=int, default=1, help='step size 1 -> select all . step size 2 -> select odd or even ')
parser.add_argument('--quantile', type=float, default=0.6, help='what qunatile to sample from Hardness')

parser.add_argument('--n', type=int, default=2, help='how many times to boost augment')

args = parser.parse_args()


if args.score == 'OE':
    save_info = 'oecc_tune'
elif args.score == 'energy':
    save_info = 'energy_ft'

args.save = args.save+save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

#torch.manual_seed(1)
#np.random.seed(args.seed)
seed_it(args.trial)

step_size=args.step_size

class Contra_base_origin(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_base_origin, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        #targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        #targets = torch.cat([targets.repeat(1, 1), targets_centers], dim=0)
        #batch_cls_count = torch.eye(self.num_classes)[targets.long()].sum(dim=0).squeeze()
        #mask = torch.eq(targets[:batch_size], targets.T).float().to(device)
        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        #features = torch.cat([features, centers1], dim=0)
        logits = features.mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #    batch_size , batch_size) - mask
        #exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        mean_log_prob_pos = mean_log_prob_pos[~torch.isnan(mean_log_prob_pos)]


        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

class Contra_base(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_base, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(1, 1), targets_centers], dim=0)
        #batch_cls_count = torch.eye(self.num_classes)[targets.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets[:batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features = torch.cat([features, centers1], dim=0)
        logits = features[:batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #    batch_size , batch_size) - mask
        #exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss

class Contra_base_ours(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_base_ours, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, features_ood, pseudo_target_ood ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ##### including OOD in contrastive #########
        ### OOD label as just negative sample in contrastive loss ######
        target_ood= torch.ones((features_ood.size(0),1), device=device)*self.num_classes

        ######################

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets_all = torch.cat([targets.repeat(1, 1), targets_centers, target_ood], dim=0)
        #target_id= torch.cat([targets.repeat(1, 1), targets_centers], dim=0)
        #batch_cls_count = torch.eye(self.num_classes)[targets_id.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets_all[:batch_size], targets_all.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits = features_all[:batch_size].mm(features_all.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #    batch_size , batch_size) - mask
        #exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        ####### ours regularize loss(deprecated) #########

        #features_ood, pseudo_target_ood
        mask_c = torch.eq(targets_centers , pseudo_target_ood.T).float().to(device)
        features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits_c = centers1.mm(features_ood.T)
        mask_logits_c = logits_c * mask_c
        loss_c=mask_logits_c.mean(1).mean(0)
        ##############
        logits_c_angle=torch.acos(logits_c)*180/np.pi
        mask_logits_c_angle=logits_c_angle * mask_c
        total_sum_mask_c=mask_c.sum(1).sum(0)
        mean_angle=mask_logits_c_angle.sum(1).sum(0)/total_sum_mask_c
        ##### mean angle of OOD #####

        return loss

class Contra_balsc(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_balsc, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(1, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(self.num_classes)[targets.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets[:batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features = torch.cat([features, centers1], dim=0)
        logits = features[:batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #print(exp_logits.size())
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            batch_size , batch_size+self.num_classes) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss

class Contra_balsc_ours(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_balsc_ours, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, features_ood, pseudo_target_ood):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ##### including OOD in contrastive #########
        target_ood= torch.ones((features_ood.size(0),1), device=device)*self.num_classes

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets_all = torch.cat([targets.repeat(1, 1), targets_centers, target_ood], dim=0)
        pseudo_target_ood_unsq=torch.unsqueeze(pseudo_target_ood,dim=1)
        #print(targets.size())
        #print(pseudo_target_ood.size())
        targets_all_class = torch.cat([targets.repeat(1, 1), targets_centers, pseudo_target_ood_unsq], dim=0)

        #targets_id= torch.cat([targets.repeat(1, 1), targets_centers], dim=0)

        batch_cls_count = torch.eye(self.num_classes)[targets_all_class.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets_all[:batch_size], targets_all.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits = features_all[:batch_size].mm(features_all.T)
        logits = torch.div(logits, self.temperature)
        #print(logits.size())
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging\
        #print(logits.size())
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets_all_class], device=device).view(1, -1).expand(
            batch_size, targets_all.size(0)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        ####### ours regularize loss (deprecated) #########
        mask_c = torch.eq(targets_centers , pseudo_target_ood.T).float().to(device)
        features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits_c = centers1.mm(features_ood.T)
        mask_logits_c = logits_c * mask_c
        loss_c=mask_logits_c.mean(1).mean(0)
        ##############
        logits_c_angle=torch.acos(logits_c)*180/np.pi
        mask_logits_c_angle=logits_c_angle * mask_c
        total_sum_mask_c=mask_c.sum(1).sum(0)
        mean_angle=mask_logits_c_angle.sum(1).sum(0)/total_sum_mask_c
        ##### mean angle of OOD #####


        return loss

class Contra_cider(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_cider, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, features_ood, pseudo_target_ood ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ##### including OOD in contrastive #########
        ### OOD label as just negative sample in contrastive loss ######
        target_ood= torch.ones((features_ood.size(0),1), device=device)*self.num_classes

        ######################

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets_all = torch.cat([targets.repeat(1, 1), targets_centers, target_ood], dim=0)
        target_id= torch.cat([targets.repeat(1, 1), targets_centers], dim=0)
        #batch_cls_count = torch.eye(self.num_classes)[targets_id.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets_all[:batch_size], targets_all.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits = features_all[:batch_size].mm(features_all.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #    batch_size , batch_size) - mask
        #exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        ####### ours regularize loss (deprecated) #########

        #features_ood, pseudo_target_ood
        mask_c = torch.eq(targets_centers , pseudo_target_ood.T).float().to(device)
        #features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits_c = centers1.mm(features_ood.T)
        mask_logits_c = logits_c * mask_c
        loss_c=mask_logits_c.mean(1).mean(0)

        mask_cc=(1-torch.eq(targets_centers, pseudo_target_ood.T).float()).to(device)
        logits_cc= torch.div(torch.matmul(centers1,features_ood.T), 1.0)

        mean_prob_neg_cc=torch.log((mask_cc*torch.exp(logits_cc)).sum(1)/mask_cc.sum(1))
        mean_prob_neg_cc=mean_prob_neg_cc[~torch.isnan(mean_prob_neg_cc)]
        loss_cc=mean_prob_neg_cc.mean()

        dis_mask=(1-torch.eq(targets_centers, targets_centers.T).float()).to(device)

        dis_logits=torch.div(torch.matmul(centers1,centers1.T), self.temperature)

        dis_logits_mask=torch.scatter(torch.ones_like(dis_mask),1,torch.arange(num_classes).view(-1,1).cuda(),0)

        dis_mask=dis_mask*dis_logits_mask
        mean_prob_neg=torch.log((dis_mask*torch.exp(dis_logits)).sum(1)/dis_mask.sum(1))
        mean_prob_neg=mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss_d=mean_prob_neg.mean()

        ##############
        logits_c_angle=torch.acos(logits_c)*180/np.pi
        mask_logits_c_angle=logits_c_angle * mask_c
        total_sum_mask_c=mask_c.sum(1).sum(0)
        mean_angle=mask_logits_c_angle.sum(1).sum(0)/total_sum_mask_c
        #print(mean_angle)
        ##### mean angle of OOD #####


        prototypes = centers1
        proxy_labels = targets_centers
        labels = targets.repeat(1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda()

        feat_dot_prototype = torch.div(torch.matmul(features, prototypes.T), self.temperature)

        logits_max_ci, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits_ci = feat_dot_prototype - logits_max_ci.detach()

        exp_logits_ci = torch.exp(logits_ci)
        log_prob_ci = logits_ci - torch.log(exp_logits_ci.sum(1, keepdim=True))

        mean_log_prob_pros_ci = (mask * log_prob_ci).sum(1)
        loss_ci_comp = -mean_log_prob_pros_ci.mean()


        return loss_ci_comp , loss_d

class Contra_cider_ours(nn.Module):
    def __init__(self, num_classes=None, temperature=0.1):
        super(Contra_cider_ours, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, centers1, features, targets, features_ood, pseudo_target_ood ):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        ##### including OOD in contrastive #########
        ### OOD label as just negative sample in contrastive loss ######
        target_ood= torch.ones((features_ood.size(0),1), device=device)*self.num_classes

        ######################

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets_all = torch.cat([targets.repeat(1, 1), targets_centers, target_ood], dim=0)


        #batch_cls_count = torch.eye(self.num_classes)[targets_id.long()].sum(dim=0).squeeze()
        mask = torch.eq(targets_all[:batch_size], targets_all.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        features_all = torch.cat([features, centers1, features_ood], dim=0)

        logits = features_all[:batch_size].mm(features_all.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        #per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #    batch_size , batch_size) - mask
        #exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits.sum(1,keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        ####### ours regularize loss (deprecated) #########

        #features_ood, pseudo_target_ood


        mask_c = torch.eq(targets_centers , pseudo_target_ood.T).float().to(device)
        #features_all = torch.cat([features, centers1, features_ood], dim=0)
        logits_c = centers1.mm(features_ood.T)
        mask_logits_c = logits_c * mask_c
        loss_c=mask_logits_c.mean(1).mean(0)

        mask_cc=(1-torch.eq(targets_centers, pseudo_target_ood.T).float()).to(device)
        logits_cc= torch.div(torch.matmul(centers1,features_ood.T), 1.0)

        mean_prob_neg_cc=torch.log((mask_cc*torch.exp(logits_cc)).sum(1)/mask_cc.sum(1))
        mean_prob_neg_cc=mean_prob_neg_cc[~torch.isnan(mean_prob_neg_cc)]
        loss_cc=mean_prob_neg_cc.mean()



        dis_mask=(1-torch.eq(targets_centers, targets_centers.T).float()).to(device)

        dis_logits=torch.div(torch.matmul(centers1,centers1.T), self.temperature)

        dis_logits_mask=torch.scatter(torch.ones_like(dis_mask),1,torch.arange(num_classes).view(-1,1).cuda(),0)

        dis_mask=dis_mask*dis_logits_mask
        mean_prob_neg=torch.log((dis_mask*torch.exp(dis_logits)).sum(1)/dis_mask.sum(1))
        mean_prob_neg=mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss_d=mean_prob_neg.mean()

        ##############
        logits_c_angle=torch.acos(logits_c)*180/np.pi
        mask_logits_c_angle=logits_c_angle * mask_c
        total_sum_mask_c=mask_c.sum(1).sum(0)
        mean_angle=mask_logits_c_angle.sum(1).sum(0)/total_sum_mask_c
        #print(mean_angle)
        ##### mean angle of OOD #####

        '''
        prototypes = centers1
        proxy_labels = targets_centers
        labels = targets_all.repeat(1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda()

        feat_dot_prototype = torch.div(torch.matmul(features_all, prototypes.T), self.temperature)

        logits_max_ci, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits_ci = feat_dot_prototype - logits_max_ci.detach()

        exp_logits_ci = torch.exp(logits_ci)
        log_prob_ci = logits_ci - torch.log(exp_logits_ci.sum(1, keepdim=True))

        mean_log_prob_pros_ci = (mask * log_prob_ci).sum(1)
        loss_ci_comp = -mean_log_prob_pros_ci.mean()
        '''

        return loss , loss_d

class MultiCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, times):
        self.transform = transform
        self.times= times

    def __call__(self, x):
        return [self.transform(x)]*self.times

n=args.n

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
#test_transform = trn.Compose([trn.ToTensor()])
if args.dataset == 'cifar10':
    train_data_in= IMBALANCECIFAR10(phase="train", imbalance_ratio=args.imbalance_ratio,root = './data/cifar10',imb_type='exp', transform=MultiCropTransform(train_transform,n))
    test_data = dset.CIFAR10('./data/cifar10', train=False, transform=test_transform)
    num_classes = 10
    prior_gamma1=torch.tensor([0.249973333,0.079566667,0.163056667,0.248683333,0.061536667,0.085046667,0.030543333,0.02319,0.023856667,0.034546667]).cuda()**0.75
    prior_gamma2=torch.tensor([0.249973333,0.079566667,0.163056667,0.248683333,0.061536667,0.085046667,0.030543333,0.02319,0.023856667,0.034546667]).cuda()**0.75
    prior_gamma1=F.normalize(prior_gamma1,p=1.0, dim=0)
    prior_gamma2=F.normalize(prior_gamma2,p=1.0, dim=0)
    prior_gamma1=prior_gamma1[None,:]
    prior_gamma2=prior_gamma2[None,:]
    load_prior='imbalance_'+args.dataset+'.npy'
    all_ood_conf=np.load(load_prior)
    indices = np.argsort(all_ood_conf)
    N = len(all_ood_conf)
    selected_indices = indices[int(args.quantile*N):int(args.quantile*N)+args.pool_size*step_size:step_size]
    if args.quantile<0:
        selected_indices=range(30000)
else:
    train_data_in= IMBALANCECIFAR100(phase="train", imbalance_ratio=args.imbalance_ratio,root = './data/cifar100',imb_type='exp', transform=MultiCropTransform(train_transform,n))
    test_data = dset.CIFAR100('./data/cifar100', train=False, transform=test_transform, download=True)
    num_classes = 100
    prior_gamma1=torch.tensor([0.007576667,0.016876667,0.06155,0.025433333,0.01779,0.036716667,0.02814,0.010246667,0.018733333,0.037023333,0.02976,0.061663333,0.034796667,0.04198,0.01685,0.01611,0.065586667,0.023993333,0.025203333,0.016963333,0.00781,0.007296667,0.025533333,0.01836,0.00327,0.027816667,0.00909,0.01173,0.00772,0.009206667,0.008083333,0.007,0.0093,0.023793333,0.00379,0.01991,0.004013333,0.016356667,0.00619,0.017383333,0.015876667,0.00644,0.00791,0.002533333,0.006926667,0.007936667,0.019663333,0.004403333,0.003353333,0.008743333,0.002523333,0.00493,0.001876667,0.002783333,0.005783333,0.00105,0.00204,0.001306667,0.002516667,0.002376667,0.00707,0.002233333,0.00221,0.000673333,0.00308,0.001453333,0.00082,0.000533333,0.00305,0.001696667,0.001936667,0.00322,0.000446667,0.000943333,0.000263333,0.000323333,0.00214,0.000133333,0.00159,0.000826667,0.00021,0.00189,0.00105,0.000486667,0.001353333,0.000356667,0.001096667,0.003323333,0.000406667,0.002836667,0.000726667,0.00007,9.33E-05,0.00002,0.002866667,0.00016,0.000243333,5.33E-05,0.000403333,0.00009]).cuda()**0.75
    prior_gamma2=torch.tensor([0.007576667,0.016876667,0.06155,0.025433333,0.01779,0.036716667,0.02814,0.010246667,0.018733333,0.037023333,0.02976,0.061663333,0.034796667,0.04198,0.01685,0.01611,0.065586667,0.023993333,0.025203333,0.016963333,0.00781,0.007296667,0.025533333,0.01836,0.00327,0.027816667,0.00909,0.01173,0.00772,0.009206667,0.008083333,0.007,0.0093,0.023793333,0.00379,0.01991,0.004013333,0.016356667,0.00619,0.017383333,0.015876667,0.00644,0.00791,0.002533333,0.006926667,0.007936667,0.019663333,0.004403333,0.003353333,0.008743333,0.002523333,0.00493,0.001876667,0.002783333,0.005783333,0.00105,0.00204,0.001306667,0.002516667,0.002376667,0.00707,0.002233333,0.00221,0.000673333,0.00308,0.001453333,0.00082,0.000533333,0.00305,0.001696667,0.001936667,0.00322,0.000446667,0.000943333,0.000263333,0.000323333,0.00214,0.000133333,0.00159,0.000826667,0.00021,0.00189,0.00105,0.000486667,0.001353333,0.000356667,0.001096667,0.003323333,0.000406667,0.002836667,0.000726667,0.00007,9.33E-05,0.00002,0.002866667,0.00016,0.000243333,5.33E-05,0.000403333,0.00009]).cuda()**0.75
    prior_gamma1=F.normalize(prior_gamma1,p=1.0, dim=0)
    prior_gamma2=F.normalize(prior_gamma2,p=1.0, dim=0)
    prior_gamma1=prior_gamma1[None,:]
    prior_gamma2=prior_gamma2[None,:]
    load_prior='imbalance_'+args.dataset+'.npy'
    all_ood_conf=np.load(load_prior)
    indices = np.argsort(all_ood_conf)
    N = len(all_ood_conf)
    selected_indices = indices[int(args.quantile*N):int(args.quantile*N)+args.pool_size*step_size:step_size]
    if args.quantile<0:
        selected_indices=range(30000)

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

#ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))

ood_data = Subset(TinyImages(root='./data', transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
     trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])), list(selected_indices))
ood_data_inf = TinyImages(root='./data', transform=test_transform)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out_all_inf = torch.utils.data.DataLoader(
    ood_data_inf,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

test_loader_out = torch.utils.data.DataLoader(
    ood_data_inf,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
# Create model
net = ResNet18(num_classes=num_classes, return_features=False)
model = ResNet18(num_classes=num_classes, return_features=False)

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module
# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        
        model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                  '_pretrained_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name), strict=False)
            model.load_state_dict(torch.load(model_name), strict=False)

            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    model = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    model.cuda()

    #torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////
def influence(value, weight, dim=None, keepdim=False):

    influence= torch.sum(value*F.relu(weight), dim=dim, keepdim=keepdim)
    #influence = influence.squeeze(dim)

    return influence

def train():
    net.train()  # enter train mode
    model.eval()  # enter train mode


    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for (in_set, labels, _), (out_set, _) in zip(train_loader_in, train_loader_out):
        in_set_all=in_set[0]
        target_all=labels
        for i in range (n-1):
            in_set_all= torch.cat([in_set_all, in_set[i+1]],dim=0)
            target_all= torch.cat([target_all, labels],dim=0)
        #print(in_set_all.size())
        #print(out_set.size())
        data = torch.cat((in_set_all, out_set), 0)
        target=target_all


        data, target = data.cuda(), target.cuda()


        x, projected_f, center = net.forward_all(data)
        x_mom = model(data[:labels.size(0)])
        # backward
        scheduler.step()
        optimizer.zero_grad()
        temp=args.dstemp

        #loss = F.cross_entropy(x[:len(in_set[0])], target)
        loss = F.cross_entropy(x[:labels.size(0)], target[:labels.size(0)])
        dist_loss = F.kl_div(F.log_softmax(x[:labels.size(0)] / temp, dim=1),
                        F.softmax(x_mom[:labels.size(0)]/ temp, dim=1),
                        reduction='batchmean') * temp * temp
        loss+=dist_loss

        if args.dataset == 'cifar10':
            A_tr = 0.7328  # Training accuracy of CIFAR-10 baseline model
        else:
            A_tr = 0.4022  # Training accuracy of CIFAR-100 baseline model
        sm = torch.nn.Softmax(dim=1)  # Create a Softmax
        probabilities = sm(x)  # Get the probabilites for both In and Outlier Images
        max_probs, _ = torch.max(probabilities, dim=1)  # Take the maximum probabilities produced by softmax
        lambda_1 = 0.03
        lambda_2 = 0.03


        x_out=x[in_set_all.size(0):]
        _, pseudo_label_out=torch.max(x_out, dim=1)
        f_out= projected_f[in_set_all.size(0):]

        OOD_regularize_loss=0

        if args.loss_mode==1:
            Contra_criterion=Contra_base(num_classes=num_classes, temperature=0.1)
            Contra_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target)
        elif args.loss_mode==2:
            Contra_criterion=Contra_base_ours(num_classes=num_classes, temperature=0.1)
            Contra_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target, f_out, pseudo_label_out)
        elif args.loss_mode==3:
            Contra_criterion=Contra_cider(num_classes=num_classes, temperature=0.1)
            Contra_loss, reg_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target, f_out, pseudo_label_out)
            OOD_regularize_loss+=reg_loss
        elif args.loss_mode==4:
            Contra_criterion=Contra_cider_ours(num_classes=num_classes, temperature=0.1)
            Contra_loss, reg_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target, f_out, pseudo_label_out)
            OOD_regularize_loss+=reg_loss
        elif args.loss_mode==5:
            Contra_criterion=Contra_balsc(num_classes=num_classes, temperature=0.1)
            Contra_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target)
        elif args.loss_mode==6:
            Contra_criterion=Contra_balsc_ours(num_classes=num_classes, temperature=0.1)
            Contra_loss=Contra_criterion(center, projected_f[:in_set_all.size(0)], target)

        loss+=args.beta1*Contra_loss
        loss+=args.beta2*OOD_regularize_loss

        # cross-entropy from softmax distribution to uniform distribution
        # cross-entropy from softmax distribution to uniform distribution
        softmax=F.softmax(x,dim=1)
        influence_gamma1=influence(softmax,prior_gamma1,dim=1)
        influence_gamma2=influence(softmax,prior_gamma2,dim=1)

        Ec_out_dist=influence_gamma1[in_set_all.size(0):]
        Ec_out_weight=influence_gamma2[in_set_all.size(0):]
        if args.score == 'energy':
            Ec_out = -torch.logsumexp(x[in_set_all.size(0):], dim=1)
            Ec_out = Ec_out - (args.lamb * Ec_out_dist)
            Ec_in = -torch.logsumexp(x[:labels.size(0)], dim=1)
            loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + (torch.pow(F.relu((args.m_out - Ec_out)), 2)*Ec_out_weight).sum()/Ec_out_weight.sum())
        elif args.score == 'OE':
            #loss += 0.5 * -(x[in_set_all.size(0):].mean(1) - torch.logsumexp(x[in_set_all.size(0):], dim=1)).mean()
            prob_diff_in = max_probs[:labels.size(0)] - A_tr  # Push towards the training accuracy of the baseline
            loss += lambda_1 * torch.sum(prob_diff_in ** 2)  ## 1st Regularization term
            prob_diff_out = probabilities[in_set_all.size(0):][:] - (1 / num_classes)
            loss += lambda_2 * torch.sum(torch.abs(prob_diff_out))  ## 2nd Regularization term

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info+'_dstemp_'+str(args.dstemp)+ '_quantile_' + str(args.quantile) +'_loss_mode_'+str(args.loss_mode)+'_times_'+str(args.n)+ '_beta_' + str(args.beta1)+ str(args.beta2)+ '_CC_trial_' + str(
    args.trial) + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()


    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch)+'_dstemp_'+str(args.dstemp)+ '_quantile_' + str(args.quantile)+'_loss_mode_'+str(args.loss_mode)+'_times_'+str(args.n)+ '_beta_' + str(args.beta1)+ str(args.beta2) + '_CC_trial_' + str(
                   args.trial) + '.pt'))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                             '_' + save_info + '_epoch_' + str(epoch - 1)+'_dstemp_'+str(args.dstemp)+ '_quantile_' + str(args.quantile) +'_loss_mode_'+str(args.loss_mode)+'_times_'+str(args.n)+ '_beta_' + str(args.beta1)+ str(args.beta2) + '_CC_trial_' + str(args.trial) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info+'_dstemp_'+str(args.dstemp)+ '_quantile_' + str(args.quantile) +'_loss_mode_'+str(args.loss_mode)+'_times_'+str(args.n)+ '_beta_' + str(args.beta1)+ str(args.beta2) + '_CC_trial_' + str(
        args.trial) + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
