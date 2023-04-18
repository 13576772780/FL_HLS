# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit: Paul Pu Liang
import random

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import FedProx
from models.test import test_img_local, test_img_local_all
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdateMAML(object):

    def __init__(self, args, dataset=None, idxs=None, optim=None,indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.optim = optim
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1,lr_in=0.0001, c=False):
        net.train()
        # train and update
        lr_in = lr*0.001
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        # MAML code adapted from AntreasAntoniou/HowToTrainYourMAMLPytorch/few_shot_learning_system.py - credit: Antreas Antoniou
        local_eps = self.args.local_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(2)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)

                    split = self.args.local_bs 
                    sup_x, sup_y = data.to(self.args.device), targets.to(self.args.device)
                    targ_x, targ_y = data.to(self.args.device), targets.to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    log_probs_sup = net(sup_x, hidden_train)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]

                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                        
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    split = int(8* images.size()[0]/10)
                    sup_x, sup_y = images[:split].to(self.args.device), labels[:split].to(self.args.device)
                    targ_x, targ_y = images[split:].to(self.args.device), labels[split:].to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    log_probs_sup = net(sup_x)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
                        
                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]
                        
                    loss_sup.backward(retain_graph=True)
                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()
 
                batch_loss.append(loss_sup.item())
                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss_sup.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd#, num_updates


class LocalUpdateScaffold(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1, c=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        local_eps = self.args.local_ep

        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss_fi = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    
                    loss.backward()
                    optimizer.step()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    log_probs = net(images)
                    loss_fi = self.loss_func(log_probs, labels)
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                    optimizer.step()

                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, num_updates

class LocalUpdateAPFL(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net,ind=None,w_local=None, idx=-1, lr=0.1):
        net.train()
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        # train and update
        local_eps = self.args.local_ep
        args = self.args
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates >= self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if  'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    w_loc_new = {}
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    optimizer.zero_grad()
                    wt = copy.deepcopy(net.state_dict())
                    net.zero_grad()

                    del hidden_train
                    hidden_train = net.init_hidden(self.args.local_bs)

                    net.load_state_dict(w_loc_new)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.args.alpha_apfl*self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar
                    
                else:
                        
                    w_loc_new = {} 
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    wt = copy.deepcopy(net.state_dict())

                    net.load_state_dict(w_loc_new)
                    log_probs = net(images)
                    loss = self.args.alpha_apfl*self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar

                num_updates += 1
                if num_updates >= self.args.local_updates:
                    break

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(),w_local, sum(epoch_loss) / len(epoch_loss), self.indd

class LocalUpdateDitto(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net,ind=None, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    w_0 = copy.deepcopy(net.state_dict())
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train) 
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()

                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])

                        net.load_state_dict(w_net)
                        optimizer.zero_grad()
                else:
                    w_0 = copy.deepcopy(net.state_dict())
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                        net.load_state_dict(w_net)
                        optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates >= self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset: 
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
         
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None        
        
        self.dataset=dataset
        self.idxs=idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )

        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                             lr=lr,
                             gmf=self.args.gmf,
                             mu=self.args.mu,
                             ratio=1/self.args.num_users,
                             momentum=0.5,
                             nesterov = False,
                             weight_decay = 1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg =='fedavg' or self.args.alg == 'prox':
                local_eps= 10
                # net_keys = [*net.state_dict().keys()]
                # if 'cifar' in self.args.dataset:
                #     w_glob_keys = [net.weight_keys[i] for i in [0,1,3,4]]
                # elif 'sent140' in self.args.dataset:
                #     w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
                # elif 'mnist' in self.args.dataset:
                #     w_glob_keys = [net.weight_keys[i] for i in [0,1,2]]
            elif 'maml' in self.args.alg:
                local_eps = 10
                w_glob_keys = []
            else:
                local_eps =  max(10,local_eps-self.args.local_rep_ep)
        
        head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # then do local epochs for the representation
            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                     param.requires_grad = True 
       
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    #通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd


class LocalUpdateIncrement(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None, dataset_test=None, dict_users_test=None, client_num = 0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs
        self.dataset_test = dataset_test
        self.dict_users_test = dict_users_test
        self.client_num = client_num

    def train(self, net, w_glob_keys, first=False,isNew=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None, local_eps=10, head_eps=5, last=False):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay':0.0001},
                {'params': bias_p, 'weight_decay':0}
            ],
            lr=lr, momentum=0.5
        )
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(lr=lr, parameters=net.parameters(), weight_decay=0.005, moment=0.5)
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        # local_eps = self.args.local_ep
        # if last:
        #     if self.args.alg == 'fedavg' or self.args.alg == 'prox':
        #         local_eps = 10
        #         net_keys = [*net.state_dict().keys()]
        #         if 'cifar' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
        #         elif 'sent140' in self.args.dataset:
        #             w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        #         elif 'mnist' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0, 1, 2]]
        #     elif 'maml' in self.args.alg:
        #         local_eps = 10
        #         w_glob_keys = []
        #     else:
        #         local_eps = max(10, local_eps - self.args.local_rep_ep)

        # head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False
            #如果是第1个客户端，训练全部层
            if first:
                for name, param in net.named_parameters():
                    param.requires_grad = True
            # for FedRep, first do local epochs for the head
            elif (iter < head_eps and self.args.alg == 'fedrep') or last or isNew:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif (iter >= head_eps and self.args.alg == 'fedrep'):
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    # 通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))


            if isNew == False and iter == head_eps - 1:
                net_local = copy.deepcopy(net)
                test_accuracy, test_loss = test_img_local(net_local, self.dataset_test, self.args, idxs=self.dict_users_test, concept_matrix_local= concept_matrix_local)
                train_accuracy, train_loss = test_img_local(net_local, self.dataset, self.args, idxs=self.idxs, concept_matrix_local= concept_matrix_local)

                print('        train local model (freeze embeding):client {:3d},  Train loss: {:.3f}, Train accuracy: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f} \n'.format(
                            self.client_num, train_loss, train_accuracy, test_loss, test_accuracy))

            if isNew == False and iter == local_eps - 1:
                net_local = copy.deepcopy(net)
                test_accuracy, test_loss = test_img_local(net_local, self.dataset_test, self.args, idxs=self.dict_users_test, concept_matrix_local= concept_matrix_local)
                train_accuracy, train_loss = test_img_local(net_local, self.dataset, self.args, idxs=self.idxs, concept_matrix_local= concept_matrix_local)
                print('        train local model (unfreeze embeding):client {:3d},  Train loss: {:.3f}, Train accuracy: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f} \n'.format(
                        self.client_num, train_loss, train_accuracy, test_loss, test_accuracy))

            if isNew == True and iter == local_eps - 1:
                net_local = copy.deepcopy(net)
                test_accuracy, test_loss = test_img_local(net_local, self.dataset_test, self.args, idxs=self.dict_users_test, concept_matrix_local= concept_matrix_local)
                train_accuracy, train_loss = test_img_local(net_local, self.dataset, self.args, idxs=self.idxs, concept_matrix_local= concept_matrix_local)
                print(
                    '        init --> train local model(freeze embeding):client {:3d},  Train loss: {:.3f}, Train accuracy: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f} \n'.format(
                        self.client_num, train_loss, train_accuracy, test_loss, test_accuracy))

        return net.state_dict(), train_loss, self.indd

class LocalUpdateIncrementResnet18(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, w_glob_keys, first=False,isNew=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None, local_eps=10, head_eps=2):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        # local_eps = self.args.local_ep
        # if last:
        #     if self.args.alg == 'fedavg' or self.args.alg == 'prox':
        #         local_eps = 10
        #         net_keys = [*net.state_dict().keys()]
        #         if 'cifar' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
        #         elif 'sent140' in self.args.dataset:
        #             w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        #         elif 'mnist' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0, 1, 2]]
        #     elif 'maml' in self.args.alg:
        #         local_eps = 10
        #         w_glob_keys = []
        #     else:
        #         local_eps = max(10, local_eps - self.args.local_rep_ep)

        # head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False
            #如果是第1个客户端，训练全部层
            if first:
                for name, param in net.named_parameters():
                    if name not in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True
            # for FedRep, first do local epochs for the head
            elif (iter < head_eps and self.args.alg == 'fedrep') or isNew:
                for name, param in net.named_parameters():
                    if name not in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif (iter >= head_eps and self.args.alg == 'fedrep'):
                for name, param in net.named_parameters():
                    if name not in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    # 通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

class LocalUpdatePAC(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs
        self.features = None


    def train(self, net, w_glob_keys, class_center_glob, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
                elif 'mnist' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 2]]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    #通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])


                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()

                    #如果要更新特征提取层，则需要考虑质心
                    if iter >= head_eps and self.args.alg == 'fedrep' and not last:
                        #传入的hook是一个回调函数，每次前向传播会调用该函数，函数内部可以把值存储到数组中，方便后面处理
                        #获取特征值
                        if self.args.model == "mlp":
                            net.layer_hidden2.register_forward_hook(self.hook)
                        elif self.args.model == "cnn":
                            net.fc2.register_forward_hook(self.hook)
                        elif self.args.model == "resnet18":
                            net.linear.register_forward_hook(self.hook_input)
                        log_probs = net(images)

                        #计算正则项
                        class_center_batch = np.array([class_center_glob[i] for i in labels])
                        if self.args.model == "resnet18":
                            self.features = self.features[0]
                        sub_clc = self.features.to(self.args.device) - torch.from_numpy(class_center_batch).to(self.args.device)
                        reg_loss = torch.mean(torch.square(sub_clc)) * self.args.pac_param
                        loss = self.loss_func(log_probs, labels) + reg_loss
                    else:
                        log_probs = net(images)
                        loss = self.loss_func(log_probs, labels)

                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        #计算本地客户端的类的质心
        class_center_local = np.zeros(class_center_glob.shape)
        class_num = np.zeros(class_center_glob.shape[0])
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                labels = torch.tensor(concept_matrix_local[labels.numpy()])
            if 'sent140' in self.args.dataset:
                pass
            else:
                images = images.to(self.args.device)
                net.zero_grad()
                # 获取特征值
                if self.args.model == "mlp":
                    net.layer_hidden2.register_forward_hook(self.hook)
                elif self.args.model == "cnn":
                    net.fc2.register_forward_hook(self.hook)
                elif self.args.model == "resnet18":
                    net.linear.register_forward_hook(self.hook_input)
                net(images)
                if self.args.model == "resnet18":
                    self.features = self.features[0]
                # self.features = self.features.to(self.args.device)
                featrue = self.features.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for idx, cls in enumerate(labels):
                    class_center_local[cls] += featrue[idx]
                    class_num[cls] += 1

        # for idx, cln in enumerate(class_num):
        #     if cln > 0:
        #         class_center_local[idx] = class_center_local[idx] / cln

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, class_center_local, class_num


    def hook(self, module, input, output):
        self.features = output

        return None
    def hook_input(self, module, input, output):
        self.features = input

        return None


class LocalUpdatePACKMEANS(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs
        self.features = None

    def train(self, net, w_glob_keys, class_center_glob, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1,
              concept_matrix_local=None):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
                elif 'mnist' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 2]]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    # 通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()

                    # 如果要更新特征提取层，则需要考虑质心
                    if iter >= head_eps and self.args.alg == 'fedrep' and not last:
                        # 传入的hook是一个回调函数，每次前向传播会调用该函数，函数内部可以把值存储到数组中，方便后面处理
                        # 获取特征值
                        if self.args.model == "mlp":
                            net.layer_hidden2.register_forward_hook(self.hook)
                        elif self.args.model == "cnn":
                            net.fc2.register_forward_hook(self.hook)
                        log_probs = net(images)

                        # 计算正则项
                        class_center_batch = np.array([class_center_glob[i] for i in labels])
                        sub_clc = self.features - torch.from_numpy(class_center_batch)
                        reg_loss = torch.mean(torch.square(sub_clc)) * self.args.pac_param
                        loss = self.loss_func(log_probs, labels) + reg_loss
                    else:
                        log_probs = net(images)
                        loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 计算本地客户端的类的质心
        class_center_local = np.zeros(class_center_glob.shape)
        class_num = np.zeros(class_center_glob.shape[0])
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                labels = torch.tensor(concept_matrix_local[labels.numpy()])
            if 'sent140' in self.args.dataset:
                pass
            else:
                images = images.to(self.args.device)
                net.zero_grad()
                # 获取特征值
                if self.args.model == "mlp":
                    net.layer_hidden2.register_forward_hook(self.hook)
                elif self.args.model == "cnn":
                    net.fc2.register_forward_hook(self.hook)
                net(images)
                featrue = self.features.detach().numpy()
                labels = labels.detach().numpy()
                for idx, cls in enumerate(labels):
                    class_center_local[cls] += featrue[idx]
                    class_num[cls] += 1

        for idx, cln in enumerate(class_num):
            if cln > 0:
                class_center_local[idx] = class_center_local[idx] / cln

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, class_center_local, class_num

    def hook(self, module, input, output):
        self.features = output
        return None
    def hook_v2(self, module, input, output):
        self.features = input
        return None

class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None,indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        else:
            self.indd=indd

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )

        epoch_loss = []
        local_eps = self.args.local_ep
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    W = W_glob.clone()
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i+1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0])+1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()
                
                else:
                
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    W = W_glob.clone().to(self.args.device)
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i+1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0])+1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd
