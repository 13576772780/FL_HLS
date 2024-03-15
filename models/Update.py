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
from torch.optim import Optimizer
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

    def train(self, net, c_list={}, idx=-1, lr=0.1, c=False, w_glob_keys=[], concept_matrix_local=None):
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
                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    #通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

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

    def train(self, net,ind=None,w_local=None, idx=-1, lr=0.1, w_glob_keys=[], concept_matrix_local=None):
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
                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    #通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

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

    def train(self, net,ind=None, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, w_glob_keys=[], concept_matrix_local=None):
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

        args = self.args 
        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    #通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])

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
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    # optimizer.zero_grad()
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


class CoteanchingLocalUpdate(object):
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
        self.last_net = None

    def filter_data(self, net, net2, concept_matrix_local=None):

        filter_idxs1 = []
        filter_idxs2 = []
        distance_net1 = {}
        distance_net2 = {}
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]
            lable_tmp = torch.from_numpy(np.array([lable_tmp])).to(torch.int64)
            data_tmp, lable_tmp = data_tmp.to(self.args.device), lable_tmp.to(self.args.device)

            net2.zero_grad()
            log_probs2 = net2(data_tmp)
            loss2 = self.loss_func(log_probs2, lable_tmp)
            distance_net2[data_idx] = loss2.item()

            net.zero_grad()
            log_probs = net(data_tmp)
            loss = self.loss_func(log_probs, lable_tmp)
            # if (loss2 < 60):
            #     filter_idxs2.append(data_idx)
            distance_net1[data_idx] = loss.item()

        sort_distance_tmp1 = sorted(distance_net1.items(), key=lambda x: x[1])
        sort_distance_tmp2 = sorted(distance_net2.items(), key=lambda x: x[1])

        filter_idxs1 = [sort_distance_tmp1[i][0] for i in range(math.floor(self.args.shard_per_user * self.args.nums_per_class * 0.9))]
        filter_idxs2 = [sort_distance_tmp2[i][0] for i in range(math.floor(self.args.shard_per_user * self.args.nums_per_class * 0.9))]

        random.shuffle(filter_idxs1)
        random.shuffle(filter_idxs2)

        self.ldr_train = DataLoader(DatasetSplit(self.dataset, filter_idxs2), batch_size=self.args.local_bs,
                                          shuffle=True)
        self.ldr_train2 = DataLoader(DatasetSplit(self.dataset, filter_idxs1), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None):
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


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']
class LocalUpdatePFedMe(object):
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

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None, w_locals = None):
        bias_p = []
        weight_p = []
        optimizer = pFedMeOptimizer(net.parameters(), lr=self.args.personal_learning_rate, lamda=self.args.lamda)
        optimizer.zero_grad()
        local_model = copy.deepcopy(net)
        w_local = local_model.state_dict()
        for k in w_locals[idx].keys():
            if k not in w_glob_keys:
                w_local[k] = w_locals[idx][k]
        local_param = copy.deepcopy(list(local_model.parameters()))
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # optimizer = torch.optim.SGD(
        #     [
        #         {'params': weight_p, 'weight_decay': 0.0001},
        #         {'params': bias_p, 'weight_decay': 0}
        #     ],
        #     lr=lr, momentum=0.5
        # )

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
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # # for FedRep, first do local epochs for the head
            # if (iter < head_eps and self.args.alg == 'fedrep') or last:
            #     for name, param in net.named_parameters():
            #         if name in w_glob_keys:
            #             param.requires_grad = False
            #         else:
            #             param.requires_grad = True
            #
            # # then do local epochs for the representation
            # elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
            #     for name, param in net.named_parameters():
            #         if name in w_glob_keys:
            #             param.requires_grad = True
            #         else:
            #             param.requires_grad = False
            #
            # # all other methods update all parameters simultaneously
            # elif self.args.alg != 'fedrep':
            #     for name, param in net.named_parameters():
            #         param.requires_grad = True

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
                    # optimizer.step()
                    persionalized_model_bar, _ = optimizer.step(local_param)

                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            for new_param, localweight in zip(persionalized_model_bar, local_param):
                localweight.data = localweight.data - self.args.lamda* self.args.learning_rate * (localweight.data - new_param.data)


            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        for param, new_param in zip(net.parameters(), local_param):
            param.data = new_param.data.clone()
        # w_locals[idx] = local_model
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
                images, labels = images.to(self.args.device), labels.to(self.args.device)
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

class LocalUpdatePACCoTeaching(object):
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
        self.ldr_train2 = self.ldr_train

    def filter_data(self, net, net2, concept_matrix_local=None):

        filter_idxs1 = []
        filter_idxs2 = []
        distance_net1 = {}
        distance_net2 = {}
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]
            lable_tmp = torch.from_numpy(np.array([lable_tmp])).to(torch.int64)
            data_tmp, lable_tmp = data_tmp.to(self.args.device), lable_tmp.to(self.args.device)

            net2.zero_grad()
            log_probs2 = net2(data_tmp)
            loss2 = self.loss_func(log_probs2, lable_tmp)
            distance_net2[data_idx] = loss2.item()

            net.zero_grad()
            log_probs = net(data_tmp)
            loss = self.loss_func(log_probs, lable_tmp)
            # if (loss2 < 60):
            #     filter_idxs2.append(data_idx)
            distance_net1[data_idx] = loss.item()

        sort_distance_tmp1 = sorted(distance_net1.items(), key=lambda x: x[1])
        sort_distance_tmp2 = sorted(distance_net2.items(), key=lambda x: x[1])

        filter_idxs1 = [sort_distance_tmp1[i][0] for i in range(math.floor(self.args.shard_per_user * self.args.nums_per_class * 0.9))]
        filter_idxs2 = [sort_distance_tmp2[i][0] for i in range(math.floor(self.args.shard_per_user * self.args.nums_per_class * 0.9))]

        random.shuffle(filter_idxs1)
        random.shuffle(filter_idxs2)

        self.ldr_train = DataLoader(DatasetSplit(self.dataset, filter_idxs2), batch_size=self.args.local_bs,
                                          shuffle=True)
        self.ldr_train2 = DataLoader(DatasetSplit(self.dataset, filter_idxs1), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train(self, net , w_glob_keys, class_center_glob, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None, is_teacher=0):
        bias_p = []
        weight_p = []
        if is_teacher == 0:
            self.ldr_train_local = self.ldr_train
        else:
            self.ldr_train_local = self.ldr_train2

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
            for batch_idx, (images, labels) in enumerate(self.ldr_train_local):

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
        # for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #     if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
        #         # 通过概念偏移矩阵进行标签概念偏移
        #         labels = torch.tensor(concept_matrix_local[labels.numpy()])
        #     if 'sent140' in self.args.dataset:
        #         pass
        #     else:
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         # 获取特征值
        #         if self.args.model == "mlp":
        #             net.layer_hidden2.register_forward_hook(self.hook)
        #         elif self.args.model == "cnn":
        #             net.fc2.register_forward_hook(self.hook)
        #         elif self.args.model == "resnet18":
        #             net.linear.register_forward_hook(self.hook_input)
        #         net(images)
        #         if self.args.model == "resnet18":
        #             self.features = self.features[0]
        #         # self.features = self.features.to(self.args.device)
        #         featrue = self.features.detach().cpu().numpy()
        #         labels = labels.detach().cpu().numpy()
        #         for idx, cls in enumerate(labels):
        #             class_center_local[cls] += featrue[idx]
        #             class_num[cls] += 1

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

class LocalUpdatePACPSL(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None, rand_set_all = None):
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
        self.ldr_train_local = self.ldr_train
        self.rand_set_all = rand_set_all
        self.targets = self.dataset.targets

    def filter_by_center(self, net, concept_matrix_local, local_class_center, iter_num, local_eps):
        # 在每轮训练前，先根据类心筛选数据
        center_distance = {}
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]

            data_tmp = data_tmp.to(self.args.device)
            net.zero_grad()
            # 获取特征值
            if self.args.model == "mlp":
                net.layer_hidden2.register_forward_hook(self.hook)
            elif self.args.model == "cnn":
                net.fc2.register_forward_hook(self.hook)
            elif self.args.model == "resnet18":
                net.linear.register_forward_hook(self.hook_input)
            net(data_tmp)
            if self.args.model == "resnet18":
                self.features = self.features[0]
            # self.features = self.features.to(self.args.device)
            featrue_tmp = self.features[0].detach().cpu().numpy()
            distance_tmp = np.sqrt(np.sum(np.square(local_class_center[lable_tmp] - featrue_tmp)))
            # distance_tmp = np.dot(local_class_center[lable_tmp], featrue_tmp) / (np.linalg.norm(local_class_center[lable_tmp]) * np.linalg.norm(featrue_tmp))
            # if distance_tmp > 800 :
            #     continue
            if lable_tmp not in center_distance.keys():
                center_distance[lable_tmp] = {}
            center_distance[lable_tmp][data_idx] = distance_tmp

        # 对每类样本的距离进行排序
        filter_idxs = []
        for d_key in center_distance.keys():
            sort_center_distance_tmp = sorted(center_distance[d_key].items(), key=lambda x: x[1])
            d_count = 0
            for (s_d_key, s_d_val) in sort_center_distance_tmp:
                filter_idxs.append(s_d_key)
                d_count += 1
                if d_count >= (self.args.nums_per_class * (iter_num + 1)/ local_eps):
                    break
                # if d_count >= (self.args.nums_per_class * (iter_num + 1)  / 3):
                # if d_count >= (self.args.nums_per_class * 0.7):
                #     break
        random.shuffle(filter_idxs)

        self.ldr_train_local = DataLoader(DatasetSplit(self.dataset, filter_idxs), batch_size=self.args.local_bs,
                                          shuffle=True)

    def modify_label_by_center(self, net, concept_matrix_local, local_class_center, iter_num, local_eps, class_set=None):
        # 在每轮训练前，先根据类心筛选数据
        center_distance = {}
        self.targets = np.copy(self.dataset.targets)
        targets = self.dataset.targets
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]

            data_tmp = data_tmp.to(self.args.device)
            net.zero_grad()
            # 获取特征值
            if self.args.model == "mlp":
                net.layer_hidden2.register_forward_hook(self.hook)
            elif self.args.model == "cnn":
                net.fc2.register_forward_hook(self.hook)
            elif self.args.model == "resnet18":
                net.linear.register_forward_hook(self.hook_input)
            net(data_tmp)
            if self.args.model == "resnet18":
                self.features = self.features[0]
            # self.features = self.features.to(self.args.device)
            featrue_tmp = self.features[0].detach().cpu().numpy()
            # new_label = class_set[np.argmin([np.sqrt(np.sum(np.square(local_class_center[i] - featrue_tmp))) for i in class_set])]
            new_label = class_set[
                np.argmin([np.dot(local_class_center[i], featrue_tmp) / (np.linalg.norm(local_class_center[i]) * np.linalg.norm(featrue_tmp)) for i in class_set])]
            if new_label != lable_tmp:
                targets[data_idx] = new_label
            # distance_tmp = np.dot(local_class_center[lable_tmp], featrue_tmp) / (np.linalg.norm(local_class_center[lable_tmp]) * np.linalg.norm(featrue_tmp))
            # if distance_tmp > 800 :
            #     continue
            # if lable_tmp not in center_distance.keys():
            #     center_distance[lable_tmp] = {}
            # center_distance[lable_tmp][data_idx] = distance_tmp

        # 对每类样本的距离进行排序
        # filter_idxs = []
        # for d_key in center_distance.keys():
        #     sort_center_distance_tmp = sorted(center_distance[d_key].items(), key=lambda x: x[1])
        #     d_count = 0
        #     for (s_d_key, s_d_val) in sort_center_distance_tmp:
        #         filter_idxs.append(s_d_key)
        #         d_count += 1
        #         # if d_count >= (self.args.nums_per_class * (iter_num + 1) * 0.9 / local_eps):
        #         # if d_count >= (self.args.nums_per_class * (iter_num + 1)  / 3):
        #         if d_count >= (self.args.nums_per_class * 0.7):
        #             break
        # random.shuffle(filter_idxs)

        self.ldr_train_local = DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.args.local_bs,
                                          shuffle=True)

    def filter_by_loss(self, net, concept_matrix_local, iter_num, local_eps):

        distance_net = {}
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]
            lable_tmp = torch.from_numpy(np.array([lable_tmp])).to(torch.int64)
            data_tmp, lable_tmp = data_tmp.to(self.args.device), lable_tmp.to(self.args.device)

            net.zero_grad()
            log_probs = net(data_tmp)
            loss = self.loss_func(log_probs, lable_tmp)
            # if (loss2 < 60):
            #     filter_idxs2.append(data_idx)
            distance_net[data_idx] = loss.item()

        sort_distance_tmp = sorted(distance_net.items(), key=lambda x: x[1])

        filter_idxs = [sort_distance_tmp[i][0] for i in
                        range(math.floor(self.args.shard_per_user * self.args.nums_per_class * (iter_num+1) / local_eps))]

        random.shuffle(filter_idxs)

        self.ldr_train_local = DataLoader(DatasetSplit(self.dataset, filter_idxs), batch_size=self.args.local_bs,
                                    shuffle=True)



    def filter_by_loss2(self, net, concept_matrix_local, iter_num, local_eps):
        # 在每轮训练前，先根据类心筛选数据
        center_distance = {}
        for data_idx in self.idxs:
            data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                # 通过概念偏移矩阵进行标签概念偏移
                # labels = torch.tensor(concept_matrix_local[labels.numpy()])
                lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            else:
                lable_tmp = self.dataset.targets[data_idx]
            lable_tmp = torch.from_numpy(np.array([lable_tmp])).to(torch.int64).to(self.args.device)
            data_tmp = data_tmp.to(self.args.device)
            log_probs = net(data_tmp)

            loss = self.loss_func(log_probs, lable_tmp)
            # if (loss2 < 60):
            #     filter_idxs2.append(data_idx)
            if lable_tmp not in center_distance.keys():
                center_distance[lable_tmp] = {}
            center_distance[lable_tmp][data_idx] = loss.item()

        # 对每类样本的距离进行排序
        filter_idxs = []
        for d_key in center_distance.keys():
            sort_center_distance_tmp = sorted(center_distance[d_key].items(), key=lambda x: x[1])
            d_count = 0
            for (s_d_key, s_d_val) in sort_center_distance_tmp:
                filter_idxs.append(s_d_key)
                d_count += 1
                if d_count >= (self.args.nums_per_class * (iter_num + 1)/ local_eps):
                    break
                # if d_count >= (self.args.nums_per_class * (iter_num + 1)  / 3):
                # if d_count >= (self.args.nums_per_class * 0.7):
                #     break
        random.shuffle(filter_idxs)

        self.ldr_train_local = DataLoader(DatasetSplit(self.dataset, filter_idxs), batch_size=self.args.local_bs,
                                          shuffle=True)



    def train(self, net, w_glob_keys, class_center_glob, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1, concept_matrix_local=None, iter_num_now = 0, train_iter = 0):
        bias_p = []
        weight_p = []

        local_class_center = class_center_glob.copy()
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

        data_train_frag = local_eps
        for iter in range(2):
            done = False

            #在每轮训练前，先根据类心筛选数据
            # center_distance = {}
            # for data_idx in self.idxs:
            #     data_tmp = torch.from_numpy(np.array([self.dataset.data[data_idx].reshape(3, 32, 32)])).to(torch.float32)
            #     if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
            #         #通过概念偏移矩阵进行标签概念偏移
            #         # labels = torch.tensor(concept_matrix_local[labels.numpy()])
            #         lable_tmp = concept_matrix_local[self.dataset.targets[data_idx]]
            #     else:
            #         lable_tmp = self.dataset.targets[data_idx]
            #
            #     data_tmp = data_tmp.to(self.args.device)
            #     net.zero_grad()
            #     # 获取特征值
            #     if self.args.model == "mlp":
            #         net.layer_hidden2.register_forward_hook(self.hook)
            #     elif self.args.model == "cnn":
            #         net.fc2.register_forward_hook(self.hook)
            #     elif self.args.model == "resnet18":
            #         net.linear.register_forward_hook(self.hook_input)
            #     net(data_tmp)
            #     if self.args.model == "resnet18":
            #         self.features = self.features[0]
            #     # self.features = self.features.to(self.args.device)
            #     featrue_tmp = self.features[0].detach().cpu().numpy()
            #     distance_tmp = np.sqrt(np.sum(np.square(local_class_center[lable_tmp] - featrue_tmp)))
            #     if lable_tmp not in center_distance.keys():
            #         center_distance[lable_tmp] = {}
            #     center_distance[lable_tmp][data_idx] = distance_tmp
            #
            # #对每类样本的距离进行排序
            # filter_idxs = []
            # for d_key in center_distance.keys():
            #     sort_center_distance_tmp = sorted(center_distance[d_key].items(), key=lambda x: x[1])
            #     d_count = 0
            #     for (s_d_key, s_d_val) in sort_center_distance_tmp:
            #         filter_idxs.append(s_d_key)
            #         d_count += 1
            #         if d_count >= (self.args.nums_per_class * (iter+1) * 0.9 / local_eps) :
            #             break
            #
            # self.ldr_train_local = DataLoader(DatasetSplit(self.dataset, filter_idxs), batch_size=self.args.local_bs, shuffle=True)
            # if self.args.filter_alg == 'center_psl' : #and iter_num_now > 15
            #     self.filter_by_center(net=net, concept_matrix_local=concept_matrix_local, local_class_center=local_class_center, iter_num=iter, local_eps = 1)
            #     # self.modify_label_by_center(net=net, concept_matrix_local=concept_matrix_local,
            #     #                       local_class_center=local_class_center, iter_num=iter, local_eps=local_eps, class_set=self.rand_set_all[ind])
            # elif self.args.filter_alg == 'loss_psl':
            #     self.filter_by_loss(net=net, concept_matrix_local=concept_matrix_local,iter_num=iter, local_eps = local_eps)
            # else:
            #     self.ldr_train_local = self.ldr_train

            for iter2 in range(local_eps):
                # done = False
                # for FedRep, first do local epochs for the head
                if (iter2 < head_eps and self.args.alg == 'fedrep') or last:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                # then do local epochs for the representation
                elif iter2 >= head_eps and self.args.alg == 'fedrep' and not last:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    #开始训练先不过滤数据
                    if train_iter > 20:
                        if self.args.filter_alg == 'center_psl':  # and iter_num_now > 15
                            self.filter_by_center(net=net, concept_matrix_local=concept_matrix_local,
                                                  local_class_center=local_class_center, iter_num=iter2, local_eps=local_eps)
                            # self.modify_label_by_center(net=net, concept_matrix_local=concept_matrix_local,
                            #                       local_class_center=local_class_center, iter_num=iter, local_eps=local_eps, class_set=self.rand_set_all[ind])
                        elif self.args.filter_alg == 'loss_psl':
                            self.filter_by_loss2(net=net, concept_matrix_local=concept_matrix_local, iter_num=iter2,
                                                local_eps=local_eps)
                        else:
                            self.ldr_train_local = self.ldr_train

                # all other methods update all parameters simultaneously
                elif self.args.alg != 'fedrep':
                    for name, param in net.named_parameters():
                        param.requires_grad = True

                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train_local):

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
                        if iter2 >= head_eps and self.args.alg == 'fedrep' and not last:
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
                            class_center_batch = np.array([local_class_center[i] for i in labels])
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
            class_center_local = np.zeros(local_class_center.shape)
            class_num = np.zeros(local_class_center.shape[0])
            for batch_idx, (images, labels) in enumerate(self.ldr_train_local):
                if self.args.is_concept_shift == 1 or self.args.limit_local_output == 1:
                    # 通过概念偏移矩阵进行标签概念偏移
                    labels = torch.tensor(concept_matrix_local[labels.numpy()])
                if 'sent140' in self.args.dataset:
                    pass
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
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
            # for c_idx, c_val in enumerate(class_center_local):
            #     if class_num[c_idx] > 0:
            #         local_class_center[c_idx] = local_class_center[c_idx] * 0.5 + (class_center_local[c_idx] / class_num[c_idx]) * 0.5


        # for idx, cln in enumerate(class_num):
        #     if cln > 0:
        #         class_center_local[idx] = class_center_local[idx] / cln
        ##恢复修改的标签
        self.dataset.targets = self.targets
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
                        sub_clc = self.features.to(self.args.device) - torch.from_numpy(class_center_batch).to(
                            self.args.device)
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
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                # 获取特征值
                if self.args.model == "mlp":
                    net.layer_hidden2.register_forward_hook(self.hook)
                elif self.args.model == "cnn":
                    net.fc2.register_forward_hook(self.hook)
                net(images)
                featrue = self.features.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
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





class LocalUpdatePACCoTeaching2(object):
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
                images, labels = images.to(self.args.device), labels.to(self.args.device)
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


class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs

        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_tmp = DataLoader(DatasetSplit(dataset, idxs), batch_size=1, shuffle=True)

    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)

        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(
            mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))

        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if self.args.g_epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl

        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)

    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    # def train(self, net, f_G, client_num):
    def train(self, net, w_glob_keys, f_G, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1,
                  concept_matrix_local=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=0)

        epoch_loss = []

        net.eval()
        f_k = torch.zeros(self.args.num_classes, 64, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)

        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1

        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        class_num = np.zeros(f_G.shape[0])
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)

                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask[small_loss_idxs]) * \
                             self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)

                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (
                            self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, f_k, class_num