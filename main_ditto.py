# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
import random
from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, get_data_v2
from models.Update import LocalUpdate, LocalUpdateDitto
from models.test import test_img_local_all

import pdb
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Fed_ditto%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        # dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        # for idx in dict_users_train.keys():
        #     np.random.shuffle(dict_users_train[idx])
        if args.is_reset_dataset == 1:
            dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix = get_data_v2(args)

            dutrain = []
            dutest = []
            for k, v in dict_users_train.items():
                dutrain.append(v)
            for k, v in dict_users_test.items():
                dutest.append(v)
            np.save('data/sample/dict_users_train.npy', np.array(dutrain))
            np.save('data/sample/dict_users_test.npy', np.array(dutest))
            np.save('data/sample/concept_matrix.npy', np.array(concept_matrix))
        elif args.is_reset_dataset == 0:
            dataset_train, dataset_test, _, _, _ = get_data_v2(args)
            dutr = np.load('data/sample/dict_users_train.npy', allow_pickle=True)
            dute = np.load('data/sample/dict_users_test.npy', allow_pickle=True)
            concept_matrix = np.load('data/sample/concept_matrix.npy', allow_pickle=True)
            dict_users_train = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
            dict_users_test = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
            for i, v in enumerate(dutr):
                dict_users_train[i] = v
            for i, v in enumerate(dute):
                dict_users_test[i] = v


        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    # 选择公共层
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            if args.model != 'resnet18':
                w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
            else:
                keys = [key for key in net_glob.state_dict().keys()]
                w_glob_keys = [keys[0:-2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'maml':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)


    # generate list of local models for each user
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    net_local = copy.deepcopy(net_glob)
    
    # training
    # concept_matrix = []
    loss_train = []
    test_freq = args.test_freq
    indd = None
    accs = []
    accs10 = 0
    times = []
    lam = args.lam_ditto
    start = time.time()

    #为每一个客户端计算一个概念偏移矩阵
    if args.limit_local_output == 0:
        concept_matrix = []
        for id in range(args.num_users):
            concept_matrix_local = np.array([i for i in range(args.num_classes)], dtype=np.int64)
            if args.is_concept_shift == 1:
                random.shuffle(concept_matrix_local[0:int(args.concept_shift_rate*args.num_classes)])
            concept_matrix.append(concept_matrix_local)

    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        times_in = []
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if iter == args.epochs:
                    local = LocalUpdateDitto(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdateDitto(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if iter == args.epochs:
                    local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_global = copy.deepcopy(net_glob)
            w_glob_k = copy.deepcopy(net_global.state_dict())
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_k, loss, indd = local.train(net=net_global.to(args.device), ind=idx, idx=clients[idx], lr=args.lr, concept_matrix_local=concept_matrix[idx])
            else:
                w_k, loss, indd = local.train(net=net_global.to(args.device), idx=idx, lr=args.lr, concept_matrix_local=concept_matrix[idx])
            net_local = copy.deepcopy(net_glob)
            w_local = copy.deepcopy(w_locals[idx])
            net_local.load_state_dict(w_local)
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx], lr=args.lr, w_ditto=w_glob_k, lam=lam, concept_matrix_local=concept_matrix[idx])
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device),  idx=idx, lr=args.lr, w_ditto=w_glob_k, lam=lam, concept_matrix_local=concept_matrix[idx])

            loss_locals.append(copy.deepcopy(loss))

            if len(w_glob) == 0:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_k[key]/m
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_k[key]/m
                    w_locals[idx][key] = w_local[key]
            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        # get weighted average for global weights
        net_glob.load_state_dict(w_glob)
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False, concept_matrix=concept_matrix)
            accs.append(acc_test)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_ditto_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    base_dir = './save/accs_ditto_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
