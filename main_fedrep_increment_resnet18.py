# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data_v2, get_model, read_data, get_data
from models.Update import LocalUpdate, LocalUpdateIncrement
from models.test import test_img_local_all, test_img_local_all_increment

import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    with open('output.txt', 'a') as f:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FedRep%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        print(
            '# alg: {} , epochs: {}, shard_per_user: {}, limit_local_output: {}, local_rep_ep: {} , local_only: {}, is_concept_shift: {}, dataset: {}  \n'.format(
                args.alg, args.epochs, args.shard_per_user, args.limit_local_output, args.local_rep_ep, args.local_only,
                args.is_concept_shift, args.dataset))

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
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

    print(args.alg)

    # build model
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

    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        with open('output.txt', 'a') as f:
            print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{} \n)'.format(
                num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    with open('output.txt', 'a') as f:
        print("learning rate, batch size: {}, {} \n".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()

    # 为每一个客户端计算一个概念偏移矩阵
    if args.limit_local_output == 0:
        concept_matrix = []
        for id in range(args.num_users):
            concept_matrix_local = np.array([i for i in range(args.num_classes)], dtype=np.int64)
            if args.is_concept_shift == 1:
                random.shuffle(concept_matrix_local)
            concept_matrix.append(concept_matrix_local)

    local_clients = []
    idxs_users = []
    for c in range(args.num_users):

        # 把第c个client加入训练
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            if args.epochs == iter:
                local = LocalUpdateIncrement(args=args,
                                             dataset=dataset_train[list(dataset_train.keys())[c][:args.m_ft]],
                                             idxs=dict_users_train, indd=indd)
            else:
                local = LocalUpdateIncrement(args=args,
                                             dataset=dataset_train[list(dataset_train.keys())[c][:args.m_tr]],
                                             idxs=dict_users_train, indd=indd)
        else:
            if args.epochs == iter:
                local = LocalUpdateIncrement(args=args, dataset=dataset_train, idxs=dict_users_train[c][:args.m_ft])
            else:
                local = LocalUpdateIncrement(args=args, dataset=dataset_train, idxs=dict_users_train[c][:args.m_tr])

        local_clients.append(local)
        idxs_users.append(c)
        print('---------------------------------------------train_client: {} \n'.format(idxs_users))

        # 初始化第c个client
        net_local = copy.deepcopy(net_glob)
        w_local = net_local.state_dict()
        if args.alg != 'fedavg' and args.alg != 'prox':
            for k in w_locals[c].keys():
                if k not in w_glob_keys:
                    w_local[k] = w_locals[c][k]
        net_local.load_state_dict(w_local)
        if c == 0:
            w_local, loss, indd = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr,
                                              concept_matrix_local=concept_matrix[c], first=True, isNew=True,
                                              local_eps=20)
            w_glob_temp = copy.deepcopy(w_local)
            net_glob.load_state_dict(w_glob_temp)
        else:
            w_local, loss, indd = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr,
                                              concept_matrix_local=concept_matrix[c], first=False, isNew=True,
                                              local_eps=20)
        for k, key in enumerate(net_glob.state_dict().keys()):
            w_locals[c][key] = w_local[key]

        # 训练前c个客户端
        for iter in range(args.epochs + 1):

            w_glob = {}
            loss_locals = []
            # 每轮选取的客户端数
            # m = max(int(args.frac * args.num_users), 1)
            # 最后一轮选取所有客户端
            # if iter == args.epochs:
            #     m = args.num_users

            # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # w_keys_epoch = w_glob_keys

            times_in = []
            total_len = 0
            for ind, idx in enumerate(idxs_users):
                start_in = time.time()

                local = local_clients[idx]

                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()

                # 只进行本地训练
                if args.local_only == 1:
                    for k in w_locals[idx].keys():
                        if k not in w_glob_keys:
                            w_local[k] = w_locals[idx][k]

                if args.alg != 'fedavg' and args.alg != 'prox':
                    for k in w_locals[idx].keys():
                        if k not in w_glob_keys:
                            w_local[k] = w_locals[idx][k]

                net_local.load_state_dict(w_local)

                last = iter == args.epochs
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                      w_glob_keys=w_glob_keys, lr=args.lr,
                                                      concept_matrix_local=concept_matrix[idx], first=False,
                                                      isNew=False, local_eps=10, head_eps=5)
                else:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys,
                                                      lr=args.lr, concept_matrix_local=concept_matrix[idx], first=False,
                                                      isNew=False, local_eps=10, head_eps=5)
                loss_locals.append(copy.deepcopy(loss))
                total_len += lens[idx]
                # 保存本地层和全局层
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]
                else:
                    for k, key in enumerate(net_glob.state_dict().keys()):
                        if key in w_glob_keys:
                            w_glob[key] += w_local[key] * lens[idx]
                        else:
                            w_glob[key] += w_local[key] * lens[idx]
                        w_locals[idx][key] = w_local[key]

                times_in.append(time.time() - start_in)

            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # get weighted average for global weights
            for k in net_glob.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)

            w_local = net_glob.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]
            if args.epochs != iter:
                net_glob.load_state_dict(w_glob)

            if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
                if times == []:
                    times.append(max(times_in))
                else:
                    times.append(times[-1] + max(times_in))
                acc_test, loss_test = test_img_local_all_increment(net_glob, args, dataset_test, dict_users_test,
                                                                   w_glob_keys=w_glob_keys, w_locals=w_locals,
                                                                   indd=indd,
                                                                   dataset_train=dataset_train,
                                                                   dict_users_train=dict_users_train, return_all=False,
                                                                   concept_matrix=concept_matrix, num_idxxs=c + 1)
                accs.append(acc_test)
                # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
                if iter != args.epochs:
                    with open('output.txt', 'a') as f:
                        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f} \n'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                    with open('output.txt', 'a') as f:
                        print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f} \n'.format(
                            loss_avg, loss_test, acc_test))
                if iter >= args.epochs - 10 and iter != args.epochs:
                    accs10 += acc_test / 10

                # below prints the global accuracy of the single global model for the relevant algs
                if args.alg == 'fedavg' or args.alg == 'prox':
                    acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                             w_locals=None, indd=indd, dataset_train=dataset_train,
                                                             dict_users_train=dict_users_train, return_all=False,
                                                             concept_matrix=concept_matrix)
                    if iter != args.epochs:
                        with open('output.txt', 'a') as f:
                            print(
                                'Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f} \n'.format(
                                    iter, loss_avg, loss_test, acc_test))
                    else:
                        with open('output.txt', 'a') as f:
                            print(
                                'Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f} \n'.format(
                                    loss_avg, loss_test, acc_test))
                if iter >= args.epochs - 10 and iter != args.epochs:
                    accs10_glob += acc_test / 10

            if iter % args.save_every == args.save_every - 1:
                model_save_path = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(
                    args.num_users) + '_' + str(args.shard_per_user) + '_iter' + str(iter) + '.pt'
                torch.save(net_glob.state_dict(), model_save_path)
    with open('output.txt', 'a') as f:
        print('Average accuracy final 10 rounds: {} \n'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        with open('output.txt', 'a') as f:
            print('Average global accuracy final 10 rounds: {} \n'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
