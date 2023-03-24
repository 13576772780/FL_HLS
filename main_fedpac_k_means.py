# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
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
from sklearn.cluster import KMeans
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, init_class_center, get_data_v2
from models.Update import LocalUpdate, LocalUpdatePAC, LocalUpdatePACKMEANS
from models.test import test_img_local_all

import time


# num_classes 全局类有多少个，决定了聚合后类的簇的数量
# class_nums 每个客户端上传的类中心对应的样本的数量，做为计算全局类中心的权重
def get_class_center_k_means(class_center_locals, args, class_nums, class_center_glob, class_center_glob_num):

    class_center_locals.append(class_center_glob)
    class_nums.append(class_center_glob_num)
    # 展平数组，去除为0的点，为0表示客户端没有该类数据
    class_centers_without_zero = []
    class_centers_nums = []
    for idx, cln in enumerate(class_nums):
        for child_idx ,num in enumerate(cln):
            if num != 0:
                class_centers_without_zero.append(class_center_locals[idx][child_idx])
                class_centers_nums.append(num)

    class_centers_nums = np.array(class_centers_nums)
    class_centers_without_zero = np.array(class_centers_without_zero)

    model = KMeans(n_clusters=args.num_classes)
    model.fit(class_centers_without_zero)
    class_labels = model.predict(class_centers_without_zero)

    # 计算全局类中心点
    class_center_grob = np.zeros([args.num_classes, class_centers_without_zero[0].shape[0]])
    # 每个类样本的个数
    class_sample_num_grob = np.zeros([args.num_classes])
    for idx, clc in enumerate(class_centers_without_zero):
        class_center_grob[class_labels[idx]] += clc * class_centers_nums[idx]
        class_sample_num_grob[class_labels[idx]] += class_centers_nums[idx]

    for idx, clc in enumerate(class_center_grob):
        class_center_grob[idx] = clc / class_sample_num_grob[idx]

    # # 让返回给客户端的全局类中心和客户端的类中心对齐
    # cur_idx = 0
    # for idx, cln in enumerate(class_nums):
    #     for child_idx ,num in enumerate(cln):
    #         if num != 0:
    #             class_center_locals[idx][child_idx] = class_center_grob[class_labels[cur_idx]]
    #             cur_idx += 1


    return class_center_grob, class_sample_num_grob

# class ClassCenterGrob:

def get_fit_grob_center(class_center_glob, class_center_local):

    new_class_center_local = np.zeros(class_center_local.shape)
    for idx, ccl in enumerate(class_center_local):
        min_idx = np.argmin(np.square(class_center_glob - ccl))
        new_class_center_local[idx] = class_center_glob[min_idx]

    return new_class_center_local





if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    with open('output.txt', 'a') as f:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FedPAC-K-Means%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        print('# alg: {} , epochs: {}, shard_per_user: {}, limit_local_output: {}, local_rep_ep: {} , local_only: {}, is_concept_shift: {}, dataset: {}  \n'.format(
                args.alg, args.epochs, args.shard_per_user, args.limit_local_output, args.local_rep_ep, args.local_only,
                args.is_concept_shift, args.dataset))

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix = get_data_v2(args)
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
    #选择公共层
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,6,7]]
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
            print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    with open('output.txt', 'a') as f:
        print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()

    #初始化每个类的质心和每个质心计算时都数据量，用于增量聚合
    class_center_glob = init_class_center(args)
    class_center_glob_num = np.zeros(class_center_glob.shape[0])

    #TODO：这里需要考虑，是否设置为随机初始几类不同的概念偏移矩阵，而不是每个客户端都不一样
    #为每一个客户端计算一个概念偏移矩阵
    if args.limit_local_output == 0:
        concept_matrix = []
        for id in range(args.num_users):
            concept_matrix_local = np.array([i for i in range(args.num_classes)], dtype=np.int64)
            if args.is_concept_shift == 1:
                random.shuffle(concept_matrix_local)
            concept_matrix.append(concept_matrix_local)


    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []
        class_center_locals = []
        class_nums=[]
        #用户的历史类中心
        user_history_class_center={}
        #每轮选取的客户端数
        m = max(int(args.frac * args.num_users), 1)
        #最后一轮选取所有客户端
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdatePACKMEANS(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdatePACKMEANS(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdatePACKMEANS(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdatePACKMEANS(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])


            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs

            #在每次训练前，根据客户端的历史类中心选择全局类中心,如果没有历史数据，随机初始化
            if idx in user_history_class_center.keys():
                class_center_grob_local = get_fit_grob_center(class_center_glob, user_history_class_center[idx])
            else:
                class_center_grob_local = np.array([[random.random() for i in range(class_center_glob.shape[1]) ] for j in range(class_center_glob.shape[0])])
                # class_center_grob_local = class_center_glob

            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd, class_center_local, class_num = local.train(net=net_local.to(args.device), class_center_glob=class_center_grob_local, ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, lr=args.lr,last=last, concept_matrix_local=concept_matrix[idx])
            else:
                w_local, loss, indd, class_center_local, class_num = local.train(net=net_local.to(args.device), class_center_glob=class_center_grob_local, idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last, concept_matrix_local=concept_matrix[idx])
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]

            user_history_class_center[idx] = class_center_local
            #收集所有客户端的类中心和每个类中心对应的样本数量，用于后面聚类
            class_center_locals.append(class_center_local)
            class_nums.append(class_num)

            #保存本地层和全局层
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]
                    else:
                        w_glob[key] += w_local[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        #计算全局中心点
        # for cli, clv in enumerate(class_nums):
        #     if clv > 0:
        #         class_center_glob[cli] = class_center_locals[cli] / clv

        #聚类，并获得每个客户端聚类后属于的类中心点
        class_center_glob, class_center_glob_num = get_class_center_k_means(class_center_locals=class_center_locals, args=args, class_nums=class_nums,
                                                                            class_center_glob=class_center_glob, class_center_glob_num=class_center_glob_num)


        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False, concept_matrix=concept_matrix)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                with open('output.txt', 'a') as f:
                    print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                with open('output.txt', 'a') as f:
                    print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False, concept_matrix=concept_matrix)
                if iter != args.epochs:
                    with open('output.txt', 'a') as f:
                        print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
                else:
                    with open('output.txt', 'a') as f:
                        print('Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)
    with open('output.txt', 'a') as f:
        print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        with open('output.txt', 'a') as f:
            print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)




