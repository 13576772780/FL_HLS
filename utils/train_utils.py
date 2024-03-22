# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang
import copy
import random

from numpy import long
from torchvision import datasets, transforms
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP, CNN_FEMNIST, Resnet_18, ResNet, ResNet18
from utils.sampling import noniid, noniid_v2
import os
import json
import numpy as np

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

        #为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        # concept_matrix = np.array([[ -1 for i in range(10)] for j in range(args.num_users)])
        # for idx, cls in enumerate(rand_set_all):
        #     start = 0
        #     for val in cls:
        #         if concept_matrix[idx][val] == -1:
        #             concept_matrix[idx][val] = start
        #             start += 1

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, None


def get_data_v2(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes,
                                                   nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                  rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1

    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes, nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        # dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        # dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
        #                                        rand_set_all=rand_set_all)
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes,
                                                   nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                  rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1
    else:
        exit('Error: unrecognized dataset')

    if args.level_n_system != 0:
        y_train = np.array(dataset_train.targets)
        y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users_train, rand_set_all)
        dataset_train.targets = np.array(y_train_noisy, dtype='int64')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix

def get_data_v3(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes,
                                                   nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                  rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1

    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes, nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        # dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        # dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
        #                                        rand_set_all=rand_set_all)
        dict_users_train, rand_set_all = noniid_v2(dataset_train, args.num_users, args.shard_per_user, args.num_classes,
                                                   nums_per_class=args.nums_per_class)
        dict_users_test, rand_set_all = noniid_v2(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                  rand_set_all=rand_set_all, nums_per_class=args.nums_per_class)

        # 为了让没个客户端模型只有限定类数量的输出，比如只有3类输出，将每个客户端的类映射到0，1，2.。。。
        concept_matrix = np.array([[-1 for i in range(10)] for j in range(args.num_users)], dtype=np.int64)
        for idx, cls in enumerate(rand_set_all):
            start = 0
            for val in cls:
                if concept_matrix[idx][val] == -1:
                    concept_matrix[idx][val] = start
                    start += 1
    else:
        exit('Error: unrecognized dataset')

    # if args.level_n_system != 0:
    #     y_train = np.array(dataset_train.targets)
    #     y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users_train, rand_set_all)
    #     dataset_train.targets = np.array(y_train_noisy, dtype='int64')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix, rand_set_all


def get_data_from_file(args):

    if 'cifar' in args.dataset or args.dataset == 'mnist':
        if args.is_reset_dataset == 1:
            # dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix = get_data_v2(args)
            dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix, rand_set_all = get_data_v3(
                args)

            dutrain = []
            dutest = []
            for k, v in dict_users_train.items():
                dutrain.append(v)
            for k, v in dict_users_test.items():
                dutest.append(v)
            np.save('data/sample/' + args.data_store_file + '_train.npy', np.array(dutrain))
            np.save('data/sample/' + args.data_store_file + '_test.npy', np.array(dutest))
            np.save('data/sample/' + args.data_store_file + 'dataset_train_target.npy', np.array(dataset_train.targets))
            np.save('data/sample/' + args.data_store_file + 'concept_matrix.npy', np.array(concept_matrix))
            np.save('data/sample/' + args.data_store_file + 'rand_set_all.npy', np.array(rand_set_all))
        elif args.is_reset_dataset == 0:
            dataset_train, dataset_test, _, _, _, _ = get_data_v3(args)
            dutr = np.load('data/sample/' + args.data_store_file + '_train.npy', allow_pickle=True)
            dute = np.load('data/sample/' + args.data_store_file + '_test.npy', allow_pickle=True)
            concept_matrix = np.load('data/sample/' + args.data_store_file + 'concept_matrix.npy', allow_pickle=True)
            rand_set_all = np.load('data/sample/' + args.data_store_file + 'rand_set_all.npy', allow_pickle=True)
            dict_users_train = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
            dict_users_test = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
            for i, v in enumerate(dutr):
                dict_users_train[i] = v
            for i, v in enumerate(dute):
                dict_users_test[i] = v
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

        if args.level_n_system != 0:
            y_train = np.array(dataset_train.targets)
            y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users_train, rand_set_all)
            dataset_train.targets = np.array(y_train_noisy, dtype='int64')
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



    return dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix, rand_set_all


def add_noise2(args, y_train, dict_users, rand_set_all):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        #TODO:看懂，然后让客户端在只有特定类标签时，不会出现新的标签
        if args.limit_local_output == 1 or args.shard_per_user < 10:
            y_train_noisy[sample_idx[noisy_idx]] = rand_set_all[i][np.random.randint(0, args.shard_per_user, len(noisy_idx))]
        else:
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)

def add_noise(args, y_train, dict_users, rand_set_all):
    np.random.seed(args.seed)
    y_train_noisy = copy.deepcopy(y_train)
    users_arr = [i for i in range(0, args.num_users)]
    ##采样客户端
    rand_users_arr = random.sample(users_arr, int(args.num_users*args.level_n_system))

    for i in rand_users_arr:
        sample_idx = list(dict_users[i])
        #采样具体客户端上的样本
        rand_sample_idx = np.array(random.sample(sample_idx, int(len(sample_idx)*args.level_n_lowerb)))

        #随机修改采样样本的标签
        if args.limit_local_output == 1 or args.shard_per_user < 10:
            nosiy_sample_labels = rand_set_all[i][np.random.randint(0, args.shard_per_user, len(rand_sample_idx))]
            y_train_noisy[rand_sample_idx] = nosiy_sample_labels
        else:
            nosiy_sample_labels = np.random.randint(0, 10, len(rand_sample_idx))
            y_train_noisy[rand_sample_idx] = nosiy_sample_labels


        print("   Client %d, noise    level: %.4f " % (i, args.level_n_lowerb))

    return y_train_noisy, None, None

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'resnet18' and 'cifar10' in args.dataset:
        # net_glob = Resnet_18(args=args).to(args.device)
        net_glob = ResNet18(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob


def init_class_center(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
        class_center = np.array([[random.random() for j in range(net_glob.fc3.in_features)] for i in range(10)])
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
        class_center = np.array([[random.random() for j in range(net_glob.fc3.in_features)] for i in range(10)])
    elif args.model == 'resnet18' and 'cifar10' in args.dataset:
        net_glob = ResNet18(args=args).to(args.device)
        class_center = np.array([[random.random() for j in range(net_glob.linear.in_features)] for i in range(10)])
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
        #net_glob.layer_hidden2.out_features
        class_center = np.array([[random.random() for j in range(net_glob.layer_hidden2.out_features)] for i in range(10)])
        # class_center = np.array([[[random.random() for j in range(net_glob.layer_hidden2.out_features)] for i in range(10)] for k in range(args.num_users)])
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return class_center
