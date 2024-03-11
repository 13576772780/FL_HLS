import copy
import numpy as np
import torch
from numpy import long

from model.build_model import build_model
# from utils.logger import Logger
from util.aggregation import FedAvg
from util.load_data import load_data_with_noisy_label
from util.local_training import globaltest, get_local_update_objects
import time
# from metrics import cal_fscore
from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, init_class_center, get_data_v2, get_data_from_file

# def RFL(args):
    # f_save = open(args.save_dir + args.txtname + '_acc.txt', 'a')
    ##############################
    #  Load Dataset
    ##############################

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # with open('output.txt', 'a') as f:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RFL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    print(
        '# alg: {} , epochs: {}, shard_per_user: {}, limit_local_output: {}, local_rep_ep: {} , local_only: {}, is_concept_shift: {}, dataset: {}  \n'.format(
            args.alg, args.epochs, args.shard_per_user, args.limit_local_output, args.local_rep_ep, args.local_only,
            args.is_concept_shift, args.dataset))

    # if args.is_reset_dataset == 1:
    #     dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix = get_data_v2(args)
    #     # dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix = get_data(args)
    #
    #     dutrain = []
    #     dutest = []
    #     for k, v in dict_users_train.items():
    #         dutrain.append(v)
    #     for k, v in dict_users_test.items():
    #         dutest.append(v)
    #     np.save('data/sample/dict_users_train.npy', np.array(dutrain))
    #     np.save('data/sample/dict_users_test.npy', np.array(dutest))
    #     np.save('data/sample/dataset_train_target.npy', np.array(dataset_train.targets))
    #     np.save('data/sample/concept_matrix.npy', np.array(concept_matrix))
    # elif args.is_reset_dataset == 0:
    #     dataset_train, dataset_test, _, _, _ = get_data_v2(args)
    #     dutr = np.load('data/sample/dict_users_train.npy', allow_pickle=True)
    #     dute = np.load('data/sample/dict_users_test.npy', allow_pickle=True)
    #     # dataset_train.targets = np.load('data/sample/dataset_train_target.npy', allow_pickle=True)
    #     concept_matrix = np.load('data/sample/concept_matrix.npy', allow_pickle=True)
    #     dict_users_train = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    #     dict_users_test = dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    #     for i, v in enumerate(dutr):
    #         dict_users_train[i] = v
    #     for i, v in enumerate(dute):
    #         dict_users_test[i] = v
    dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix, rand_set_all = get_data_from_file(
        args)


    #替换代码的数据获取部分，后面的代码为原代码逻辑
    dict_users = dict_users_train
    y_train = dataset_train.targets



    # dataset_train, dataset_test, dict_users, y_train, gamma_s, _ = load_data_with_noisy_label(args)

    start = time.time()
    ##############################
    # Build model    ##############################
    net_glob = build_model(args)
    print(net_glob)

    ##############################
    # Training
    ##############################
    # logger = Logger(args)

    forget_rate_schedule = []

    forget_rate = args.forget_rate
    exponent = 1
    forget_rate_schedule = np.ones(args.rounds2) * forget_rate
    if args.rounds2 >= args.num_gradual:
        forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)
    else:
        forget_rate_schedule[:args.rounds2] = np.linspace(0, forget_rate ** exponent, args.rounds2)
    # Initialize f_G
    f_G = torch.randn(args.num_classes, net_glob.fc1.in_features, device=args.device)

    m = max(int(args.frac2 * args.num_users), 1)
    prob = [1 / args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        loss_locals = []
        w_locals = []
        f_locals = []
        args.g_epoch = rnd

        if (rnd + 1) in args.schedule:
            # print("Learning Rate Decay Epoch {}".format(rnd + 1))
            # print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[rnd]

        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        # Local Update
        for client_num, idx in enumerate(idxs_users):
            # Initialize local update objects
            local_update_objects = get_local_update_objects(
                args=args,
                dataset_train=dataset_train,
                dict_users=dict_users,
                net_glob=net_glob,
            )
            local = local_update_objects[idx]
            local.args = args

            w_local, loss_local, f_k = local.train(copy.deepcopy(net_glob).to(args.device), copy.deepcopy(f_G).to(args.device),
                                       client_num)

            f_locals.append(f_k)
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        # update global weights
        w_glob_fl = FedAvg(w_locals, dict_len)
        # copy weight to net_glob
        net_glob.load_state_dict(copy.deepcopy(w_glob_fl))

        # Update f_G
        sim = torch.nn.CosineSimilarity(dim=1)
        tmp = 0
        w_sum = 0
        for i in f_locals:
            sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
            w_sum += sim_weight
            tmp += sim_weight * i
        f_G = torch.div(tmp, w_sum)

        acc_s2 = globaltest(net_glob.to(args.device), dataset_test, args)
        show_info_test_acc = "Round %d global test acc  %.4f" % (rnd, acc_s2)
        print(show_info_test_acc)
        #f_save.write(show_info_test_acc)
        #f_save.flush()

        f_scores = []
        all_idxs_users = [i for i in range(args.num_users)]
        # if rnd == args.rounds2 - 1:
            # for idx in all_idxs_users:
                # f_scores.append(cal_fscore(args, net_glob.to(args.device), dataset_train, y_train, dict_users[idx]))
        show_info_Fscore = "Round %d Fscore \n" % (rnd)
        # print(show_info_Fscore)
        # print(str(f_scores))
        # file_name = "./fscore_" + args.algorithm + ".txt"
        # f = open(file_name, "w")
        # f.writelines(f_scores.to)
        # f.close()

    show_time_info = f"time : {time.time() - start}"
    # print(show_time_info)
    #f_save.write(show_time_info)
    #f_save.flush()
    #     # logging
    #     train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
    #     test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
    #     results = dict(train_acc=train_acc, train_loss=train_loss,
    #                    test_acc=test_acc, test_loss=test_loss, )
    #
    #     print('Round {:3d}'.format(rnd))
    #     print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()]))
    #
    #     logger.write(epoch=rnd + 1, **results)
    #
    # logger.close()