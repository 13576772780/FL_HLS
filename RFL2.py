import copy
import numpy as np
import torch
import time
import torch.nn.functional as F
from utils.train_utils import get_data, get_model, read_data, init_class_center, get_data_v2, get_data_from_file
from torch import nn
from utils.logger import Logger
from utils.options import args_parser
from utils.train import get_local_update_objects, FedAvg


# def RFL(args):
    # f_save = open(args.save_dir + args.txtname + '_acc.txt', 'a')
    ##############################
    #  Load Dataset
    ##############################
def call_bn(bn, x):
    return bn(x)
def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    n_total = len(data_loader.dataset)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs, _ = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).float().sum().item()

    test_loss /= n_total
    accuracy = 100.0 * correct / n_total

    return accuracy, test_loss


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        feature = h.view(h.size(0), h.size(1))

        logit = self.l_c1(feature)

        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)

        return logit, feature

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

    dataset_train, dataset_test, dict_users_train, dict_users_test, concept_matrix, rand_set_all = get_data_from_file(
        args)


    #替换代码的数据获取部分，后面的代码为原代码逻辑
    dict_users = dict_users_train
    y_train = dataset_train.targets

    start = time.time()
    # dataset_train, dataset_test, dict_users, y_train, gamma_s, _ = load_data_with_noisy_label(args)

    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)

    if args.dataset == 'mnist':
        input_channel = 1
    else:
        input_channel = 3
    net_glob = CNN(input_channel=input_channel)
    # net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)

    ##############################
    # Training
    ##############################
    logger = Logger(args)

    forget_rate_schedule = []

    forget_rate = args.forget_rate
    exponent = 1
    forget_rate_schedule = np.ones(args.epochs) * forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)

    # Initialize f_G
    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        net_glob=net_glob,
    )

    for epoch in range(args.epochs):
        local_losses = []
        local_weights = []
        f_locals = []
        args.g_epoch = epoch

        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args

            w, loss, f_k = local.train(copy.deepcopy(net_glob).to(args.device), copy.deepcopy(f_G).to(args.device),
                                       client_num)

            f_locals.append(f_k)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(local_weights)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Update f_G
        sim = torch.nn.CosineSimilarity(dim=1)
        tmp = 0
        w_sum = 0
        for i in f_locals:
            sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
            w_sum += sim_weight
            tmp += sim_weight * i
        f_G = torch.div(tmp, w_sum)

        # logging
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss, )

        print('Round {:3d}'.format(epoch))
        print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()]))

        # logger.write(epoch=epoch + 1, **results)

    # logger.close()

    print("time :", time.time() - start)


