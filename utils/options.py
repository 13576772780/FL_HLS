#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=-1, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    parser.add_argument('--alg', type=str, default='fedavg', help='FL algorithm to use')

    
    # algorithm-specific hyperparameters
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--alpha_apfl', type=float, default='0.75', help='APFL parameter alpha')
    parser.add_argument('--alpha_l2gd', type=float, default='1', help='L2GD parameter alpha')
    parser.add_argument('--lambda_l2gd', type=float, default='0.5', help='L2GD parameter lambda')
    parser.add_argument('--lr_in', type=float, default='0.0001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=2, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=13, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')

    #后面加的---FedPAC独有参数 和概念偏移参数
    parser.add_argument('--pac_param', type=float, default=0.1, help=' balance supervised loss and regularization loss')
    parser.add_argument('--is_concept_shift', type=int, default=0, help='control whether the local client is concept-shift')
    parser.add_argument('--concept_shift_rate', type=float, default=1, help='control concept-shift rate')
    parser.add_argument('--local_only', type=int, default=0, help='Only train locally')
    parser.add_argument('--limit_local_output', type=int, default=0, help='limit local output numbers')
    parser.add_argument('--nums_per_class', type=int, default=100, help='smaples number of class')
    parser.add_argument('--is_reset_dataset', type=int, default=1, help='reset train/test dataset or read from file')
    parser.add_argument('--is_reset_model', type=int, default=1, help='reset train init model')
    # parser.add_argument('--output_of_classify_header', type=int, default=10, help='output of classify header')
    # parser.add_argument('--is_class_overlapping', type=int, default=1, help='is client class  overlapping'
    parser.add_argument("--personal_learning_rate", type=float, default=0.01,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--lamda", type=int, default=30, help="Regularization term")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate for pfedme")
    parser.add_argument("--print_all", type=int, default=1, help="whether print all process")
    parser.add_argument('--data_store_file', type=str, default='dict_user', help='model name')

    #噪声生成
    parser.add_argument('--level_n_system', type=float, default=0.0, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.0, help="lower bound of noise level")
    parser.add_argument('--filter_alg', type=str, default='center_psl', help='filter type center_psl / loss_psl')
    # parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--init_steps', type=int, default=0, help="init_steps")
    parser.add_argument('--prov_steps', type=int, default=100, help="prov_steps")
    parser.add_argument('--prov_users', type=int, default=10, help="prov_users")
    parser.add_argument('--psl_step', type=int, default=1, help="prov_steps")


    #RFL
    parser.add_argument('--frac2', type=float, default=0.1,
                        help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--rounds2', type=int, default=300, help="rounds of training in usual training stage")
    parser.add_argument('--T_pl', type=int, help='T_pl: When to start using global guided pseudo labeling', default=80)
    parser.add_argument('--lambda_cen', type=float, help='lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    # parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay size")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="sgd weight decay")
    parser.add_argument('--feature_dim', type=int, help='feature dimension', default=128)
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")

    args = parser.parse_args()
    return args


