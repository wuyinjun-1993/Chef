'''
Created on Nov 11, 2020

'''

from argparse import ArgumentParser

import torch

import os, sys

calculate_influence_time_prefix ='running time to calculate influence::'

calculate_gradient_time_prefix = 'running time to calculate sample-wise gradients::'

remaining_samples_for_computing_influence_prefix='remaining samples for calculating influence::'

training_time_prefix = 'running time to train models::'

gpu_utilization_prefix = 'gpu utilization::'

gpu_mem_prefix = 'gpu memory usage::'

transformed_train_dataset_file_name = 'trans_train_dataset'

transformed_val_dataset_file_name = 'trans_val_dataset'

transformed_test_dataset_file_name = 'trans_test_dataset'

transformed_train_dataset_full_file_name = 'trans_train_dataset_full'

transformed_val_dataset_full_file_name = 'trans_val_dataset_full'

transformed_test_dataset_full_file_name = 'trans_test_dataset_full'

origin_model = 'origin_model'

curr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_done_file(data_dir):
    return 'done' in os.listdir(data_dir)
    
    
def setup_done_file(data_dir):
    fd = os.open(os.path.join(data_dir, 'done'),os.O_RDWR|os.O_CREAT)


def get_default_git_ignore_dir():
    default_git_ignore_dir = curr_path
    
    default_git_ignore_dir += '/.gitignore'
    
    return default_git_ignore_dir

def parse_args():
    
    
    
    
     
    
    
    parser = ArgumentParser()
    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('--test', action='store_true', help="test flag")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")
    
    parser.add_argument('--dir', type = str, default=None, help="directory to store data")
    
    parser.add_argument('--val_dir', type = str, default=None, help="directory to store validation data")
    
    parser.add_argument('--model', type = str, default=None, help="the model name")
    
    parser.add_argument('--dataset', type = str, default=None, help="dataset name")
    
    parser.add_argument('--epochs', type = int, default=10, help="epoch count for training")
    
    parser.add_argument('--lr', type = float, default = 0.0002, help="learning rate")
    
    parser.add_argument('--class', type = int, default = 0, help="the class to be classified")

    parser.add_argument('--bz', type = int, default = 16, help="batch size")
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'output/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    args = parser.parse_args()
    
    if not args.GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    args.device = device
    return args


def parse_train_args():
    
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
     
    
    
    parser = ArgumentParser()
    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")
    
    parser.add_argument('--dir', type = str, default=None, help="directory to store transformed data")
    
    parser.add_argument('--model', type = str, default=None, help="pretrained model name")
    
    parser.add_argument('--binary', action = 'store_true', help="whether to do binary classification")
    
    parser.add_argument('--did', type = int, default = 0, help="which disease to be classified")
    
    parser.add_argument('--dataset', type = str, default=None, help="dataset name")
    
    parser.add_argument('--epochs', type = int, default=10, help="epoch count for training")
    
    parser.add_argument('--lr', type = float, default = 0.01, help="learning rate")
    
    parser.add_argument('--class', type = int, default = 0, help="the class to be classified")

    parser.add_argument('--bz', type = int, default = 16, help="batch size")
    
    parser.add_argument('--wd', type = float, default = 0, help="l2 norm regularization coefficient")
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'output/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    args = parser.parse_args()
    
    if not args.GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    args.device = device
    return args


def parse_optim_del_args():
    
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
     
    
    
    parser = ArgumentParser()
    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")
    
    parser.add_argument('--dir', type = str, default=None, help="directory to store transformed data")
    
    parser.add_argument('--suffix', type = str, default='sl', help="directory to store transformed data")
    
    parser.add_argument('--model', type = str, default=None, help="pretrained model name")
    
    parser.add_argument('--binary', action = 'store_true', help="whether to do binary classification")
    
    parser.add_argument('--continue_labeling', action = 'store_true', help="continue labeling")
    
    parser.add_argument('--resolve_conflict', type = int, default=None, help="continue labeling")
    
    parser.add_argument('--small_test', action = 'store_true', help="flag to do small test")
    
    parser.add_argument('--incremental', action = 'store_true', help="flag to do incremental evaluation")
    
    parser.add_argument('--restart', action = 'store_true', help="clear history")
    
    parser.add_argument('--load_incremental', action = 'store_true', help="flag to load the incrementally evaluated historical information")
    
    parser.add_argument('--period', type = int, default = 20, help="period number for incremental updates")
    
    parser.add_argument('--hist_period', type = int, default = 50, help="period number for incremental updates")
    
#     parser.add_argument('--init_epochs', type = int, default = 10, help="init iterations for incremental updates")
    
    parser.add_argument('--init', type = int, default = 15, help="init iterations for incremental updates")
    
    parser.add_argument('--hist_size', type = int, default = 2, help="history size for the incremental updates")
    
    parser.add_argument('--o2u_epochs', type = int, default = 2, help="o2u epochs")
    
    parser.add_argument('--f1', action = 'store_true', help="use f1 loss")
    
    parser.add_argument('--start', action = 'store_true', help="produce new training dataset")
    
    parser.add_argument('--load_model', action = 'store_true', help="produce new training dataset")
    
    parser.add_argument('--did', type = int, default = 0, help="which disease to be classified")
    
    parser.add_argument('--dataset', type = str, default=None, help="dataset name")
    
    parser.add_argument('--norm', type = str, default=None, help="norm for evaluating model changes")
    
    parser.add_argument('--epochs', type = int, default=10, help="epoch count for training")
    
    parser.add_argument('--jump', type = int, default=10, help="batches for evaluations")
    
    parser.add_argument('--lr', type = float, default = 0.1, help="learning rate")
    
    parser.add_argument('--derived_lr', type = float, default = 0.01, help="learning rate")
    
    parser.add_argument('--derived_epochs', type = int, default = 1000, help="learning rate")
    
    parser.add_argument('--derived_bz', type = int, default = 1000, help="learning rate")
    
    parser.add_argument('--derived_l2', type = float, default = 0, help="learning rate")
    
    parser.add_argument('--tlr', type = float, default = 0.001, help="learning rate")
    
    parser.add_argument('--del_ratio', type = float, default = 0.001, help="ratio of deleted samples")
    
    parser.add_argument('--class', type = int, default = 0, help="the class to be classified")

    parser.add_argument('--bz', type = int, default = 16, help="batch size")
    
    parser.add_argument('--iter_count', type = int, default = None, help="iter_count")
    
    parser.add_argument('--GPU_measure', action = 'store_true', help="measure the gpu utilization")
    
    parser.add_argument('--no_prov', action = 'store_true', help="measure the gpu utilization")
    
    parser.add_argument('--tars', action = 'store_true', help="use the data for tars")
    
    parser.add_argument('--wd', type = float, default = 0.001, help="l2 norm regularization coefficient")
    
    parser.add_argument('--inner_epoch_count', type = int, default = 50, help="epochs of training process")
    
    parser.add_argument('--out_epoch_count', type = int, default = 10, help="epochs of outer loop")
    
    parser.add_argument('--removed_count', type = int, default = 50, help="epochs of outer loop")
    
    parser.add_argument('--no_probs', action = 'store_true', help="epochs of outer loop")
    
    parser.add_argument('--same_y_diff', action = 'store_true', help="epochs of outer loop")
    
    parser.add_argument('--regular_rate', type = float, default = 1.0, help="epochs of outer loop")
    
    parser.add_argument('--noisy_ratio', type = float, default = 0.1, help="epochs of outer loop")
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'output/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    args = parser.parse_args()
    
    if not args.GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    args.device = device
    return args



def parse_preprocessing_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type = str, default=None, help="directory to store data")
    
    parser.add_argument('--val_dir', type = str, default=None, help="directory to store validation data")
    
    parser.add_argument('--label', type = str, default=None, help="files to store the labels")
    
    parser.add_argument('--val_label', type = str, default=None, help="files to store the validation labels")
    
    parser.add_argument('--dataset', type = str, default=None, help="dataset name")
    
    args = parser.parse_args()
    
    return args


def parse_perturbation_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type = str, default=None, help="directory to store data")
    
#     parser.add_argument('--val_dir', type = str, default=None, help="directory to store validation data")
    
    parser.add_argument('--label', type = str, default=None, help="files to store the labels")
    
    parser.add_argument('--model', type = str, default=None, help="pretrained model")
    
#     parser.add_argument('--val_label', type = str, default=None, help="files to store the validation labels")
    
    parser.add_argument('--dataset', type = str, default=None, help="dataset name")
    
    parser.add_argument('--bz', type = int, default=16, help="batch size")
    
    parser.add_argument('--eps', type = float, help="perturbation bound")
    
    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('--small', action='store_true', help="small test")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")

    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'output/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    
    args = parser.parse_args()
    
    if not args.GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    args.device = device
    
    return args

def normalize_tensor(X_tensor):
    
    mean_tensor = torch.mean(X_tensor, dim=0)
    
    std_tensor = torch.std(X_tensor, dim=0)
    
    X_tensor = (X_tensor - mean_tensor)/std_tensor
    
    return X_tensor



