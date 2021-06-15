'''
Created on Dec 11, 2020

'''
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import copy
import torch.nn.functional as F
import copy
from torch import autograd
import higher
import itertools
from models.Data_preparer import MyDataset
import statistics
from collections import Counter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/pytorch_influence_functions')


from sklearn.metrics import roc_auc_score
from sklearn import metrics

from torch.utils.data import Dataset, DataLoader

import models

try:
    from utils.utils import *
    from models.util_func import *
    from real_examples.utils_real import *
except ImportError:
    from utils import *
    from util_func import *
    from utils_real import *

sum_upper_bound = 5

final_sum_upper_bound = 0.05

single_upper_bound = 0.01

retina_model = 'ResNet50'

def normalize_data(training_data, validation_data, test_data):
    training_mean = torch.mean(training_data, dim = 0)
    
    training_std = torch.std(training_data, dim = 0)
    
    transformed_training_data = (training_data - training_mean.view(1,-1))/training_std.view(1,-1)
    
    transformed_validation_data = (validation_data - training_mean.view(1,-1))/training_std.view(1,-1)
    
    transformed_test_data = (test_data - training_mean.view(1,-1))/training_std.view(1,-1)
    
#     transformed_small_data = (small_data - training_mean)/training_std
    
    return transformed_training_data, transformed_validation_data, transformed_test_data


def process_annotated_labels(train_annotated_label_list):
    
    label_num = 3
    
    training_sample_count = len(train_annotated_label_list)
    
    train_annotated_label_tensor = torch.zeros([training_sample_count, label_num])
    
    for k in range(training_sample_count):
        curr_train_annotated_label_list = train_annotated_label_list[k]
        
        curr_train_annotated_label_list.remove(-1)
        
        curr_train_annotated_label_list = curr_train_annotated_label_list[0:label_num]
        
        if len(curr_train_annotated_label_list) < 3:
            curr_train_annotated_label_list.extend([-1]*(3-len(curr_train_annotated_label_list)))
            
        train_annotated_label_tensor[k] = torch.tensor(curr_train_annotated_label_list) 
    
    
    return train_annotated_label_tensor

def obtain_fashion_examples(args, noisy = False, load_origin = False, is_tars = False):
    
    full_output_dir = args.output_dir
    
    origin_train_dataset = None
    
    origin_val_dataset = None
    
    origin_test_dataset = None
    
    
    if is_tars:
        training_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset_tars',map_location='cpu')
    else:
        training_dataset = torch.load(full_output_dir + '/trans_train_dataset_new',map_location='cpu')
    
    training_prob_labels = torch.load(full_output_dir + '/train_prob_labels',map_location='cpu')
    
    train_annotated_label_tensor = torch.load(full_output_dir + '/train_annotated_labels',map_location='cpu')
    
    train_origin_labels = training_dataset.labels.view(-1).type(torch.long)
    
    training_dataset.data = training_dataset.data.type(torch.double) 
    
    training_dataset.labels = training_prob_labels.type(torch.double)
    
    val_dataset = torch.load(full_output_dir + '/trans_val_dataset_new',map_location='cpu')
    
    val_dataset.data = val_dataset.data.type(torch.double) 
    
    test_dataset = torch.load(full_output_dir + '/trans_test_dataset_new',map_location='cpu')
    
    test_dataset.data = test_dataset.data.type(torch.double)
    
    val_dataset.labels = val_dataset.labels.view(-1).type(torch.long)
    
    test_dataset.labels = test_dataset.labels.view(-1).type(torch.long)
    
    test_dataset.data = test_dataset.data[test_dataset.labels != -1].type(torch.double)
    
    test_dataset.labels = test_dataset.labels[test_dataset.labels != -1].type(torch.long)
    
    
    if load_origin:
    
        origin_train_dataset = torch.load(full_output_dir + '/origin_train_dataset_new',map_location='cpu')
        
        origin_train_dataset.labels = training_prob_labels.type(torch.double)
        
        
        
        origin_val_dataset = torch.load(full_output_dir + '/origin_val_dataset_new',map_location='cpu')
        
        origin_test_dataset = torch.load(full_output_dir + '/origin_test_dataset_new',map_location='cpu')
        
        origin_val_dataset.labels = origin_val_dataset.labels.view(-1).type(torch.long)
        
        origin_test_dataset.labels = origin_test_dataset.labels.view(-1).type(torch.long)
    # if load_origin:
    
        origin_test_dataset.data = origin_test_dataset.data[test_dataset.labels != -1].type(torch.double)
        
        origin_test_dataset.labels = origin_test_dataset.labels[test_dataset.labels != -1].type(torch.long)
    
        origin_val_dataset.data = origin_val_dataset.data[origin_val_dataset.labels != -1].type(torch.double)
        
        origin_val_dataset.labels = origin_val_dataset.labels[origin_val_dataset.labels != -1].type(torch.long)
        
        print('train shape::', origin_train_dataset.data.shape,training_prob_labels.shape)
        
        print('valid shape::', origin_val_dataset.data.shape,origin_val_dataset.labels.shape)
        
        print('test shape::', origin_test_dataset.data.shape,origin_test_dataset.labels.shape)
        
    print('valid shape::', val_dataset.data.shape, val_dataset.labels.shape)
    
    print('test shape::', test_dataset.data.shape, test_dataset.labels.shape)
        
        
    training_dataset.data, val_dataset.data, test_dataset.data = normalize_data(training_dataset.data, val_dataset.data, test_dataset.data)
    
    print(len(training_dataset), training_dataset.data.shape, len(train_annotated_label_tensor))
    
    noise_sample_ids = torch.load(full_output_dir + '/noisy_sample_ids')
    
    print(len(noise_sample_ids))
    
    print(val_dataset.data.shape)
    
    # train_annotated_label_tensor = process_annotated_labels(train_annotated_label_list)
    
    return training_dataset, train_origin_labels, val_dataset, test_dataset, train_annotated_label_tensor, full_output_dir, None, None, origin_train_dataset, origin_val_dataset, origin_test_dataset


def convert_list_to_tensor(train_annotated_label_list, count = 5):
    
    train_annotated_label_tensor = torch.zeros([len(train_annotated_label_list), count], dtype = torch.long)
    for k in range(len(train_annotated_label_list)):
        # print(train_annotated_label_list[k][0:count])
        
        curr_train_annotated_label_list = train_annotated_label_list[k]
        
        curr_train_annotated_label_tensor = torch.tensor(curr_train_annotated_label_list[0:count])
        
        train_annotated_label_tensor[k][0:len(curr_train_annotated_label_tensor)] = curr_train_annotated_label_tensor
        
        if len(curr_train_annotated_label_tensor) < count:
            # print('here')
            
            train_annotated_label_tensor[k][len(curr_train_annotated_label_tensor):] = torch.tensor([-2]*(count - len(curr_train_annotated_label_tensor)))
        
    return train_annotated_label_tensor
    
def filter_small_list(train_annotated_label_list, count = 5):
    
    train_ids_with_enough_labels = []
    
    train_annotated_label_list_with_enough_labels = []
    
    for k in range(len(train_annotated_label_list)):
        if len(train_annotated_label_list[k]) >= count:
            
            # unique_values = set(train_annotated_label_list[k])
            #
            # value_count_list = []
            #
            # for v in unique_values:
                # value_count = train_annotated_label_list[k].count(v)
                # value_count_list.append(value_count)
                #
            # np.sort(np.array(value_count_list))
            
            train_ids_with_enough_labels.append(k)
            train_annotated_label_list_with_enough_labels.append(train_annotated_label_list[k])
            
            
            
    return torch.tensor(train_ids_with_enough_labels), train_annotated_label_list_with_enough_labels

def obtain_fact_examples(args, noisy = False, load_origin = False, is_tars = False):
    
    full_output_dir = args.output_dir
    
    training_origin_dataset = None
    
    val_origin_dataset = None
    
    test_origin_dataset = None
    
    training_feature_tensor = torch.load(full_output_dir + '/train_feature_tensor').type(torch.double)
    
    training_prob_labels = torch.load(full_output_dir + '/train_prob_labels').type(torch.double)
    
    valid_feature_tensor = torch.load(full_output_dir + '/valid_feature_tensor').type(torch.double)
    
    valid_labels = torch.load(full_output_dir + '/valid_labels')
    
    test_feature_tensor = torch.load(full_output_dir + '/test_feature_tensor').type(torch.double)
    
    test_labels = torch.load(full_output_dir + '/test_labels')
    
    if load_origin:
        valid_origin_feature_tensor = torch.load(os.path.join(full_output_dir, 'valid_origin_feature_tensor')).type(torch.double)
            
        train_origin_feature_tensor = torch.load(os.path.join(full_output_dir, 'train_origin_feature_tensor')).type(torch.double)
        
        test_origin_feature_tensor = torch.load(os.path.join(full_output_dir, 'test_origin_feature_tensor')).type(torch.double)
    
    
    
    
    
    training_feature_tensor, valid_feature_tensor, test_feature_tensor = normalize_data(training_feature_tensor, valid_feature_tensor, test_feature_tensor)
    
    training_prob_labels = training_prob_labels.detach()
    
    training_dataset = MyDataset(training_feature_tensor, training_prob_labels)
    
    val_dataset = MyDataset(valid_feature_tensor, valid_labels)
    
    test_dataset = MyDataset(test_feature_tensor, test_labels)
    
    # training_prob_labels = torch.load(full_output_dir + '/prob_labels')
    
    train_annotated_label_tensor = convert_list_to_tensor(torch.load(full_output_dir + '/train_annotated_label_list'))
    
    train_origin_labels = torch.mode(train_annotated_label_tensor, dim = 1)[0]
    
    # train_origin_labels = training_dataset.labels.view(-1).type(torch.long)
    #
    # training_dataset.data = training_dataset.data.type(torch.double) 
    #
    # training_dataset.labels = training_prob_labels.type(torch.double)
    
    # val_dataset = torch.load(full_output_dir + '/valid_dataset')
    
    val_dataset.data = val_dataset.data[val_dataset.labels != 2].type(torch.double)
    
    # test_dataset = torch.load(full_output_dir + '/test_dataset')
    if load_origin:
        valid_origin_feature_tensor = valid_origin_feature_tensor[val_dataset.labels != 2].type(torch.double)
        
        test_origin_feature_tensor = test_origin_feature_tensor[test_dataset.labels != 2].type(torch.double)
    
    test_dataset.data = test_dataset.data[test_dataset.labels != 2].type(torch.double)
    
    
    
    
    
    if load_origin:
        training_origin_dataset = MyDataset(train_origin_feature_tensor, training_prob_labels)
        
        val_origin_dataset = MyDataset(valid_origin_feature_tensor, valid_labels.clone()[valid_labels != 2])
        
        test_origin_dataset = MyDataset(test_origin_feature_tensor, test_labels.clone()[test_labels != 2])
        
        # val_origin_dataset.data = val_origin_dataset.data[valid_labels != 2]
        #
        # val_origin_dataset.labels = val_origin_dataset.labels[valid_labels != 2]
        #
        # test_origin_dataset.data = test_origin_dataset.data[test_labels != 2]
        #
        # test_origin_dataset.labels = test_origin_dataset.labels[test_labels != 2]
    
    val_dataset.labels = val_dataset.labels[val_dataset.labels != 2].view(-1).type(torch.long)
    
    test_dataset.labels = test_dataset.labels[test_dataset.labels != 2].view(-1).type(torch.long)
    
    # test_dataset.data = test_dataset.data[test_dataset.labels != -1].type(torch.double)
    #
    # test_dataset.labels = test_dataset.labels[test_dataset.labels != -1].type(torch.long)
    
    # train_annotated_label_tensor = process_annotated_labels(train_annotated_label_list)
    print(torch.unique(train_origin_labels), torch.unique(val_dataset.labels), torch.unique(test_dataset.labels))
    print(training_dataset.data.shape, val_dataset.data.shape, test_dataset.data.shape)
    print(train_origin_labels.shape)
    return training_dataset, train_origin_labels, val_dataset, test_dataset, train_annotated_label_tensor, full_output_dir, None, None, training_origin_dataset, val_origin_dataset, test_origin_dataset

def get_multi_mode(mode_count_list):
    
    # max_
    multi_mode_list = []
    
    max_count = 0
    
    # for val in curr_train_annotated_label_list:
    for k in range(len(mode_count_list)):
        if mode_count_list[k][1] > max_count:
            multi_mode_list.clear()
            multi_mode_list.append(mode_count_list[k][0])
            
            max_count = mode_count_list[k][1]
        else:
            if mode_count_list[k][1] == max_count:
                multi_mode_list.append(mode_count_list[k][0])
    
    return multi_mode_list
            
        
    

def obtain_origin_labels(train_annotated_label_tensor):
    agg_res = []
    
    for i in range(len(train_annotated_label_tensor)):
        curr_train_annotated_label_list = train_annotated_label_tensor[i].tolist()
        
        if -2 in curr_train_annotated_label_list:
            # print('here')
        
            curr_train_annotated_label_list.remove(-2)
        
        mode_count_list = Counter(curr_train_annotated_label_list).most_common()
        
        multi_mode_value = get_multi_mode(mode_count_list)
        # multi_mode_value = statistics.multimode(curr_train_annotated_label_list)
        
        if len(multi_mode_value) > 1:
            agg_res.append(-1)
        else:
            agg_res.append(multi_mode_value[0])
            
    return torch.tensor(agg_res)

def obtain_twitter_examples(args, noisy = False, load_origin = False, is_tars = False):
    
    full_output_dir = args.output_dir
    
    training_feature_tensor = torch.load(full_output_dir + '/train_features').type(torch.double)#[:,-10:]
    
    training_prob_labels = torch.load(full_output_dir + '/train_labels').type(torch.double)
    
    valid_feature_tensor = torch.load(full_output_dir + '/valid_features').type(torch.double)#[:,-10:]
    
    valid_labels = torch.load(full_output_dir + '/valid_labels')
    
    test_feature_tensor = torch.load(full_output_dir + '/test_features').type(torch.double)#[:,-10:]
    
    test_labels = torch.load(full_output_dir + '/test_labels')
    
    train_origin_dataset = None
    
    valid_origin_dataset = None
    
    test_origin_dataset = None
    
    if load_origin:
    
        train_origin_features = torch.load(os.path.join(args.output_dir, 'train_origin_features')).type(torch.double)
            
        valid_origin_features = torch.load(os.path.join(args.output_dir, 'valid_origin_features')).type(torch.double)
        
        test_origin_features = torch.load(os.path.join(args.output_dir, 'test_origin_features')).type(torch.double)
    
    # training_prob_labels = torch.load(full_output_dir + '/prob_labels')
    
    train_annotated_label_tensor = torch.load(full_output_dir + '/full_train_annotated_labels')
    
    
    train_ids_with_enough_labels = torch.tensor(list(range(training_feature_tensor.shape[0])))
    # train_ids_with_enough_labels, train_annotated_label_tensor = filter_small_list(train_annotated_label_tensor, count = 3)
    
    train_annotated_label_tensor = convert_list_to_tensor(train_annotated_label_tensor, 3)
    
    # torch.unique(train_annotated_label_tensor, dim=1)
    #
    # train_origin_labels = torch.mode(train_annotated_label_tensor, dim = 1)[0]
    train_origin_labels = obtain_origin_labels(train_annotated_label_tensor)
    
    training_feature_tensor = training_feature_tensor[train_ids_with_enough_labels]
    
    training_prob_labels = training_prob_labels[train_ids_with_enough_labels]
    
    training_feature_tensor, valid_feature_tensor, test_feature_tensor = normalize_data(training_feature_tensor, valid_feature_tensor, test_feature_tensor)
    
    
    training_dataset = MyDataset(training_feature_tensor, training_prob_labels)
    
    val_dataset = MyDataset(valid_feature_tensor, valid_labels)
    
    test_dataset = MyDataset(test_feature_tensor, torch.tensor(list(test_labels)))
    
    
    
    # train_origin_labels = training_dataset.labels.view(-1).type(torch.long)
    #
    # training_dataset.data = training_dataset.data.type(torch.double) 
    #
    # training_dataset.labels = training_prob_labels.type(torch.double)
    
    # val_dataset = torch.load(full_output_dir + '/valid_dataset')
    
    val_dataset.data = val_dataset.data[val_dataset.labels != 2].type(torch.double)
    
    if load_origin:
        valid_origin_features = valid_origin_features[val_dataset.labels != 2].type(torch.double)
    
    # test_dataset = torch.load(full_output_dir + '/test_dataset')
    
    test_dataset.data = test_dataset.data[test_dataset.labels != 2].type(torch.double)
    
    if load_origin:
        test_origin_features = test_origin_features[test_dataset.labels != 2].type(torch.double)
    
    val_dataset.labels = val_dataset.labels[val_dataset.labels != 2].view(-1).type(torch.long)
    
    # valid_origin_labels = val_dataset.labels[val_dataset.labels != 2].view(-1).type(torch.long)
    
    test_dataset.labels = test_dataset.labels[test_dataset.labels != 2].view(-1).type(torch.long)
    
    
    
    # clean_sample_ids = torch.tensor([])
    #
    # noise_sample_ids = torch.tensor(list(range(training_dataset.labels.shape[0])))
    #
    # torch.save(clean_sample_ids, full_output_dir + '/clean_sample_ids')
    #
    # torch.save(noise_sample_ids, full_output_dir + '/noisy_sample_ids')
    # test_dataset.data = test_dataset.data[test_dataset.labels != -1].type(torch.double)
    #
    # test_dataset.labels = test_dataset.labels[test_dataset.labels != -1].type(torch.long)
    
    # train_annotated_label_tensor = process_annotated_labels(train_annotated_label_list)
    # print(torch.unique(train_origin_labels), torch.unique(val_dataset.labels), torch.unique(test_dataset.labels))
    print(len(training_dataset.data), len(training_dataset.labels), len(train_origin_labels), len(train_annotated_label_tensor))
    # print()
    noise_sample_ids = torch.load(full_output_dir + '/noisy_sample_ids')
    
    print(len(noise_sample_ids))
    
    print('train shape::', training_feature_tensor.shape)
    
    if load_origin:
        train_origin_dataset = MyDataset(train_origin_features, training_prob_labels)
        
        valid_origin_dataset = MyDataset(valid_origin_features, val_dataset.labels)
        
        test_origin_dataset = MyDataset(test_origin_features, test_dataset.labels)
        
        print('train shape::', train_origin_features.shape)
    
    return training_dataset, train_origin_labels, val_dataset, test_dataset, train_annotated_label_tensor, full_output_dir, None, None, train_origin_dataset, valid_origin_dataset, test_origin_dataset


def obtain_chexpert_examples(args, noisy = False, load_origin=False, is_tars = False):
    disease_id = 1
    
    dataset_name = args.dataset
    
        
#     if not os.path.exists(full_output_dir):
#         os.makedirs(full_output_dir)
    if not noisy:
        
        full_output_dir = os.path.join(args.output_dir, dataset_name + '_ResNet50')
        training_dataset = torch.load(full_output_dir + '/trans_train_dataset')
        
        training_dataset.labels = training_dataset.labels[:,disease_id]
        
        val_dataset = torch.load(full_output_dir + '/trans_val_dataset')
        
        val_dataset.labels = val_dataset.labels[:,disease_id]
        
        binary = True
        
        return training_dataset, val_dataset, None, full_output_dir, binary

    else:
        
        origin_train_dataset = None
            
        origin_val_dataset = None
        
        origin_test_dataset = None
        
        full_output_dir = args.output_dir
        
        if is_tars:
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset_tars')
        else:
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
        
        valid_dataset = torch.load(full_output_dir + '/validation_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
    
        full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
        
        small_dataset = None#torch.load(full_output_dir + '/small_dataset')
        
        selected_small_sample_ids = None
        selected_noisy_sample_ids = None
        
        
        if os.path.exists(full_output_dir + '/selected_small_sample_ids'):
            selected_small_sample_ids = torch.load(full_output_dir + '/selected_small_sample_ids')
        
        if os.path.exists(full_output_dir + '/selected_noisy_sample_ids'):
            selected_noisy_sample_ids = torch.load(full_output_dir + '/selected_noisy_sample_ids')
        
        train_dataset.labels = torch.tensor(train_dataset.labels.tolist()).type(torch.DoubleTensor)
        
        if load_origin:
            origin_train_dataset = torch.load(full_output_dir + '/origin_train_dataset_new')
            
            origin_train_dataset.labels = train_dataset.labels
            
            origin_val_dataset = torch.load(full_output_dir + '/origin_val_dataset_new')
            
            origin_test_dataset = torch.load(full_output_dir + '/origin_test_dataset_new')
            
            print('data shape::', origin_train_dataset.data.shape, origin_val_dataset.data.shape, origin_test_dataset.data.shape)
            
        print('data shape::', train_dataset.data.shape, valid_dataset.data.shape, test_dataset.data.shape)
        
        return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir, selected_small_sample_ids,selected_noisy_sample_ids, origin_train_dataset, origin_val_dataset, origin_test_dataset  


def obtain_alarm_examples_origin(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = os.path.join(args.output_dir, dataset_name)
    
#     train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#     
#     valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#     
#     test_dataset = torch.load(full_output_dir + '/preprocessed_test')
    
    train_dataset = torch.load(full_output_dir + '/train_dataset')
    
    valid_dataset = torch.load(full_output_dir + '/valid_dataset')
    
    test_dataset = torch.load(full_output_dir + '/test_dataset')
    
    
#     if torch.unique(train_dataset.labels).shape[0] > 2:
#         train_dataset.labels = (train_dataset.labels >= 3).type(torch.LongTensor)
#         valid_dataset.labels = (valid_dataset.labels >= 3).type(torch.LongTensor)
#         test_dataset.labels = (test_dataset.labels >= 3).type(torch.LongTensor)
    
#     train_dataset.labels = 
    
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(test_dataset, batch_size=args.bz)
    
  
    binary = True
    
    print('load done')
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 

def obtain_retina_examples_origin(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = os.path.join(args.output_dir, dataset_name + '_' + retina_model)
    
#     train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#     
#     valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#     
#     test_dataset = torch.load(full_output_dir + '/preprocessed_test')
    
    train_dataset = torch.load(full_output_dir + '/trans_train_dataset' + '_new')
    
    valid_dataset = torch.load(full_output_dir + '/trans_val_dataset' + '_new')
    
    test_dataset = torch.load(full_output_dir + '/trans_test_dataset' + '_new')
    
    
    if torch.unique(train_dataset.labels).shape[0] > 2:
        train_dataset.labels = (train_dataset.labels <= 1).type(torch.LongTensor)
        valid_dataset.labels = (valid_dataset.labels <= 1).type(torch.LongTensor)
        test_dataset.labels = (test_dataset.labels <= 1).type(torch.LongTensor)
    
#     train_dataset.labels = 
    
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(test_dataset, batch_size=args.bz)
    
#     if args.start:
#         data_preparer = models.Data_preparer()
#         train_DL, valid_DL = data_preparer.prepare_retina(full_output_dir, args.bz)
#         
#         test_DL = data_preparer.prepare_test_retina(full_output_dir, args.bz)
#         
#         
#         torch.save(train_DL, full_output_dir + '/train_DL')
#         
#         torch.save(valid_DL, full_output_dir + '/valid_DL')
#         
#         torch.save(test_DL, full_output_dir + '/test_DL')
#     
#     else:
#         train_DL = torch.load(full_output_dir + '/train_DL')
#         
#         valid_DL = torch.load(full_output_dir + '/valid_DL')
#         
#         test_DL = torch.load(full_output_dir + '/test_DL')
    
    binary = True
    
    print('load done')
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 


def obtain_oct_examples_origin(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir#os.path.join(args.output_dir, dataset_name)
    
#     train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#     
#     valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#     
#     test_dataset = torch.load(full_output_dir + '/preprocessed_test')
    
    train_dataset = torch.load(full_output_dir + '/trans_train_dataset')
    
    valid_dataset = torch.load(full_output_dir + '/trans_val_dataset')
    
    test_dataset = torch.load(full_output_dir + '/trans_test_dataset')
    
    
#     if torch.unique(train_dataset.labels).shape[0] > 2:
#     train_dataset.labels = (train_dataset.labels < 3).type(torch.LongTensor)
#     valid_dataset.labels = (valid_dataset.labels < 3).type(torch.LongTensor)
#     test_dataset.labels = (test_dataset.labels < 3).type(torch.LongTensor)
    
#     train_dataset.labels = 
    
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(test_dataset, batch_size=args.bz)
    
#     if args.start:
#         data_preparer = models.Data_preparer()
#         train_DL, valid_DL = data_preparer.prepare_retina(full_output_dir, args.bz)
#         
#         test_DL = data_preparer.prepare_test_retina(full_output_dir, args.bz)
#         
#         
#         torch.save(train_DL, full_output_dir + '/train_DL')
#         
#         torch.save(valid_DL, full_output_dir + '/valid_DL')
#         
#         torch.save(test_DL, full_output_dir + '/test_DL')
#     
#     else:
#         train_DL = torch.load(full_output_dir + '/train_DL')
#         
#         valid_DL = torch.load(full_output_dir + '/valid_DL')
#         
#         test_DL = torch.load(full_output_dir + '/test_DL')
    
    binary = True
    
    print('load done')
    
#     full_output_dir = os.path.join(args.output_dir, dataset_name)
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 


def obtain_mnist_examples_origin(args, origin = False):
    dataset_name = args.dataset
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
        
    data_preparer = models.Data_preparer()
    
    dataset_train, dataset_test = get_train_test_data_loader_by_name_lr(data_preparer, args.output_dir, dataset_name)
    
    train_dataset, val_dataset = partition_val_dataset(dataset_train, ratio=0.1)

    torch.save(train_dataset, full_output_dir + '/train_dataset')
    
    torch.save(val_dataset, full_output_dir + '/val_dataset')
    
    torch.save(dataset_test, full_output_dir + '/dataset_test')
    
    train_DL = DataLoader(dataset_train, args.bz)
    
    valid_DL = DataLoader(val_dataset, args.bz)
    
    test_DL = DataLoader(dataset_test, args.bz)
    
    binary = False
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 

def obtain_mimic_examples_origin(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir
    
#     if not origin:
#         
#         train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#         
#         valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#         
#         test_dataset = torch.load(full_output_dir + '/preprocessed_test')
#     else:
#         train_dataset = torch.load(full_output_dir + '/preprocessed_train_origin')
#         
#         valid_dataset = torch.load(full_output_dir + '/preprocessed_valid_origin')
#         
#         test_dataset = torch.load(full_output_dir + '/preprocessed_test_origin')
        
    train_dataset = torch.load(full_output_dir + '/trans_train_dataset')
        
    valid_dataset = torch.load(full_output_dir + '/trans_val_dataset')
    
    test_dataset = torch.load(full_output_dir + '/trans_test_dataset')
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(test_dataset, batch_size=args.bz)
    
    binary = True
    
    print('load done')
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 


def parition_chexpert_valid_test_dataset2(train_dataset, trans_test_dataset, origin_train_dataset, origin_test_dataset):
    
    positive_ids = torch.nonzero(train_dataset.labels == 1).view(-1).tolist()
    
    negtive_ids = torch.nonzero(train_dataset.labels == 0).view(-1).tolist()
    
    print('train shape::', train_dataset.data.shape, origin_train_dataset.data.shape)
    
    print('test shape::', trans_test_dataset.data.shape, origin_test_dataset.data.shape)
    
    print('label diff::', torch.norm(train_dataset.labels.type(torch.double) - origin_train_dataset.labels.type(torch.double)), torch.norm(trans_test_dataset.labels.type(torch.double)- origin_test_dataset.labels.type(torch.double)))
    
    selected_pos_ids, _ = sample_data_and_select_test_samples(torch.sum(trans_test_dataset.labels.view(-1) == 1), positive_ids, trans_test_dataset.data[trans_test_dataset.labels.view(-1) == 1], train_dataset.data, train_dataset.labels)
    
    selected_neg_ids, _ = sample_data_and_select_test_samples(torch.sum(trans_test_dataset.labels.view(-1) == 0), negtive_ids, trans_test_dataset.data[trans_test_dataset.labels.view(-1) == 0], train_dataset.data, train_dataset.labels)
    
    valid_ids = []
    
    valid_ids.extend(selected_pos_ids)
    
    valid_ids.extend(selected_neg_ids)
    
    train_ids = list(set(list(range(train_dataset.data.shape[0]))).difference(set(valid_ids)))
    
    train_features = train_dataset.data[torch.tensor(train_ids)]
    
    train_labels = train_dataset.labels[torch.tensor(train_ids)]
    
    origin_train_features = origin_train_dataset.data[torch.tensor(train_ids)]
    
    origin_train_labels = origin_train_dataset.labels[torch.tensor(train_ids)]
    
    
    
    valid_features = train_dataset.data[torch.tensor(valid_ids)]
    
    valid_labels = train_dataset.labels[torch.tensor(valid_ids)]
    
    origin_valid_features = origin_train_dataset.data[torch.tensor(valid_ids)]
    
    origin_valid_labels = origin_train_dataset.labels[torch.tensor(valid_ids)]
    
    
    return MyDataset(train_features, train_labels), MyDataset(valid_features, valid_labels), MyDataset(origin_train_features, origin_train_labels), MyDataset(origin_valid_features, origin_valid_labels)
    
    

def partition_chexpert_valid_test_dataset(trans_val_dataset):
    trans_val_features = trans_val_dataset.data
    
    trans_val_labels = trans_val_dataset.labels
    
    pos_trans_val_features = trans_val_features[trans_val_labels == 1]
    
    pos_trans_val_labels = trans_val_labels[trans_val_labels == 1]
    
    neg_trans_val_features = trans_val_features[trans_val_labels == 0]
    
    neg_trans_val_labels = trans_val_labels[trans_val_labels == 0]
    
    test_ratio = 0.5
    
    pos_rand_ids = torch.randperm(pos_trans_val_features.shape[0])
    
    pos_test_count = int(pos_rand_ids.shape[0]*test_ratio) 
    
    selected_pos_test_features = pos_trans_val_features[pos_rand_ids[0:pos_test_count]]
    
    selected_pos_test_labels = pos_trans_val_labels[pos_rand_ids[0:pos_test_count]]
    
    selected_pos_val_features = pos_trans_val_features[pos_rand_ids[pos_test_count:]]
    
    selected_pos_val_labels = pos_trans_val_labels[pos_rand_ids[pos_test_count:]]
    
    
    
    
    neg_rand_ids = torch.randperm(neg_trans_val_features.shape[0])
    
    neg_test_count = int(neg_rand_ids.shape[0]*test_ratio) 
    
    selected_neg_test_features = neg_trans_val_features[neg_rand_ids[0:neg_test_count]]
    
    selected_neg_test_labels = neg_trans_val_labels[neg_rand_ids[0:neg_test_count]]
    
    selected_neg_val_features = neg_trans_val_features[neg_rand_ids[neg_test_count:]]
    
    selected_neg_val_labels = neg_trans_val_labels[neg_rand_ids[neg_test_count:]]
    
    
    trans_val_dataset = MyDataset(torch.cat([selected_pos_val_features, selected_neg_val_features], 0), torch.cat([selected_pos_val_labels, selected_neg_val_labels], 0))
    
    selected_test_dataset = MyDataset(torch.cat([selected_pos_test_features, selected_neg_test_features], 0), torch.cat([selected_pos_test_labels, selected_neg_test_labels], 0))

    return trans_val_dataset, selected_test_dataset

def sample_data_and_select_test_samples(test_count, remaining_ids, valid_features, full_features, full_labels):
    # dist = np.random.multivariate_normal(valid_mean.numpy(), torch.diag(valid_std).numpy())
    
    selected_ids = []
    
    for k in range(test_count):
        # sampled_feature = torch.normal(mean=valid_mean, std=valid_std)
        sampled_feature = valid_features[k]
        
        min_value = None
        
        min_id = None
        
        for j in range(len(remaining_ids)):
            curr_feature = full_features[remaining_ids[j]]
            curr_label = full_labels[remaining_ids[j]]
            
            distance = -(torch.dot(curr_feature,sampled_feature)/(torch.norm(curr_feature)*torch.norm(sampled_feature))).item()#
            
            if min_value is None:
                min_value = distance
                
                min_id = j
            
            # if torch.norm(curr_feature - sampled_feature).item() < min_value:
            if distance < min_value:
                
                min_value = distance#(torch.dot(curr_feature,sampled_feature)/(torch.norm(curr_feature)*torch.norm(sampled_feature))).item()#
                
                min_id = j
            
        
        selected_ids.append(remaining_ids[min_id])
        
        print(full_labels[remaining_ids[min_id]])
        
        del remaining_ids[min_id]
        
    return selected_ids, remaining_ids

def obtain_chexpert_examples_origin(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir
    
    selected_label_id = 1
    
#     if not origin:
#         
#         train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#         
#         valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#         
#         test_dataset = torch.load(full_output_dir + '/preprocessed_test')
#     else:
#         train_dataset = torch.load(full_output_dir + '/preprocessed_train_origin')
#         
#         valid_dataset = torch.load(full_output_dir + '/preprocessed_valid_origin')
#         
#         test_dataset = torch.load(full_output_dir + '/preprocessed_test_origin')
        
    train_dataset = torch.load(full_output_dir + '/trans_train_dataset')
    
    train_dataset.labels = train_dataset.labels[:,selected_label_id]
    
        
    valid_dataset = torch.load(full_output_dir + '/trans_val_dataset')
    
    
    origin_train_dataset = torch.load(full_output_dir + '/origin_train_dataset')
        
    origin_val_dataset = torch.load(full_output_dir + '/origin_val_dataset')
    
    origin_train_dataset.labels = origin_train_dataset.labels[:,selected_label_id]
        
    # origin_test_dataset = torch.load(full_output_dir + '/origin_test_dataset')
    
    
    
    
    
    valid_dataset.labels = valid_dataset.labels[:,selected_label_id]
    
    origin_val_dataset.labels = origin_val_dataset.labels[:,selected_label_id]
    
    valid_dataset.data = valid_dataset.data[~(valid_dataset.labels == -1)]
    
    valid_dataset.labels = valid_dataset.labels[~(valid_dataset.labels == -1)]
    
    origin_val_dataset.data = origin_val_dataset.data[~(valid_dataset.labels == -1)]
    
    origin_val_dataset.labels = origin_val_dataset.labels[~(valid_dataset.labels == -1)]
    
    # valid_dataset, test_dataset = partition_chexpert_valid_test_dataset(valid_dataset)
    
    # origin_train_dataset.labels = origin_train_dataset.labels[]
    
    train_dataset.data = train_dataset.data[~(train_dataset.labels == -1)]
    
    train_dataset.labels = train_dataset.labels[~(train_dataset.labels == -1)]
    
    
    origin_train_dataset.data = origin_train_dataset.data[~(origin_train_dataset.labels == -1)]
    
    origin_train_dataset.labels = origin_train_dataset.labels[~(origin_train_dataset.labels == -1)]
    
    
    train_dataset, test_dataset, origin_train_dataset, origin_test_dataset = parition_chexpert_valid_test_dataset2(train_dataset, valid_dataset, origin_train_dataset, origin_val_dataset)
    
    
    # test_dataset = torch.load(full_output_dir + '/trans_test_dataset_new')
    #
    # test_dataset.labels = test_dataset.labels[:,selected_label_id]
    #
    # test_dataset.data = test_dataset.data[~(test_dataset.labels == -1)]
    #
    # test_dataset.labels = test_dataset.labels[~(test_dataset.labels == -1)]
    #
    #
    #
    #
    # train_dataset.data = train_dataset.data[~(train_dataset.labels == -1)]
    #
    # train_dataset.labels = train_dataset.labels[~(train_dataset.labels == -1)]
    
    
    # full_train_val_data = torch.cat([train_dataset.data, valid_dataset.data, test_dataset.data], dim = 0)
    #
    # full_train_val_labels = torch.cat([train_dataset.labels, valid_dataset.labels,test_dataset.labels], dim = 0)
    #
    # random_ids = torch.randperm(full_train_val_data.shape[0])
    #
    # val_size = int(random_ids.shape[0]*0.1)
    #
    # val_ids = random_ids[0:val_size]
    #
    # test_size = 0
    #
# #     int(random_ids.shape[0]*0.2)
# #     
# #     test_ids = random_ids[val_size:val_size+test_size]
    #
    # train_ids = random_ids[val_size+test_size:] 
    #
    #
    #
# #     updated_test_dataset = models.MyDataset(full_train_val_data[test_ids],full_train_val_labels[test_ids]) 
    #
    # print('train size::', train_ids.shape[0])
    #
    # print('valid size::', val_ids.shape[0])
    #
# #     print('test size::', test_ids.shape[0])
    #
    transformed_training_data, transformed_validation_data, transformed_test_data = normalize_data(train_dataset.data, valid_dataset.data, test_dataset.data)
    
    train_dataset.data, valid_dataset.data, test_dataset.data = transformed_training_data, transformed_validation_data, transformed_test_data 
    #
    # updated_train_dataset = models.MyDataset(transformed_training_data,full_train_val_labels[train_ids])
    #
    # updated_valid_dataset = models.MyDataset(transformed_validation_data,full_train_val_labels[val_ids])
    #
    # updated_test_dataset = models.MyDataset(transformed_test_data,test_dataset.labels)
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(test_dataset, batch_size=args.bz)
    
    torch.save(origin_train_dataset, full_output_dir + '/origin_train_dataset_new')
        
    torch.save(origin_val_dataset, full_output_dir + '/origin_val_dataset_new')
    
    torch.save(origin_test_dataset, full_output_dir + '/origin_test_dataset_new')
    
    
    print('train dataset shape::', train_dataset.data.shape, train_dataset.labels.shape)
    
    print('origin train dataset shape::', origin_train_dataset.data.shape, origin_train_dataset.labels.shape)
    
    print('valid dataset shape::', valid_dataset.data.shape, valid_dataset.labels.shape)
    
    print('origin valid dataset shape::', origin_val_dataset.data.shape, origin_val_dataset.labels.shape)
    
    print('test dataset shape::', test_dataset.data.shape, test_dataset.labels.shape)
    
    print('origin test dataset shape::', origin_test_dataset.data.shape, origin_test_dataset.labels.shape)
    
    binary = True
    
    print('load done')
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 

def obtain_mnist_examples(args, noisy=False):
    dataset_name = args.dataset
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
#     if not noisy:
#         train_dataset = torch.load(full_output_dir + '/preprocessed_train')
#         
#         valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
#         
#         test_dataset = torch.load(full_output_dir + '/preprocessed_test')
#     
#         train_DL = DataLoader(train_dataset, batch_size=args.bz)
#         
#         valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
#         
#         test_DL = DataLoader(test_dataset, batch_size=args.bz)
#         
#         train_DL.dataset.data = train_DL.dataset.data.view(train_DL.dataset.data.shape[0], -1)
#         
#         valid_DL.dataset.data = valid_DL.dataset.data.view(valid_DL.dataset.data.shape[0], -1)
#         
#         test_DL.dataset.data = test_DL.dataset.data.view(test_DL.dataset.data.shape[0], -1) 
#         
#         binary = True
#         
#         print('load done')
#         
#         return train_DL.dataset, valid_DL.dataset, test_DL.dataset, full_output_dir, binary 
#     
#     else:
    train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
    
    valid_dataset = torch.load(full_output_dir + '/validation_dataset')
    
    test_dataset = torch.load(full_output_dir + '/test_dataset')
    
    small_dataset = torch.load(full_output_dir + '/small_dataset')

    full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')

    return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir 


def obtain_oct_examples(args, noisy=False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir
    
    if not noisy:
        train_dataset = torch.load(full_output_dir + '/preprocessed_train')
        
        valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
        
        test_dataset = torch.load(full_output_dir + '/preprocessed_test')
    
        train_DL = DataLoader(train_dataset, batch_size=args.bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
        
        test_DL = DataLoader(test_dataset, batch_size=args.bz)
        
        train_DL.dataset.data = train_DL.dataset.data.view(train_DL.dataset.data.shape[0], -1)
        
        valid_DL.dataset.data = valid_DL.dataset.data.view(valid_DL.dataset.data.shape[0], -1)
        
        test_DL.dataset.data = test_DL.dataset.data.view(test_DL.dataset.data.shape[0], -1) 
        
        binary = True
        
        print('load done')
        
        return train_DL.dataset, valid_DL.dataset, test_DL.dataset, full_output_dir, binary 
    
    else:
        train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
        
        valid_dataset = torch.load(full_output_dir + '/validation_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
    
        full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
        
        small_dataset = None#torch.load(full_output_dir + '/small_dataset')
        
        selected_small_sample_ids = None
        selected_noisy_sample_ids = None
        
        
        if os.path.exists(full_output_dir + '/selected_small_sample_ids'):
            selected_small_sample_ids = torch.load(full_output_dir + '/selected_small_sample_ids')
        
        if os.path.exists(full_output_dir + '/selected_noisy_sample_ids'):
            selected_noisy_sample_ids = torch.load(full_output_dir + '/selected_noisy_sample_ids')
        
        train_dataset.labels = torch.tensor(train_dataset.labels.tolist()).type(torch.DoubleTensor)
    
        return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir, selected_small_sample_ids,selected_noisy_sample_ids  
    

def obtain_mimic_examples(args, noisy=False, load_origin = False, is_tars = False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir
    
    if not noisy:
        train_dataset = torch.load(full_output_dir + '/preprocessed_train')
        
        valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
        
        test_dataset = torch.load(full_output_dir + '/preprocessed_test')
    
        train_DL = DataLoader(train_dataset, batch_size=args.bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
        
        test_DL = DataLoader(test_dataset, batch_size=args.bz)
        
        train_DL.dataset.data = train_DL.dataset.data.view(train_DL.dataset.data.shape[0], -1)
        
        valid_DL.dataset.data = valid_DL.dataset.data.view(valid_DL.dataset.data.shape[0], -1)
        
        test_DL.dataset.data = test_DL.dataset.data.view(test_DL.dataset.data.shape[0], -1) 
        
        binary = True
        
        print('load done')
        
        return train_DL.dataset, valid_DL.dataset, test_DL.dataset, full_output_dir, binary 
    
    else:
        # train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
        
        if is_tars:
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset_tars')
        else:
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
        
        valid_dataset = torch.load(full_output_dir + '/validation_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
        
        origin_train_dataset = None
            
        origin_val_dataset = None
        
        origin_test_dataset = None
        
        if load_origin:
        
            origin_train_dataset = torch.load(full_output_dir + '/origin_train_dataset')
            
            origin_train_dataset.data = origin_train_dataset.data[(origin_train_dataset.labels != -1)]
            
            origin_train_dataset.labels = train_dataset.labels 
            
            origin_val_dataset = torch.load(full_output_dir + '/origin_val_dataset')
            
            origin_test_dataset = torch.load(full_output_dir + '/origin_test_dataset')
            
            print('train shape::', origin_train_dataset.data.shape, origin_train_dataset.labels.shape)
            
            print('valid shape::', origin_val_dataset.data.shape, origin_val_dataset.labels.shape)
            
            print('test shape::', origin_test_dataset.data.shape, origin_test_dataset.labels.shape)
    
        full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
        
        print('train shape', train_dataset.data.shape, train_dataset.labels.shape)
        
        small_dataset = None#torch.load(full_output_dir + '/small_dataset')
        
        selected_small_sample_ids = None
        selected_noisy_sample_ids = None
        
        
        if os.path.exists(full_output_dir + '/selected_small_sample_ids'):
            selected_small_sample_ids = torch.load(full_output_dir + '/selected_small_sample_ids')
        
        if os.path.exists(full_output_dir + '/selected_noisy_sample_ids'):
            selected_noisy_sample_ids = torch.load(full_output_dir + '/selected_noisy_sample_ids')
        
        train_dataset.labels = torch.tensor(train_dataset.labels.tolist()).type(torch.DoubleTensor)
    
        return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir, selected_small_sample_ids,selected_noisy_sample_ids, origin_train_dataset, origin_val_dataset, origin_test_dataset  
    

def obtain_retina_examples_origin_DL(args):
    dataset_name = args.dataset
    full_output_dir = os.path.join(args.output_dir, dataset_name)
    
    
#     torch.load(full_output_dir + '')
    
    if args.start:
        data_preparer = models.Data_preparer()
        train_DL, valid_DL = data_preparer.prepare_retina(full_output_dir, args.bz)
         
        test_DL = data_preparer.prepare_test_retina(full_output_dir, args.bz)
         
         
        torch.save(train_DL, full_output_dir + '/train_DL')
         
        torch.save(valid_DL, full_output_dir + '/valid_DL')
         
        torch.save(test_DL, full_output_dir + '/test_DL')
     
    else:
        train_DL = torch.load(full_output_dir + '/train_DL')
         
        valid_DL = torch.load(full_output_dir + '/valid_DL')
         
        test_DL = torch.load(full_output_dir + '/test_DL')
    
    binary = True
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 


def obtain_mimic_examples_origin_DL(args, origin = False):
    dataset_name = args.dataset
    full_output_dir = args.output_dir
    
    if args.start:
        data_preparer = models.Data_preparer()
        
        prepare_data_function = getattr(data_preparer, 'prepare_' + args.dataset.lower())
        
        train_DL, valid_DL, test_DL = prepare_data_function(full_output_dir, args.bz, origin = origin)
        
        if not origin:
        
            torch.save(train_DL, full_output_dir + '/train_DL')
             
            torch.save(valid_DL, full_output_dir + '/valid_DL')
             
            torch.save(test_DL, full_output_dir + '/test_DL')
        else:
            torch.save(train_DL, full_output_dir + '/train_DL_origin')
             
            torch.save(valid_DL, full_output_dir + '/valid_DL_origin')
             
            torch.save(test_DL, full_output_dir + '/test_DL_origin')
     
    else:
        if not origin:
            train_DL = torch.load(full_output_dir + '/train_DL')
             
            valid_DL = torch.load(full_output_dir + '/valid_DL')
             
            test_DL = torch.load(full_output_dir + '/test_DL')
            
        else:
            train_DL = torch.load(full_output_dir + '/train_DL_origin')
             
            valid_DL = torch.load(full_output_dir + '/valid_DL_origin')
             
            test_DL = torch.load(full_output_dir + '/test_DL_origin')
    
    binary = True
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary 
    
#     train_folder = os.path.join(full_output_dir, 'train')
#     
#     valid_folder = os.path.join(full_output_dir, 'train')
        
#     if not os.path.exists(full_output_dir):
#         os.makedirs(full_output_dir)
#         
#     training_dataset = torch.load(full_output_dir + '/trans_train_dataset')


def obtain_retina_examples(args, noisy = False, load_origin = False, is_tars = False):
#     disease_id = 1
    
    dataset_name = args.dataset
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    if not noisy:
        training_dataset = torch.load(full_output_dir + '/trans_train_dataset' + '_new')
        
    #     training_dataset.labels = training_dataset.labels[:,disease_id]
        
        val_dataset = torch.load(full_output_dir + '/trans_val_dataset' + '_new')
        
        dataset_test = torch.load(full_output_dir + '/trans_test_dataset' + '_new')
        
    #     val_dataset.labels = val_dataset.labels[:,disease_id]
        binary = False
        
        return training_dataset, val_dataset, dataset_test, full_output_dir, binary
    
    else:
        
        if is_tars:
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset_tars')
        else:
            
            train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
        
        train_dataset.labels = train_dataset.labels.type(torch.DoubleTensor)
        
        valid_dataset = torch.load(full_output_dir + '/validation_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
        
        small_dataset = None#torch.load(full_output_dir + '/small_dataset')
    
        full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
        
        origin_train_dataset = None
        
        origin_val_dataset = None
        
        origin_test_dataset = None
        
        if load_origin:
        
            origin_train_dataset = torch.load(full_output_dir + '/origin_train_dataset' + '_new')
            
            origin_val_dataset = torch.load(full_output_dir + '/origin_val_dataset' + '_new')
            
            origin_test_dataset = torch.load(full_output_dir + '/origin_test_dataset' + '_new')
            
            print('data shape::', origin_train_dataset.data.shape, origin_val_dataset.data.shape, origin_test_dataset.data.shape)
            # if torch.unique(train_dataset.labels).shape[0] > 2:
            #     origin_train_dataset.labels = (origin_train_dataset.labels >= 3).type(torch.LongTensor)
            #     origin_val_dataset.labels = (origin_val_dataset.labels >= 3).type(torch.LongTensor)
            #     origin_test_dataset.labels = (origin_test_dataset.labels >= 3).type(torch.LongTensor)
            origin_train_dataset.labels = train_dataset.labels

#         if not os.path.exists(full_output_dir + '/normalized'):
#             train_data , valid_data, test_data = normalize_data(train_dataset.data, valid_dataset.data, test_dataset.data)
#             train_dataset.data= train_data
#             valid_dataset.data = valid_data
#             test_dataset.data = test_data
# #             small_dataset.data = small_data
#             
#             torch.save(train_dataset, full_output_dir + '/full_training_noisy_dataset_normalized')
#         
#             torch.save(valid_dataset, full_output_dir + '/validation_dataset_normalized')
#             
#             torch.save(test_dataset, full_output_dir + '/test_dataset_normalized')
#             
#             torch.save(small_dataset, full_output_dir + '/small_dataset_normalized')
#         else:
#             train_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset_normalized')
#         
#             valid_dataset = torch.load(full_output_dir + '/validation_dataset_normalized')
#             
#             test_dataset = torch.load(full_output_dir + '/test_dataset_normalized')
#             
#             small_dataset = torch.load(full_output_dir + '/small_dataset_normalized')
#             
#             normalized = True
#             
#             torch.save(normalized, full_output_dir + '/normalized')

    
        selected_small_sample_ids = None
        selected_noisy_sample_ids = None
    
        
    
        return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir, selected_small_sample_ids,selected_noisy_sample_ids, origin_train_dataset, origin_val_dataset, origin_test_dataset

#         return train_dataset, full_training_origin_labels, valid_dataset, test_dataset, small_dataset, full_output_dir

def obtain_mnist_examples_del(args):
    
    dataset_name = args.dataset
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
        
    data_preparer = models.Data_preparer()
    
    if args.start:
    
        dataset_train, dataset_test = get_train_test_data_loader_by_name_lr(data_preparer, args.output_dir, dataset_name)
        
        train_dataset, val_dataset = partition_val_dataset(dataset_train, ratio=0.1)
    
        torch.save(train_dataset, full_output_dir + '/train_dataset')
        
        torch.save(val_dataset, full_output_dir + '/val_dataset')
        
        torch.save(dataset_test, full_output_dir + '/dataset_test')
    
    else:
        train_dataset = torch.load(full_output_dir + '/train_dataset')
        
        val_dataset = torch.load(full_output_dir + '/val_dataset')
        
        dataset_test = torch.load(full_output_dir + '/dataset_test')
    
#     dataset_val =
    
    
#     full_training_noisy_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
#         
#     full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
#     
#     validation_dataset = torch.load(full_output_dir + '/validation_dataset')
#     
#     dataset_test = torch.load(full_output_dir + '/test_dataset')
    binary = False
    return train_dataset, val_dataset, dataset_test, full_output_dir, binary


def obtain_cifar10_examples_origin(args):
    
    dataset_name = args.dataset
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
        
    data_preparer = models.Data_preparer()
    
    if args.start:
    
        dataset_train, dataset_test = get_train_test_data_loader_by_name_lr(data_preparer, args.output_dir, dataset_name)
        
        train_dataset, val_dataset = partition_val_dataset(dataset_train, ratio=0.1)
    
        torch.save(train_dataset, full_output_dir + '/train_dataset')
        
        torch.save(val_dataset, full_output_dir + '/val_dataset')
        
        torch.save(dataset_test, full_output_dir + '/dataset_test')
    
    else:
        train_dataset = torch.load(full_output_dir + '/train_dataset')
        
        val_dataset = torch.load(full_output_dir + '/val_dataset')
        
        dataset_test = torch.load(full_output_dir + '/dataset_test')
    
#     dataset_val =
    
    
#     full_training_noisy_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
#         
#     full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
#     
#     validation_dataset = torch.load(full_output_dir + '/validation_dataset')
#     
#     dataset_test = torch.load(full_output_dir + '/test_dataset')
    binary = False
    
    train_DL = DataLoader(train_dataset, batch_size=args.bz)
    
    valid_DL = DataLoader(val_dataset, batch_size=args.bz)
    
    test_DL = DataLoader(dataset_test, batch_size=args.bz)
    
    return train_DL, valid_DL, test_DL, full_output_dir, binary

def obtain_cifar10_examples(args):
    
    dataset_name = args.dataset
    
    full_output_dir = os.path.join(args.output_dir, dataset_name)
        
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
        
    data_preparer = models.Data_preparer()
    
    if args.start:
    
        train_dataset, val_dataset, dataset_test = get_train_test_data_loader_by_name_lr(data_preparer, args.output_dir, dataset_name)
        
#         train_dataset, val_dataset = partition_val_dataset(dataset_train, ratio=0.05)
    
        torch.save(train_dataset, full_output_dir + '/train_dataset')
        
        torch.save(val_dataset, full_output_dir + '/val_dataset')
        
        torch.save(dataset_test, full_output_dir + '/dataset_test')
    
    else:
        train_dataset = torch.load(full_output_dir + '/train_dataset')
        
        val_dataset = torch.load(full_output_dir + '/val_dataset')
        
        dataset_test = torch.load(full_output_dir + '/dataset_test')
    
#     dataset_val =
    
    
#     full_training_noisy_dataset = torch.load(full_output_dir + '/full_training_noisy_dataset')
#         
#     full_training_origin_labels = torch.load(full_output_dir + '/full_training_origin_labels')
#     
#     validation_dataset = torch.load(full_output_dir + '/validation_dataset')
#     
#     dataset_test = torch.load(full_output_dir + '/test_dataset')
    binary = False
    
#     train_DL = DataLoader(train_dataset, batch_size=args.bz)
#     
#     valid_DL = DataLoader(val_dataset, batch_size=args.bz)
#     
#     test_DL = DataLoader(dataset_test, batch_size=args.bz)
    
    return train_dataset, val_dataset, dataset_test, full_output_dir, binary

def get_loss_n_accuracy(model, data_loader, args, num_classes=2):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
#     criterion.reduction = 'mean'

    criterion = model.get_loss_function()
    criterion2 = model.get_loss_function(f1 = args.f1)
    model.eval()                                     
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels, ids) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
    
        # compute the total loss over minibatch
        outputs = model(inputs)
        pred_labels = model.determine_labels(inputs)
        
        if not args.f1:
            avg_minibatch_loss = criterion(outputs, labels)
        else:
            avg_minibatch_loss = criterion2(pred_labels, labels, r_weight = None, is_training=True, num_class = args.num_class)
        total_loss += avg_minibatch_loss.detach().cpu().item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
#         pred_labels = (F.sigmoid(outputs) > 0.5).int()

        

#         pred_labels = (outputs > 0.5).int()
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels.view(-1), labels.view(-1))).detach().cpu().item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)

def reweight_alg(train_dataset, meta_dataset, test_dataset, args, model, full_output_dir):
    
    train_loader = DataLoader(train_dataset, batch_size=args.bz, shuffle=True)
    test_loader =  DataLoader(test_dataset, batch_size=args.bz, shuffle=False)
    meta_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
    valid_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
    
    meta_loader = itertools.cycle(meta_loader)
    
    
    w_array =torch.rand([train_dataset.data.shape[0]], dtype = train_dataset.data.dtype, requires_grad=True)
    
    mu_array = torch.rand([train_dataset.data.shape[0]], dtype = train_dataset.data.dtype, requires_grad=True)
    
    beta_array = torch.rand([train_dataset.data.shape[0]], dtype = train_dataset.data.dtype, requires_grad=True)
    
    gamma = torch.rand(1, dtype = train_dataset.data.dtype, requires_grad=True, device = args.device)
    
    removed_count = args.removed_count
    
    lr = args.lr
    
    training_data_count = train_dataset.data.shape[0]
    
    criterion2 = model.get_loss_function(f1 = args.f1)
    
    criterion = model.soft_loss_function
    
    
    
    opt = model.get_optimizer(args.tlr, args.wd)
#     args['run_epochs'] = 10000
    
#     sum_upper_bound = 0.5
#     
#     single_upper_bound = 0.01
    
    upper_lower_bound_satisfied = False
    
    pid = os.getpid()
    
    print('process ID::', pid)
    
    for ep in range(1, args.out_epoch_count+1):
        model.train()
        
        train_loss = 0
        
        max_w = None
        
        min_w = None
        
        for iter, (inputs, labels, ids) in enumerate(train_loader):
            inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                                labels.to(device=args.device, non_blocking=True)
            
            w_array.requires_grad = True
            mu_array.requires_grad = True
            
            beta_array.requires_grad = True
            
            gamma.requires_grad = True
            
            
            opt.zero_grad()
            with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
                # 1. Update meta model on training data
                
                
                eps = w_array[ids]
                
                if args.GPU:
                    eps = eps.to(args.device)
#                 criterion.reduction = 'none'
                if not args.f1:
                    
                    meta_train_outputs = meta_model(inputs)
                    
                    meta_train_loss = criterion(meta_train_outputs, labels.type_as(meta_train_outputs))
                    meta_train_loss = torch.sum(eps.view(-1) * meta_train_loss.view(-1))/torch.sum(eps)
                else:
                    meta_train_outputs = meta_model.determine_labels(inputs, soft=True)
                    
                    meta_train_loss = criterion2(meta_train_outputs, labels.type_as(meta_train_outputs), r_weight = eps.view(-1), is_training=True, num_class = args.num_class)
#                 eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=args['device'])


                
                
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                meta_inputs, meta_labels, meta_ids =  next(meta_loader)
                meta_inputs, meta_labels = meta_inputs.to(device=args.device, non_blocking=True),\
                                 meta_labels.to(device=args.device, non_blocking=True)
    
    
                
#                 criterion.reduction = 'mean'
#                 criterion.reduction = 'none'
#                 meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(meta_val_outputs))
                if not args.f1:            
                    
                    meta_val_outputs = meta_model(meta_inputs)
                    
                    meta_val_loss = criterion2(meta_val_outputs, meta_labels)
                else:
                    meta_val_outputs = meta_model.determine_labels(meta_inputs, soft=True)
                    
                    meta_val_loss = criterion2(meta_val_outputs, meta_labels.type_as(meta_val_outputs), r_weight = None, is_training=True, num_class = args.num_class)
                
                sub_mu_array = mu_array[ids]
                
                sub_beta_array = beta_array[ids]
                
                if args.GPU:
                    sub_mu_array = sub_mu_array.to(args.device)
                    sub_beta_array = sub_beta_array.to(args.device)
                
#                 meta_val_loss = torch.sum(eps.view(-1) * meta_val_loss.view(-1) - eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                total_meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                
                eps_grads = torch.autograd.grad(total_meta_val_loss, eps, retain_graph=True)[0].detach()
                
#                 mu_grads = torch.autograd.grad(meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
#                 
#                 beta_grads = torch.autograd.grad(meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
#                 
#                 gamma_grads = torch.autograd.grad(meta_val_loss, gamma)[0].detach()
    
            # 3. Compute weights for current training batch
            w_array.requires_grad = False
            
            w_array[ids] = w_array[ids]-lr*eps_grads.cpu()
            
            global sum_upper_bound, single_upper_bound
            
            if max_w is None:
                max_w = torch.max(w_array).item()
                min_w = torch.min(w_array).item()
            else:
                max_w = max(torch.max(w_array).item(), max_w)
                min_w = min(torch.min(w_array).item(), min_w)
            
            if torch.abs(training_data_count - removed_count - torch.sum(w_array)) < sum_upper_bound and torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                lr = lr /2
                sum_upper_bound = sum_upper_bound/5
                single_upper_bound = single_upper_bound/5 
            
            if torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                upper_lower_bound_satisfied = True
            
            else:
                upper_lower_bound_satisfied = False
            
            
            total_meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
            
            
            mu_grads = torch.autograd.grad(total_meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
                
            beta_grads = torch.autograd.grad(total_meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
            
            gamma_grads = torch.autograd.grad(total_meta_val_loss, gamma)[0].detach()
            
            
            mu_array.requires_grad = False
            
            beta_array.requires_grad = False
            
            gamma.requires_grad = False
            
            mu_array[ids] = torch.clamp(mu_array[ids] + lr*mu_grads.cpu(), min = 0)
             
            beta_array[ids] = torch.clamp(beta_array[ids] + lr*beta_grads.cpu(), min = 0)
             
            gamma = torch.clamp(gamma + lr*gamma_grads, min = -1, max = 1)
            
            

            
#             mu_array[ids] = torch.clamp(mu_array[ids] + lr*(-w_array[ids]/torch.sum(w_array[ids])), min = 0)
#              
#             beta_array[ids] = torch.clamp(beta_array[ids] + lr*((w_array[ids]-1)/torch.sum(w_array[ids])), min = 0)
#              
#             gamma = gamma + lr*(torch.sum((training_data_count - removed_count)/training_data_count-w_array[ids])/torch.sum(w_array[ids]))
            
#             w_array[ids] = torch.clamp(w_array[ids]-lr*eps_grads, min=0, max=1)
            
#             w_array[ids] = w_array[ids]-lr*eps_grads
            
            w_array[ids] = torch.clamp(w_array[ids], min=0, max=1)
            
#             l1_norm = torch.sum(w_array)
#             if l1_norm != 0:
#                 w_array = w_array / l1_norm
#                 
#             w_array[ids] = torch.clamp(w_array[ids], min=0, max=1)
#             else:
#                 w = w_array
    
#             w_array[ids] = w_tilde.view(-1)
    
            # 4. Train model on weighted batch
            outputs = model(inputs)
#             criterion.reduction = 'none'
            curr_w = w_array[ids].view(-1)
            
            if args.GPU:
                curr_w = curr_w.to(args.device)

            
            if not args.f1:
                minibatch_loss = criterion(outputs, labels.type_as(outputs))
            
            
                minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
            else:
                
                outputs = model.determine_labels(inputs, soft=True)
                
                minibatch_loss = criterion2(outputs, labels.type_as(outputs), r_weight = curr_w, is_training=True, num_class = args.num_class)
            
            train_loss += (minibatch_loss.detach().cpu()*torch.sum(curr_w.detach().cpu())).item()
            minibatch_loss.backward()
            opt.step()
    
            # keep track of epoch loss/accuracy
            
#             pred_labels = (F.sigmoid(outputs) > 0.5).int()
            
#             pred_labels = (outputs > 0.5).int()
#             
#             train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
    
    
    
        print('epochs::', ep)
    
#         for k in range(10):
#              
#             train_loss, train_acc = 0, 0
#              
#             for iter, (inputs, labels, ids) in enumerate(train_loader):
#                 inputs, labels = inputs.to(device=args.device, non_blocking=True),\
#                                     labels.to(device=args.device, non_blocking=True)
#                  
#             
#                 curr_w = w_array[ids].view(-1)
#                 
#                 if args.GPU:
#                     curr_w = curr_w.to(args.device)
#                  
#                 opt.zero_grad()
#                 outputs = model(inputs)
#     #             criterion.reduction = 'none'
#                 minibatch_loss = criterion(outputs, labels.type_as(outputs))
#                 minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
#                 minibatch_loss.backward()
#                 opt.step()
#                 train_loss += minibatch_loss.detach().cpu().item()*torch.sum(curr_w.detach().cpu())
#                  
#             print('full train loss::', k, train_loss/torch.sum(w_array))
    
        # inference after epoch
        with torch.no_grad():
            train_loss = train_loss/torch.sum(w_array)#, train_acc/len(train_dataset)      
            valid_loss, (valid_acc, valid_per_class_acc) = get_loss_n_accuracy(model, valid_loader, args, args.num_class)
            test_loss, (test_acc, test_per_class_acc) = get_loss_n_accuracy(model, test_loader, args, args.num_class)                                  
            # log/print data
#             writer.add_scalar('Test/Loss', test_loss, ep)
#             writer.add_scalar('Test/Accuracy', test_acc, ep)
#             writer.add_scalar('Training/Loss', train_loss, ep)
#             writer.add_scalar('Training/Accuracy', train_acc, ep)
            print(f'|Train/Valid/Test Loss: {train_loss:.4f} / {valid_loss:.4f} / {test_loss:.4f}|', end='--')
            print(f'|valid acc/Test Acc: {valid_acc: 4f} / {test_acc:.4f}|', end='\r')
    
        sorted_w_array, sorted_ids = torch.sort(w_array.view(-1), descending=True)
        
        print('sorted array::')
        
        print(sorted_w_array[0:removed_count])
        
        print(sorted_w_array[train_dataset.data.shape[0] - removed_count:])
        
        print('sorted array ids::')
        
        print(sorted_ids[train_dataset.data.shape[0] - removed_count:])
        
#         full_output_dir = os.path.join(args.output_dir, args.dataset)
        
        torch.save(sorted_ids[train_dataset.data.shape[0] - removed_count:], full_output_dir + '/removed_ids_' + str(pid) + '_' + str(ep))
        
#         print(sorted_ids[0:train_dataset.data.shape[0] - removed_count])
        print('learning rate::', lr)
        
        print('r_sum_gap::', torch.sum(w_array) - (training_data_count - removed_count))    
        
        print('mu::', torch.max(mu_array), torch.min(mu_array))
        
        print('beta::', torch.max(beta_array), torch.min(beta_array))
        
        print('w::', max_w, min_w)
        
        print('gamma::', gamma)    
        
        print('here')
        
        if upper_lower_bound_satisfied and torch.abs(torch.sum(w_array) - (training_data_count - removed_count)) < final_sum_upper_bound:
            break


def reweight_alg_delete(train_dataset, meta_dataset, test_dataset, training_DL, val_DL, test_DL, args, model, full_output_dir):
    
    training_data_count = 0
    
    if training_DL is None:
        train_dataset.lenth = train_dataset.data.shape[0]
        train_loader = DataLoader(train_dataset, batch_size=args.bz, shuffle=True)
        
        if test_dataset is not None:
            test_dataset.lenth = test_dataset.data.shape[0]
            test_loader =  DataLoader(test_dataset, batch_size=args.bz, shuffle=False)
        else:
            test_loader = None
            
        meta_dataset.lenth = meta_dataset.data.shape[0]
        meta_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
        valid_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
        
        training_data_count = train_dataset.data.shape[0]
        training_data_type = train_dataset.data.dtype
        meta_loader = itertools.cycle(meta_loader)
    else:
        train_loader = training_DL
        
        valid_loader = val_DL
        
        meta_loader = val_DL
        
        if test_DL is not None:
            test_loader =  test_DL
        else:
            test_loader = None
        training_data_count = training_DL.dataset.lenth
        
        meta_loader = itertools.cycle(meta_loader)
        
        meta_sample, _, _ = next(meta_loader)
        
        training_data_type = meta_sample.dtype
        
    
    w_array =torch.rand([training_data_count], dtype = training_data_type, requires_grad=True)
    
    mu_array = torch.rand([training_data_count], dtype = training_data_type, requires_grad=True)
    
    beta_array = torch.rand([training_data_count], dtype = training_data_type, requires_grad=True)
    
    gamma = torch.rand(1, dtype = training_data_type, requires_grad=True, device = args.device)
    
    removed_count = args.removed_count
    
    lr = args.lr
    
#     training_data_count = train_dataset.data.shape[0]
    
    
    criterion = model.soft_loss_function
    
    criterion2 = model.get_loss_function()
    
    opt = model.get_optimizer(args.tlr, args.wd)
#     args['run_epochs'] = 10000
    
#     sum_upper_bound = 0.5
#     
#     single_upper_bound = 0.01
    
    upper_lower_bound_satisfied = False
    
    pid = os.getpid()
    
    print('process ID::', pid)
    
    for ep in range(1, args.out_epoch_count+1):
        model.train()
        
        train_loss = 0
        
        max_w = None
        
        min_w = None
        
        for iter, (inputs, labels, ids) in enumerate(train_loader):
            inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                                labels.to(device=args.device, non_blocking=True)
            
            w_array.requires_grad = True
            mu_array.requires_grad = True
            
            beta_array.requires_grad = True
            
            gamma.requires_grad = True
            
            
            opt.zero_grad()
            with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
                # 1. Update meta model on training data
                meta_train_outputs = meta_model(inputs)
                
                
                
                criterion2.reduction = 'none'
                meta_train_loss = criterion2(meta_train_outputs, labels)
#                 eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=args['device'])

                eps = w_array[ids]
                
                if args.GPU:
                    eps = eps.to(args.device)
                
#                 t1 = time.time()
                meta_train_loss = torch.sum(eps.view(-1) * meta_train_loss.view(-1))/torch.sum(eps)
#                 t2 = time.time()
#                 meta_train_loss2 = torch.sum(torch.mul(eps.view(-1), meta_train_loss.view(-1)))/torch.sum(eps)
#                 t3 = time.time()
#                 
#                 print('time1::', t2 - t1)
#                 
#                 print('time2::', t3 - t2)
                
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                meta_inputs, meta_labels, meta_ids =  next(meta_loader)
                meta_inputs, meta_labels = meta_inputs.to(device=args.device, non_blocking=True),\
                                 meta_labels.to(device=args.device, non_blocking=True)
    
                meta_val_outputs = meta_model(meta_inputs)
#                 criterion.reduction = 'mean'
#                 criterion.reduction = 'none'
#                 meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(meta_val_outputs))
                
                criterion2.reduction = 'mean'
                meta_val_loss = criterion2(meta_val_outputs, meta_labels)
                
                sub_mu_array = mu_array[ids]
                
                sub_beta_array = beta_array[ids]
                
                if args.GPU:
                    sub_mu_array = sub_mu_array.to(args.device)
                    sub_beta_array = sub_beta_array.to(args.device)
                
#                 meta_val_loss = torch.sum(eps.view(-1) * meta_val_loss.view(-1) - eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                total_meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                
                eps_grads = torch.autograd.grad(total_meta_val_loss, eps, retain_graph=True)[0].detach()
                
#                 mu_grads = torch.autograd.grad(meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
#                 
#                 beta_grads = torch.autograd.grad(meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
#                 
#                 gamma_grads = torch.autograd.grad(meta_val_loss, gamma)[0].detach()
    
            # 3. Compute weights for current training batch
            w_array.requires_grad = False
            
            w_array[ids] = w_array[ids]-lr*eps_grads.cpu()
            
            global sum_upper_bound, single_upper_bound
            
            if max_w is None:
                max_w = torch.max(w_array).item()
                min_w = torch.min(w_array).item()
            else:
                max_w = max(torch.max(w_array).item(), max_w)
                min_w = min(torch.min(w_array).item(), min_w)
            
            if torch.abs(training_data_count - removed_count - torch.sum(w_array)) < sum_upper_bound and torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                lr = lr /2
                sum_upper_bound = sum_upper_bound/5
                single_upper_bound = single_upper_bound/5 
            
            if torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                upper_lower_bound_satisfied = True
            
            else:
                upper_lower_bound_satisfied = False
            
            
            total_meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
            
            
            mu_grads = torch.autograd.grad(total_meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
                
            beta_grads = torch.autograd.grad(total_meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
            
            gamma_grads = torch.autograd.grad(total_meta_val_loss, gamma)[0].detach()
            
            
            mu_array.requires_grad = False
            
            beta_array.requires_grad = False
            
            gamma.requires_grad = False
            
            mu_array[ids] = torch.clamp(mu_array[ids] + lr*mu_grads.cpu(), min = 0)
             
            beta_array[ids] = torch.clamp(beta_array[ids] + lr*beta_grads.cpu(), min = 0)
             
            gamma = torch.clamp(gamma + lr*gamma_grads, min = -1, max = 1)
            
            

            
#             mu_array[ids] = torch.clamp(mu_array[ids] + lr*(-w_array[ids]/torch.sum(w_array[ids])), min = 0)
#              
#             beta_array[ids] = torch.clamp(beta_array[ids] + lr*((w_array[ids]-1)/torch.sum(w_array[ids])), min = 0)
#              
#             gamma = gamma + lr*(torch.sum((training_data_count - removed_count)/training_data_count-w_array[ids])/torch.sum(w_array[ids]))
            
#             w_array[ids] = torch.clamp(w_array[ids]-lr*eps_grads, min=0, max=1)
            
#             w_array[ids] = w_array[ids]-lr*eps_grads
            
            w_array[ids] = torch.clamp(w_array[ids], min=1e-7, max=1)
            
#             l1_norm = torch.sum(w_array)
#             if l1_norm != 0:
#                 w_array = w_array / l1_norm
#                 
#             w_array[ids] = torch.clamp(w_array[ids], min=0, max=1)
#             else:
#                 w = w_array
    
#             w_array[ids] = w_tilde.view(-1)
    
            # 4. Train model on weighted batch
            outputs = model(inputs)
            criterion2.reduction = 'none'
            minibatch_loss = criterion2(outputs, labels)
            
            curr_w = w_array[ids].view(-1)
            
            if args.GPU:
                curr_w = curr_w.to(args.device)
            
            minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
            
            train_loss += (minibatch_loss.detach().cpu()*torch.sum(curr_w.detach().cpu())).item()
            minibatch_loss.backward()
            opt.step()
    
            # keep track of epoch loss/accuracy
            
#             pred_labels = (F.sigmoid(outputs) > 0.5).int()
            
#             pred_labels = (outputs > 0.5).int()
#             
#             train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
    
    
    
        print('epochs::', ep)
    
#         for k in range(1):
#               
#             train_loss, train_acc = 0, 0
#               
#             for iter, (inputs, labels, ids) in enumerate(train_loader):
#                 inputs, labels = inputs.to(device=args.device, non_blocking=True),\
#                                     labels.to(device=args.device, non_blocking=True)
#                   
#              
#                 curr_w = w_array[ids].view(-1)
#                  
#                 if args.GPU:
#                     curr_w = curr_w.to(args.device)
#                   
#                 opt.zero_grad()
#                 outputs = model(inputs)
#                 criterion2.reduction = 'none'
#                 minibatch_loss = criterion2(outputs, labels)
#                 minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
#                 minibatch_loss.backward()
#                 opt.step()
#                 train_loss += minibatch_loss.detach().cpu().item()*torch.sum(curr_w.detach().cpu())
#                   
#             print('full train loss::', k, train_loss/torch.sum(w_array))
    
        # inference after epoch
        with torch.no_grad():
            train_loss = train_loss/torch.sum(w_array)#, train_acc/len(train_dataset)      
            valid_loss, (valid_acc, valid_per_class_acc) = get_loss_n_accuracy(model, valid_loader, args, args.num_class)
            if test_loader is not None:
                test_loss, (test_acc, test_per_class_acc) = get_loss_n_accuracy(model, test_loader, args, args.num_class)                                  
            # log/print data
#             writer.add_scalar('Test/Loss', test_loss, ep)
#             writer.add_scalar('Test/Accuracy', test_acc, ep)
#             writer.add_scalar('Training/Loss', train_loss, ep)
#             writer.add_scalar('Training/Accuracy', train_acc, ep)
                print(f'|Train/Valid/Test Loss: {train_loss:.4f} / {valid_loss:.4f} / {test_loss:.4f}|', end='--')
                print(f'|valid acc/Test Acc: {valid_acc: 4f} / {test_acc:.4f}|', end='\r')
            else:
                print(f'|Train/Valid: {train_loss:.4f} / {valid_loss:.4f}|', end='--')
                print(f'|valid acc: {valid_acc: 4f} /', end='\r')
    
        sorted_w_array, sorted_ids = torch.sort(w_array.view(-1), descending=True)
        
        print('sorted array::')
        
        print(sorted_w_array[0:removed_count])
        
        print(sorted_w_array[training_data_count - removed_count:])
        
        print('sorted array ids::')
        
        print(sorted_ids[training_data_count - removed_count:])
        
        
        
        torch.save(sorted_ids[training_data_count - removed_count:], full_output_dir + '/removed_ids_' + str(pid) + '_' + str(ep))
        
#         print(sorted_ids[0:train_dataset.data.shape[0] - removed_count])
        print('learning rate::', lr)
        
        print('r_sum_gap::', torch.sum(w_array) - (training_data_count - removed_count))    
        
        print('mu::', torch.max(mu_array), torch.min(mu_array))
        
        print('beta::', torch.max(beta_array), torch.min(beta_array))
        
        print('w::', max_w, min_w)
        
        print('gamma::', gamma)    
        
        print('here')
        
        if upper_lower_bound_satisfied and torch.abs(torch.sum(w_array) - (training_data_count - removed_count)) < final_sum_upper_bound:
            break

def unlabeled_loss(unlabeled_X, labeled_X, exp_labeled_output, eps, model):
    
    unlabeled_X_output_before_last_layer = model.forward_before_last_layer(unlabeled_X)
    
    labeled_X_output_before_last_layer = model.forward_before_last_layer(labeled_X)
    
    distances = torch.exp(-transductive_coeff*torch.sum((unlabeled_X_output_before_last_layer.view(unlabeled_X_output_before_last_layer.shape[0], 1, unlabeled_X_output_before_last_layer.shape[1]) - labeled_X_output_before_last_layer.view(1, labeled_X_output_before_last_layer.shape[0], labeled_X_output_before_last_layer.shape[1]))**2, dim = -1))
    
    meta_outputs = model.forward_from_before_last_layer(unlabeled_X_output_before_last_layer)
    
    labeled_output = model.forward_from_before_last_layer(labeled_X_output_before_last_layer)
    
#     if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
    prediction_diff = torch.sum((meta_outputs.view(meta_outputs.shape[0], 1, meta_outputs.shape[1]) - labeled_output.view(1, labeled_output.shape[0], labeled_output.shape[1]))**2*eps.view(-1,1,1), dim = -1)
    
    return torch.sum(distances*prediction_diff)/(labeled_output.shape[0]*torch.sum(eps))

def reweight_alg_transductive(labeled_train_dataset, unlabeled_train_dataset, meta_dataset, test_dataset, args, model):
    
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.bz, shuffle=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=int(args.bz/2), shuffle=True)
    test_loader =  DataLoader(test_dataset, batch_size=args.bz, shuffle=False)
    meta_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
    valid_loader = DataLoader(meta_dataset, batch_size=args.bz, shuffle=True)
    
    meta_loader = itertools.cycle(meta_loader)
    unlabeled_train_loader = itertools.cycle(unlabeled_train_loader)
    
    w_array =torch.rand([unlabeled_train_dataset.data.shape[0]], dtype = unlabeled_train_dataset.data.dtype, requires_grad=True)
    
    mu_array = torch.rand([unlabeled_train_dataset.data.shape[0]], dtype = unlabeled_train_dataset.data.dtype, requires_grad=True)
    
    beta_array = torch.rand([unlabeled_train_dataset.data.shape[0]], dtype = unlabeled_train_dataset.data.dtype, requires_grad=True)
    
    gamma = torch.rand(1, dtype = unlabeled_train_dataset.data.dtype, requires_grad=True, device = args.device)
    
    removed_count = args.removed_count
    
    lr = args.lr
    
    training_data_count = labeled_train_dataset.data.shape[0]
    unlabeled_training_data_count = unlabeled_train_dataset.data.shape[0]
    
    criterion = model.soft_loss_function
    
    criterion2 = model.get_loss_function()
    
    opt = model.get_optimizer(args.tlr, args.wd)
#     args['run_epochs'] = 10000
    
#     sum_upper_bound = 0.5
#     
#     single_upper_bound = 0.01
    
    upper_lower_bound_satisfied = False
    
    pid = os.getpid()
    
    print('process ID::', pid)
    
    for ep in range(1, args.out_epoch_count+1):
        model.train()
        
        train_loss = 0
        
        max_w = None
        
        min_w = None
        
        
        for iter, (inputs, labels, _) in enumerate(labeled_train_loader):
            inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                                labels.to(device=args.device, non_blocking=True)
            
            w_array.requires_grad = True
            mu_array.requires_grad = True
            
            beta_array.requires_grad = True
            
            gamma.requires_grad = True
            
            
            opt.zero_grad()
            with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
                # 1. Update meta model on training data
                meta_train_outputs = meta_model(inputs)
#                 criterion.reduction = 'none'
                meta_train_loss = criterion2(meta_train_outputs, labels)
#                 eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=args['device'])

#                 meta_train_loss = torch.sum(eps.view(-1) * meta_train_loss.view(-1))/torch.sum(eps)

                unlabeled_x, unlabeled_y, ids = next(unlabeled_train_loader)

                eps = w_array[ids]
                
                if args.GPU:
                    eps = eps.to(args.device)
                    
                    unlabeled_x = unlabeled_x.to(args.device)
                    
                    unlabeled_y = unlabeled_y.to(args.device)

                meta_train_loss += unlabeled_loss(unlabeled_x, inputs, meta_train_outputs, eps, model)
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                meta_inputs, meta_labels, meta_ids =  next(meta_loader)
                meta_inputs, meta_labels = meta_inputs.to(device=args.device, non_blocking=True),\
                                 meta_labels.to(device=args.device, non_blocking=True)
    
                meta_val_outputs = meta_model(meta_inputs)
#                 criterion.reduction = 'mean'
#                 criterion.reduction = 'none'
#                 meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(meta_val_outputs))
                
                meta_val_loss = criterion2(meta_val_outputs, meta_labels)
                
                sub_mu_array = mu_array[ids]
                
                sub_beta_array = beta_array[ids]
                
                if args.GPU:
                    sub_mu_array = sub_mu_array.to(args.device)
                    sub_beta_array = sub_beta_array.to(args.device)
                
#                 meta_val_loss = torch.sum(eps.view(-1) * meta_val_loss.view(-1) - eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((unlabeled_training_data_count - removed_count)/unlabeled_training_data_count - eps))/torch.sum(eps)
                
                
                eps_grads = torch.autograd.grad(meta_val_loss, eps, retain_graph=True)[0].detach()
                
#                 mu_grads = torch.autograd.grad(meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
#                 
#                 beta_grads = torch.autograd.grad(meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
#                 
#                 gamma_grads = torch.autograd.grad(meta_val_loss, gamma)[0].detach()
    
            # 3. Compute weights for current training batch
            w_array.requires_grad = False
            
            w_array[ids] = w_array[ids]-lr*eps_grads.cpu()
            
            global sum_upper_bound, single_upper_bound
            
            if max_w is None:
                max_w = torch.max(w_array).item()
                min_w = torch.min(w_array).item()
            else:
                max_w = max(torch.max(w_array).item(), max_w)
                min_w = min(torch.min(w_array).item(), min_w)
            
            if torch.abs(unlabeled_training_data_count - removed_count - torch.sum(w_array)) < sum_upper_bound and torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                lr = lr /2
                sum_upper_bound = sum_upper_bound/5
                single_upper_bound = single_upper_bound/5 
            
            
            if torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
                upper_lower_bound_satisfied = True
            
            else:
                upper_lower_bound_satisfied = False
            
            meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((unlabeled_training_data_count - removed_count)/unlabeled_training_data_count - eps))/torch.sum(eps)
            
            
            mu_grads = torch.autograd.grad(meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
                
            beta_grads = torch.autograd.grad(meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
            
            gamma_grads = torch.autograd.grad(meta_val_loss, gamma)[0].detach()
            
            
            mu_array.requires_grad = False
            
            beta_array.requires_grad = False
            
            gamma.requires_grad = False
            
            mu_array[ids] = torch.clamp(mu_array[ids] + lr*mu_grads.cpu(), min = 0)
             
            beta_array[ids] = torch.clamp(beta_array[ids] + lr*beta_grads.cpu(), min = 0)
             
            gamma = torch.clamp(gamma + lr*gamma_grads, min = -1, max = 1)
            
            

            
#             mu_array[ids] = torch.clamp(mu_array[ids] + lr*(-w_array[ids]/torch.sum(w_array[ids])), min = 0)
#              
#             beta_array[ids] = torch.clamp(beta_array[ids] + lr*((w_array[ids]-1)/torch.sum(w_array[ids])), min = 0)
#              
#             gamma = gamma + lr*(torch.sum((training_data_count - removed_count)/training_data_count-w_array[ids])/torch.sum(w_array[ids]))
            
#             w_array[ids] = torch.clamp(w_array[ids]-lr*eps_grads, min=0, max=1)
            
#             w_array[ids] = w_array[ids]-lr*eps_grads
            
            w_array[ids] = torch.clamp(w_array[ids], min=0, max=1)
            
#             l1_norm = torch.sum(w_array)
#             if l1_norm != 0:
#                 w_array = w_array / l1_norm
#                 
#             w_array[ids] = torch.clamp(w_array[ids], min=0, max=1)
#             else:
#                 w = w_array
    
#             w_array[ids] = w_tilde.view(-1)
    
            # 4. Train model on weighted batch
            outputs = model(inputs)
#             criterion.reduction = 'none'
            minibatch_loss = criterion2(outputs, labels)
            
            curr_w = w_array[ids].view(-1)
            
            if args.GPU:
                curr_w = curr_w.to(args.device)
            
            minibatch_loss += unlabeled_loss(unlabeled_x, inputs, meta_train_outputs, eps, model)
            
#             minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
            
            train_loss += (minibatch_loss.detach().cpu()*inputs.shape[0])
            minibatch_loss.backward()
            opt.step()
    
            # keep track of epoch loss/accuracy
            
#             pred_labels = (F.sigmoid(outputs) > 0.5).int()
            
#             pred_labels = (outputs > 0.5).int()
#             
#             train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
    
    
    
        print('epochs::', ep)
    
#         for k in range(10):
#              
#             train_loss, train_acc = 0, 0
#              
#             for iter, (inputs, labels, ids) in enumerate(train_loader):
#                 inputs, labels = inputs.to(device=args.device, non_blocking=True),\
#                                     labels.to(device=args.device, non_blocking=True)
#                  
#             
#                 curr_w = w_array[ids].view(-1)
#                 
#                 if args.GPU:
#                     curr_w = curr_w.to(args.device)
#                  
#                 opt.zero_grad()
#                 outputs = model(inputs)
#     #             criterion.reduction = 'none'
#                 minibatch_loss = criterion(outputs, labels.type_as(outputs))
#                 minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
#                 minibatch_loss.backward()
#                 opt.step()
#                 train_loss += minibatch_loss.detach().cpu().item()*torch.sum(curr_w.detach().cpu())
#                  
#             print('full train loss::', k, train_loss/torch.sum(w_array))
    
        # inference after epoch
        with torch.no_grad():
            train_loss = train_loss/training_data_count#, train_acc/len(train_dataset)      
            valid_loss, (valid_acc, valid_per_class_acc) = get_loss_n_accuracy(model, valid_loader, args, args.num_class)
            test_loss, (test_acc, test_per_class_acc) = get_loss_n_accuracy(model, test_loader, args, args.num_class)                                  
            # log/print data
#             writer.add_scalar('Test/Loss', test_loss, ep)
#             writer.add_scalar('Test/Accuracy', test_acc, ep)
#             writer.add_scalar('Training/Loss', train_loss, ep)
#             writer.add_scalar('Training/Accuracy', train_acc, ep)
            print(f'|Train/Valid/Test Loss: {train_loss:.4f} / {valid_loss:.4f} / {test_loss:.4f}|', end='--')
            print(f'|valid acc/Test Acc: {valid_acc: 4f} / {test_acc:.4f}|', end='\r')
    
        sorted_w_array, sorted_ids = torch.sort(w_array.view(-1), descending=True)
        
        print('sorted array::')
        
        print(sorted_w_array[0:removed_count])
        
        print(sorted_w_array[unlabeled_training_data_count - removed_count:])
        
        print('sorted array ids::')
        
        print(sorted_ids[unlabeled_training_data_count - removed_count:])
        
        full_output_dir = os.path.join(args.output_dir, args.dataset)
        
#         torch.save(sorted_ids[unlabeled_training_data_count - removed_count:], full_output_dir + '/removed_ids_' + str(ep))
        torch.save(sorted_ids[unlabeled_training_data_count - removed_count:], full_output_dir + '/removed_ids_' + str(pid) + '_' + str(ep))
        
#         print(sorted_ids[0:train_dataset.data.shape[0] - removed_count])
        print('learning rate::', lr)
        
        print('r_sum_gap::', torch.sum(w_array) - (unlabeled_training_data_count - removed_count))    
        
        
        print('mu::', torch.max(mu_array), torch.min(mu_array))
        
        print('beta::', torch.max(beta_array), torch.min(beta_array))
        
        print('w::', max_w, min_w)
        
        print('gamma::', gamma)    
        
        print('here')
        
        if upper_lower_bound_satisfied and torch.abs(torch.sum(w_array) - (unlabeled_training_data_count - removed_count)) < final_sum_upper_bound:
            break

    
#     end_time.record()
#     end_time= time.time()
# #     torch.cuda.synchronize()
# #     time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
#     time_elapsed_secs = end_time - start_time
#     time_elapsed_mins = time_elapsed_secs/60
#     print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')
