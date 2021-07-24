'''
Created on Dec 5, 2020

'''
import glob
import os
import subprocess
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import tensorflow as tf
from sklearn.model_selection import train_test_split

# from snorkel.classification.data import DictDataset, DictDataLoader

import sys, os
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/pytorch_influence_functions')

from sklearn.metrics import roc_auc_score
from sklearn import metrics

try:
    from models.util_func import *
    from utils.utils import *
    # from train import *
    # import pytorch_influence_functions as ptif
    
    
except ImportError:
    from util_func import *
    from utils import *
    # from train import *
    # import pytorch_influence_functions as ptif
    


# import nvidia_smi

import models

noisy_weight = 1

transductive_coeff = 0.01

def valid_model(model, valid_DL, loss_func, tag, is_GPU, device, f1 = False):
    
    avg_loss = 0
    
    pred_acc = 0
    
    y_out = []
    
    y_true = []
    
    y_pred = []
    
    data_val_size = valid_DL.dataset.lenth
    
    with torch.no_grad():
        for X, Y, ids in valid_DL:
            
#             if (len(Y.shape) == 2 and Y.shape[1] > 1):
#                 Y = torch.argmax(Y, 1)
            
            X = X.type(torch.DoubleTensor)
            
#             if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                 if args.binary:
#                     Y = Y[:,args.did].type(torch.LongTensor)
#                 else:
#                     Y = Y.type(torch.LongTensor)
#             
#             else:
#                 if args.binary:
#                     Y = Y[:,args.did].type(torch.DoubleTensor)
#                 else:
#                     Y = Y.type(torch.DoubleTensor)
            
            
            if is_GPU:
                X = X.to(device)
                Y = Y.to(device)
            
            output = model(X)
            
            if f1:
                curr_loss_func = model.get_loss_function('mean', f1 = f1)
                
#                 model_out = model.forward(X)
                model_out = model.determine_labels(X, soft = True)
                
                curr_loss = curr_loss_func(model_out, Y.type(torch.long), num_class = model_out.shape[1])
            
            else:
                if loss_func is None:
                    curr_loss_func = model.get_loss_function('mean')
                    
    #                 if model.binary:
    #                     curr_loss = curr_loss_func(model.forward(X), Y.view(Y.shape[0], 1).type(X.dtype))
    #                 else:
                    curr_loss = curr_loss_func(model.forward(X), Y)
                    
    #                 curr_loss = curr_loss_func(model.forward(X), Y.type(torch.long))
                else:
                    curr_loss_func = loss_func
                    curr_loss = curr_loss_func(model.forward(X).view(Y.shape), Y)
            
            pred = model.determine_labels(X)
            
                
            y_out.append(output.cpu())
            
            pred_acc += torch.sum(pred.cpu() == Y.cpu())
    
            y_true.append(Y.cpu())
            
            y_pred.append(pred.cpu())
    
    
#             curr_loss = args.loss(output.view(-1), Y)
#             print(Y)
#             print(curr_loss)
            avg_loss += curr_loss.cpu().detach()*X.shape[0]
            
    avg_loss = avg_loss/data_val_size
    
    pred_acc = pred_acc*1.0/data_val_size
    
    
    y_pred_array = torch.cat(y_pred).numpy()
    
    y_true_array = torch.cat(y_true)
    
    y_out_array = torch.cat(y_out)
    
#     y_true_array[-1] = 1

#     fpr, tpr, thresholds = metrics.roc_curve(
#                 y_true_array, y_pred_array, pos_label=1)
#     
#     auc = metrics.auc(fpr, tpr)
    
#     print('auc score::', roc_auc_score(y_true_array, y_pred_array), auc)
#     if not (isinstance(curr_loss_func, nn.NLLLoss) or isinstance(curr_loss_func, nn.CrossEntropyLoss)):
#     if model.binary and y_true_array.shape[0] > 1:
    try:
        print(tag + ' auc score::', roc_auc_score(onehot(y_true_array, output.shape[1]).numpy(), F.softmax(y_out_array, 1).numpy()))
    except ValueError:
        print(tag + ' auc score::', 0)
    
    print(tag + ' dataset loss and accuracy::', avg_loss, pred_acc)


def valid_model_dataset(model, valid_dataset, loss_func, bz, tag, is_GPU, device, f1=False):
    
    avg_loss = 0
    
    pred_acc = 0
    
    y_out = []
    
    y_true = []
    
    y_pred = []
    
    data_val_size = valid_dataset.data.shape[0]
    
    if f1:
        # bz = data_val_size
        output_label_list = []
        exp_label_list = []
    
    model = model.to(device)
    
    with torch.no_grad():
#         for X, Y, ids in valid_DL:
        for k in range(0, data_val_size, bz):
            end_id = k + bz
            
            if end_id > data_val_size:
                end_id = data_val_size
            
            
#             curr_rand_ids = random_ids[k: end_id]
            
            X = valid_dataset.data[k: end_id]
            
            if isinstance(model, models.Logistic_regression) or isinstance(model, models.Binary_Logistic_regression):
                X = X.view(X.shape[0], -1)
            
            Y = valid_dataset.labels[k: end_id]
            
#             if (len(Y.shape) == 2 and Y.shape[1] > 1):
#                 Y = torch.argmax(Y, 1)
            
            X = X.type(torch.DoubleTensor)
            
#             if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                 if args.binary:
#                     Y = Y[:,args.did].type(torch.LongTensor)
#                 else:
#                     Y = Y.type(torch.LongTensor)
#             
#             else:
#                 if args.binary:
#                     Y = Y[:,args.did].type(torch.DoubleTensor)
#                 else:
#                     Y = Y.type(torch.DoubleTensor)
            
            
            if is_GPU:
                X = X.to(device)
                Y = Y.to(device)
            
            output = model(X)
            
#             if loss_func is None:
#                 curr_loss_func = model.get_loss_function('mean')
#                 
# #                 if model.binary:
# #                     curr_loss = curr_loss_func(model.forward(X), Y.view(Y.shape[0], 1).type(X.dtype))
# #                 else:
#                 curr_loss = curr_loss_func(model.forward(X), Y)
#                 
# #                 curr_loss = curr_loss_func(model.forward(X), Y.type(torch.long))
#             else:
#                 curr_loss_func = loss_func
#                 curr_loss = curr_loss_func(model.forward(X).view(Y.shape), Y)
            if loss_func is None:
                        
                if not f1:
                    curr_loss_func = model.get_loss_function('mean')
                    if not isinstance(curr_loss_func,nn.BCELoss):
                        curr_loss = curr_loss_func(model.forward(X), Y.type(torch.long)).view(-1)
                    else:
                        curr_loss = curr_loss_func(model.forward(X), Y.type(torch.double)).view(-1)
                else:
                    curr_loss_func = model.get_loss_function(f1 = f1)
                    
#                     model_out = F.softmax(model.forward(X), dim = 1)
                    model_out = model.determine_labels(X, soft = True)
                    
                    output_label_list.append(model_out.cpu())
                    
                    exp_label_list.append(Y.cpu())
                    
                    # curr_loss = curr_loss_func(model_out, Y, r_weight = None, is_training=True, num_class = model_out.shape[1])
            else:
                if not f1:
                
                    curr_loss_func = loss_func
                
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
    #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                    curr_loss = curr_loss_func(model.forward(X).view(Y.shape), Y).view(-1)
                
                else:
                    curr_loss_func = loss_func
                    
#                     model_out = F.softmax(model.forward(X), dim = 1)
                    model_out = model.determine_labels(X, soft = True)
                    
                    curr_loss = curr_loss_func(model_out, Y, r_weight = None, is_training=True, num_class = model_out.shape[1])

            
            pred = model.determine_labels(X)
            
            # print(output)
            y_out.append(output.cpu())
            
            pred_acc += torch.sum(pred.cpu() == Y.cpu())
    
            y_true.append(Y.cpu())
            
            y_pred.append(pred.cpu())
    
    
#             curr_loss = args.loss(output.view(-1), Y)
#             print(Y)
#             print(curr_loss)
            if not f1:
                avg_loss += curr_loss.cpu().detach()*X.shape[0]
    
    if f1:
        
        full_model_out = torch.cat(output_label_list, 0)
        
        full_exp_labels = torch.cat(exp_label_list, 0)
        
        avg_loss = curr_loss_func(full_model_out, full_exp_labels, r_weight = None, is_training=True, num_class = model_out.shape[1])
    
    else:
        
        avg_loss = avg_loss/data_val_size
    
    pred_acc = pred_acc*1.0/data_val_size
    
    
    y_pred_array = torch.cat(y_pred).numpy()
    
    y_true_array = torch.cat(y_true)
    
    y_out_array = torch.cat(y_out)
    
#     y_true_array[-1] = 1

#     fpr, tpr, thresholds = metrics.roc_curve(
#                 y_true_array, y_pred_array, pos_label=1)
#     
#     auc = metrics.auc(fpr, tpr)
    
#     print('auc score::', roc_auc_score(y_true_array, y_pred_array), auc)
#     if not (isinstance(curr_loss_func, nn.NLLLoss) or isinstance(curr_loss_func, nn.CrossEntropyLoss)):
#     if model.binary and y_true_array.shape[0] > 1:
    
#     try:
#         print(tag + ' auc score::', roc_auc_score(onehot(y_true_array, output.shape[1]).numpy(), F.softmax(y_out_array, 1).numpy()))
#     except ValueError:
#         print(tag + ' auc score::', 0)
#     
    print(tag + ' dataset loss and accuracy::', avg_loss, pred_acc)

    return avg_loss
# def valid_models(model, val_dataset, bz, tag, is_GPU, device, loss_func):
#     
#     data_val_size = val_dataset.data.shape[0]
#     
#     avg_loss = 0
#     
#     pred_acc = 0
#     
#     y_true = []
#     
#     y_pred = []
#     
#     y_out = []
#     
#     with torch.no_grad():
#         for k in range(0, data_val_size, bz):
#             end_id = k + bz
#             
#             if end_id > data_val_size:
#                 end_id = data_val_size
#             
#             
# #             curr_rand_ids = random_ids[k: end_id]
#             
#             X = val_dataset.data[k: end_id]
#             
# #             X = X.view(X.shape[0], -1)
#             
#             Y = val_dataset.labels[k: end_id]
#             
#             if (len(Y.shape) == 2 and Y.shape[1] > 1):
#                 Y = torch.argmax(Y, 1)
#             
#             X = X.type(torch.DoubleTensor)
#             
# #                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
# #                     if args.binary:
# #                         Y = Y[:,args.did].type(torch.LongTensor)
# #                     else:
# #                         Y = Y.type(torch.LongTensor)
# #                 
# #                 else:
# #                     if args.binary:
# #                         Y = Y[:,args.did].type(torch.DoubleTensor)
# #                     else:
# #                         Y = Y.type(torch.DoubleTensor)
#             
#             
#             if is_GPU:
#                 X = X.to(device)
#                 Y = Y.to(device)
#             
#             output = model.forward(X)
#             
#             pred = model.determine_labels(X)
# #             if args.binary:
#                 
#             y_out.append(output.cpu().view(-1))
#             
#             pred_acc += torch.sum(pred.cpu() == Y.cpu())
#     
#             y_true.append(Y.cpu())
#             
#             y_pred.append(pred.cpu())
#     
#             if loss_func is None:
#                 curr_loss_func = model.get_loss_function('mean')
#                 curr_loss = curr_loss_func(model.forward(X), Y.type(torch.long))
#             else:
#                 curr_loss_func = loss_func
#             
# #                 curr_loss2 = self.get_loss_function()
# #                 print(torch.max())
# #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
#                 curr_loss = curr_loss(model.forward(X).view(Y.shape), Y.type(torch.long))
#                 
#     
# #                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
# #                     curr_loss = args.loss(output, Y)
# #                 else:
# #                     curr_loss = args.loss(output.view(-1), Y)
#     
#     
# #             curr_loss = args.loss(output.view(-1), Y)
# #             print(Y)
# #             print(curr_loss)
#             avg_loss += curr_loss.cpu().detach()*(end_id - k)
#             
#     avg_loss = avg_loss/data_val_size
#     
#     pred_acc = pred_acc*1.0/data_val_size
#     
#     
#     y_pred_array = torch.cat(y_pred).numpy()
#     
#     y_true_array = torch.cat(y_true).numpy()
#     
# #         y_out_array = torch.cat(y_out).numpy()
# #     
# #         fpr, tpr, thresholds = metrics.roc_curve(
# #                     y_true_array, y_pred_array, pos_label=1)
# #         
# #         auc = metrics.auc(fpr, tpr)
#     
# #     print('auc score::', roc_auc_score(y_true_array, y_pred_array), auc)
# #         if not (isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss)):
# #             print(tag + ' auc score::', roc_auc_score(y_true_array, y_out_array), auc)
#     
#     print(tag + ' dataset loss and accuracy::', avg_loss, pred_acc)
#     
#     return avg_loss

# def get_model_para_shape_list(para_list):
    #
    # shape_list = []
    #
    # full_shape_list = []
    #
    # total_shape_size = 0
    #
    # for para in list(para_list):
    #
        # all_shape_size = 1
        #
        #
        # for i in range(len(para.shape)):
            # all_shape_size *= para.shape[i]
            #
        # total_shape_size += all_shape_size
        # shape_list.append(all_shape_size)
        # full_shape_list.append(para.shape)
        #
    # return full_shape_list, shape_list, total_shape_size

def get_model_para_shape_list(para_list):
    
    shape_list = []
    
    full_shape_list = []
    
    total_shape_size = 0
    
    for para in list(para_list):
        
        all_shape_size = 1
        
        
        for i in range(len(para.shape)):
            all_shape_size *= para.shape[i]
        
        total_shape_size += all_shape_size
        shape_list.append(all_shape_size)
        full_shape_list.append(para.shape)
        
    return full_shape_list, shape_list, total_shape_size

def get_model_para_list(model):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        res_list.append(param.data.cpu().clone())
        
        i += 1
        
    return res_list

def get_vectorized_grads(model, device = None):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        if device is None:
            res_list.append(param.grad.data.to('cpu').view(-1).clone())
        else:
            res_list.append(param.grad.data.to(device).view(-1).clone())
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping=True, test_performance = True, measure_GPU_utils = False, r_weight = None, exp_w_list = None, exp_grad_list = None):
#         origin_X, origin_Y = train_dataset.data, train_dataset.labels
        # if_dnn = True
        
        if not ((type(model) is models.Logistic_regression) or (type(model) is models.Binary_Logistic_regression)):
            # if_dnn = False
            
            capture_prov = False
            is_early_stopping = True
            test_performance = True
            # capture_prov = True
            # is_early_stopping = False
        
        if args.no_prov:
            capture_prov = False  
            is_early_stopping = True
            test_performance = True
        
        
        print('capture prov::', capture_prov)
        epoch = 0
        early_stopping = models.EarlyStopping(patience=epochs, verbose=True)
        
        mini_batch_epoch = 0
        
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
        # if is_GPU and measure_GPU_utils:
            # nvidia_smi.nvmlInit()
            # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(args.GPUID)
            # GPU_mem_usage_list = []
# #             GPU_utilization_list = []
    # #     if is_GPU:
    # #         lr.theta = lr.theta.to(device)
        # if is_GPU and measure_GPU_utils:
            # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # init_memory = res.used
            #
            # print('gpu utilization::', res.used)
        
        theta_list = []
        
        grad_list = []
        
        w_list = []
        
        grad_list = []
        
        if random_ids_multi_super_iterations is None:
            curr_random_ids_multi_super_iterations = []
        # else:
            # epochs = len(random_ids_multi_super_iterations)
        
        for j in range(epochs):
            
            print('epoch::', j)
            
            end = False
            
            accuracy = 0
            
            if not random_ids_multi_super_iterations is None:
                random_ids = random_ids_multi_super_iterations[j]
            else:
                random_ids = torch.randperm(origin_X.shape[0])
                curr_random_ids_multi_super_iterations.append(random_ids)
    #         learning_rate = lrs[j]
            
            
#             X = origin_X[random_ids]
#             
#             Y = origin_Y[random_ids]
            
            
            avg_loss = 0
            
            for i in range(0,origin_X.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= origin_X.shape[0]:
                    end_id = origin_X.shape[0]
        
                # if is_GPU and measure_GPU_utils:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    #
# #                     GPU_utilization_list.append(res.gpu)
#
                    # print('gpu utilization::', j,i, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                    
#                     print('gpu utilization::', j,i, res.gpu)
# #                     GPU_utilization_list.append(res.gpu)
#                     
#                     GPU_mem_usage_list.append(res.gpu)
                curr_rand_ids = random_ids[i:end_id]
                batch_x, batch_y = origin_X[curr_rand_ids], origin_Y[curr_rand_ids]
                
                batch_y = batch_y.type(torch.double)
                
                if r_weight is not None:
                    batch_r_weight = r_weight[curr_rand_ids]
                
                if isinstance(model, models.Logistic_regression) or isinstance(model, models.Binary_Logistic_regression):
                    
                    # print(batch_x.shape)
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    
                    batch_y = batch_y.to(device)
                    
                    if r_weight is not None:
                        batch_r_weight = batch_r_weight.to(device)
#                 if self.lr.theta.grad is not None:
#                     self.lr.theta.grad.zero_()
        
                optimizer.zero_grad()
                
                if f1:
                    curr_loss = model.get_loss_function('mean', f1 = f1)
                    
#                     model_out = model.forward(batch_x)
                    model_out = model.determine_labels(batch_x, soft = True)
                    
                    loss = curr_loss(model_out, batch_y.type(torch.long), num_class = model_out.shape[1])
                
                else:
                    if loss_func is None:
                        curr_loss = model.get_loss_function('none')
                        loss = curr_loss(model.forward(batch_x), batch_y.type(torch.long))
                        if r_weight is not None:
                            loss = torch.mean(loss.view(-1) * batch_r_weight.view(-1))
                        
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
    #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        if r_weight is None:
                            loss = curr_loss(model.forward(batch_x).view(batch_y.shape), batch_y)
                        
                        else:
                            
                            model_out = model.forward(batch_x)
                            
                            # print(batch_x.shape, model_out.shape, batch_y.shape)
                            
                            loss = curr_loss(model_out.view(batch_y.shape), batch_y, batch_r_weight)
                
                # print(batch_x, batch_y)
                
                loss.backward()
                
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                
                if exp_grad_list is not None:
                    curr_grad_list = get_devectorized_parameters(exp_grad_list[mini_batch_epoch], full_shape_list, shape_list)
                
                avg_loss += loss.cpu().detach()*(end_id - i)
                
                if capture_prov:
                    w_list.append(get_model_para_list(model))
                    curr_grad = get_vectorized_grads(model)
#                     print(torch.norm(curr_grad))
                    grad_list.append(curr_grad)
                else:
                    if i == 0 and j == 0:
                        w_list.append(get_model_para_list(model))
                        
                        curr_grad = get_vectorized_grads(model)
                        grad_list.append(curr_grad)
                
                optimizer.step()
                
                epoch = epoch + 1
                
                mini_batch_epoch += 1
            
                # if is_GPU and measure_GPU_utils:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    #
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # print('gpu utilization::', j,i, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                    
#                 del batch_x, batch_y
                
#             avg_loss = avg_loss
            
            print('loss::', j, avg_loss/(origin_X.shape[0]))
            
#             del X
#             
#             del Y
            
            if test_performance:
                valid_loss = valid_model_dataset(model, val_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
            
                if is_early_stopping:
                    early_stopping(valid_loss, model, j)
            
#             if val_dataset is not None:
# #                 model, valid_dataset, loss_func, bz, tag, is_GPU, device
#                 valid_model_dataset(model, val_dataset, loss_func, batch_size, 'valid', is_GPU, device)
    #         del random_ids
            if test_performance and test_dataset is not None:
                valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
            
            if test_performance and is_early_stopping:
                if early_stopping.early_stop:
                    models.set_model_parameters(model, early_stopping.model_param, device)
                    print("Early stopping")
    #                 valid_models(model, valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                    
                    valid_model_dataset(model, val_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
                    
                    if test_dataset is not None:
    #                     valid_models(model, test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                        valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
                    break
            # print('training with weight loss::', avg_loss/(origin_X.shape[0]))
            # valid_model_dataset(model, val_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
            # valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
        
        if not random_ids_multi_super_iterations is None:
            curr_random_ids_multi_super_iterations = random_ids_multi_super_iterations
        
        if test_performance and is_early_stopping:
            models.set_model_parameters(model, early_stopping.model_param, device)
            
            print("Early stopping")
    #                 valid_models(model, valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                    
            valid_model_dataset(model, val_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
            
            if test_dataset is not None:
#                     valid_models(model, test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
        # if is_GPU and measure_GPU_utils:
            # return w_list, grad_list,curr_random_ids_multi_super_iterations, GPU_mem_usage_list
#         valid_model_dataset(model, train_dataset, curr_loss, batch_size, 'train', is_GPU, device, f1 = f1)
        
        return w_list, grad_list,curr_random_ids_multi_super_iterations



def train_model_with_weight(model, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False, valid_dataset = None, test_dataset = None, f1 = False):
    
        epoch = 0
        
        early_stopping = models.EarlyStopping(patience=10, verbose=True)

        mini_batch_epoch = 0
        
    #     if is_GPU:
    #         lr.theta = lr.theta.to(device)
        
        
        theta_list = []
        
        grad_list = []
        
        for j in range(epochs):
            
#             print('epoch::', j)
            
            end = False
            
            accuracy = 0
            
            if not random_ids_multi_super_iterations is None:
                random_ids = random_ids_multi_super_iterations[j]
            else:
                random_ids = torch.randperm(origin_X.shape[0])
        
    #         learning_rate = lrs[j]
            
            X = origin_X[random_ids]
            
            Y = origin_Y[random_ids]
            
            
            curr_r_weights = r_weights[random_ids]
            
            avg_loss = 0
            
            for i in range(0,X.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= X.shape[0]:
                    end_id = X.shape[0]
        
                batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                
                batch_y = batch_y.type(torch.double)
                
#                 if model.__name__ == 'Logistic_regression' or model.__name__ == 'Binary_Logistic_regression': 
                if isinstance(model, models.Logistic_regression) or isinstance(model, models.Binary_Logistic_regression):
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    
                    batch_y = batch_y.to(device)
                    
                    batch_r_weight = batch_r_weight.to(device)
                
#                 if self.lr.theta.grad is not None:
#                     self.lr.theta.grad.zero_()
        
                optimizer.zero_grad()
                
                if loss_func is None:
                    
                    if not f1:
                        curr_loss = model.get_loss_function('none')
                        loss = torch.sum(curr_loss(model.forward(batch_x), batch_y.type(torch.long)).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    else:
                        curr_loss = model.get_loss_function(f1 = f1)
                        
                        model_out = model.determine_labels(batch_x, soft = True)
                        
                        loss = curr_loss(model_out, batch_y, r_weight = batch_r_weight.view(-1), is_training=True, num_class = model_out.shape[1])
                    
                    
                else:
                    if not f1:
                    
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
    #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = torch.sum(curr_loss(model.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    
                    else:
                        curr_loss = loss_func
                        
                        model_out = model.determine_labels(batch_x, soft = True)
                        
                        loss = curr_loss(model_out, batch_y, r_weight = batch_r_weight.view(-1), is_training=True, num_class = model_out.shape[1])
#                     if torch.isnan(loss):
#                         print('here')
                
#                     print(i, j, loss)
                
#                 if loss < 0:
#                     print('here')
                
#                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                
#                 print(torch.norm(gap)) 
                
#                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)

#                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
        
                
        
                loss.backward()
                
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                
                avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
                
                optimizer.step()
#                 with torch.no_grad():
    
    
                    
#                     self.lr.theta -= learning_rate * self.lr.theta.grad
                    
#                     gap = torch.norm(self.lr.theta.grad)
                    
                
                 
                
                epoch = epoch + 1
                
                mini_batch_epoch += 1
            
            
#             avg_loss = avg_loss
            
            print('loss::', j, avg_loss/(torch.sum(r_weights)))
            
            del X
            
            del Y
            
#                 if valid_dataset is not None:
#             valid_loss = valid_model(model, valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
            valid_loss = valid_model_dataset(model, valid_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
            
            early_stopping(valid_loss, model)
            
            
            if test_dataset is not None:
#                 valid_models(model, test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
                
            if early_stopping.early_stop:
                models.set_model_parameters(model, early_stopping.model_param, device)
                print("Early stopping")
#                 valid_models(model, valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                valid_model_dataset(model, valid_dataset, None, batch_size, 'valid', is_GPU, device, f1 = f1)
                
                if test_dataset is not None:
#                     valid_models(model, test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                    valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = f1)
                break
    #         del random_ids
            
        print('training with weight loss::', avg_loss/(torch.sum(r_weights)))

def get_mislabeled_ids(model, dataset, args):
    
    mislabeled_ids = []
    
    for i in range(0,dataset.data.shape[0], args.bz):
                
        end_id = i + args.bz
        
        if end_id >= dataset.data.shape[0]:
            end_id = dataset.data.shape[0]

        batch_x, batch_y = dataset.data[i:end_id], dataset.labels[i:end_id]
        
        if args.GPU:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
        
        batch_y = batch_y.type(torch.double)
        
        batch_y_pred = model.determine_labels(batch_x)
        
        mislabeled_ids.append(~(batch_y_pred.view(-1) == batch_y.view(-1)))
    
    mislabeled_ids_tensor = torch.nonzero(torch.cat(mislabeled_ids, dim = 0))
    
    return mislabeled_ids_tensor 
    
        

def train_model(model, optimizer, train_DL, valid_DL, epochs, is_GPU, device, loss_func = None, skipped_ids = None, f1 = False):
    
    model.train()
    
    epoch = 0
    
    
    
    
    for j in range(epochs):
        
        avg_loss = 0
        
#         for i in range(0,X.shape[0], batch_size):
        
        mini_batch_epoch = 0
        for batch_x, batch_y, ids in train_DL:
            
            batch_x = batch_x.type(torch.DoubleTensor)
            
            batch_x = batch_x[ids == ids]
            
            batch_y = batch_y[ids == ids]
            
#             if skipped_ids is not None:
# #                 compareview = t2.repeat(t1.shape[0],1).T
#                 
#                 idx = (ids == skipped_ids)
            
            if is_GPU:
                batch_x = batch_x.to(device)
                
                batch_y = batch_y.to(device)
                
            optimizer.zero_grad()
            
            if f1:
                curr_loss = model.get_loss_function('mean', f1 = f1)
                
#                 model_out = model.forward(batch_x)
                model_out = model.determine_labels(batch_x, soft = True)
                
                loss = curr_loss(model_out, batch_y.type(torch.long), num_class = model_out.shape[1])
            
            else:
                if loss_func is None:
                    curr_loss = model.get_loss_function('mean')
    #                 if model.binary:
    #                     loss = curr_loss(model.forward(batch_x), batch_y.view(batch_y.shape[0], 1).type(batch_x.dtype))
    #                 else:
                    loss = curr_loss(model.forward(batch_x), batch_y)
                else:
                    curr_loss = loss_func
                    loss = curr_loss(model.forward(batch_x).view(batch_y.shape), batch_y)
            
            loss.backward()
            
            avg_loss += loss.cpu().detach()*batch_x.shape[0]
            
            optimizer.step()
            
            epoch = epoch + 1
            
            print('curr loss::', mini_batch_epoch, loss.cpu().detach())
            
            mini_batch_epoch += 1
        
        
#             avg_loss = avg_loss
        
        print('train loss::', j, avg_loss/(train_DL.dataset.lenth))
        
#         if j%10 == 0:
        if valid_DL is not None:
            valid_model(model, valid_DL, loss_func, 'valid', is_GPU, device, f1 = f1)
        
        
        
    print('training with weight loss::', avg_loss/(train_DL.dataset.lenth))
    
    return model



def load_spam_dataset(load_train_labels: bool = True, split_dev_valid: bool = False):
#     if os.path.basename(os.getcwd()) == "snorkel-tutorials":
#         os.chdir("spam")
#     try:
#         subprocess.run(["bash", "download_data.sh"], check=True, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         print(e.stderr.decode())
#         raise e

#     git_ignore_dir = args.git_ignore_dir
    git_ignore_dir = get_default_git_ignore_dir()

    filenames = sorted(glob.glob(git_ignore_dir + "/youtube/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_dev = df_train.sample(100, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test




def df_to_features(vectorizer, df, split):
    """Convert pandas DataFrame containing spam data to bag-of-words PyTorch features."""
    words = [row.text for i, row in df.iterrows()]

    if split == "train":
        feats = vectorizer.fit_transform(words)
    else:
        feats = vectorizer.transform(words)
    X = feats.todense()
    Y = df["label"].values
    return X, Y


# def create_dict_dataloader(X, Y, split, **kwargs):
#     """Create a DictDataLoader for bag-of-words features."""
#     ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
#     return DictDataLoader(ds, **kwargs)


def get_pytorch_mlp(hidden_dim, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return nn.Sequential(*layers)


def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

def O2U_second_stage(model, optimizer1, criterion, epochs, train_dataset, validation_dataset, test_dataset, full_labeled_id_tensor, is_GPU, device, removed_count, batch_size=16, random_ids_multi_epochs = None, r_weight = None):
    
    noise_or_not = torch.ones(train_dataset.data.shape[0]).bool()
    
    noise_or_not[full_labeled_id_tensor] = False
    
    # if_dnn = True
    #
    # if not ((type(model) is models.Logistic_regression) or (type(model) is models.Binary_Logistic_regression)):
    #         # if_dnn = False
    #
    #     if_dnn = False
    
#     train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=16,
#                                                num_workers=32,
#                                                shuffle=True)
#     optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#     criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
    moving_loss_dic=torch.zeros_like(noise_or_not)
    ndata = train_dataset.lenth
    
    w_list = []
    
    grad_list = []
    
    
    if random_ids_multi_epochs is None:
        curr_random_ids_multi_epochs = []

    for epoch in range(1, epochs):
        # train models
        globals_loss=0
        model.train()
        with torch.no_grad():
            valid_model_dataset(model, validation_dataset, None, batch_size, 'validation', is_GPU, device)
            valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device)
            
#             evaluate_model_test_dataset(test_dataset, model, args, tag = 'validation')
#             accuracy=evaluate(test_loader, network)
        example_loss= torch.zeros_like(noise_or_not,dtype=float)

        t = (epoch % 10 + 1) / float(10)
        lr = (1 - t) * 0.01 + t * 0.001

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        
        if random_ids_multi_epochs is None:
            random_ids = torch.randperm(train_dataset.data.shape[0])
            curr_random_ids_multi_epochs.append(random_ids)
        else:
            random_ids = random_ids_multi_epochs[epoch-1]
        
        X = train_dataset.data[random_ids]
        
        Y = train_dataset.labels[random_ids]
        
#         for i, (images, labels, indexes) in enumerate(train_loader_detection):
        for i in range(0,X.shape[0], batch_size):
                
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]
    
            indexes = random_ids[i:end_id].view(-1)
            
            images, labels = X[indexes], Y[indexes]
            
            batch_r_weight = None
            
            if r_weight is not None:
                batch_r_weight = r_weight[indexes]
            
            if is_GPU:
                images = images.to(device)
                labels = labels.to(device)
                
                if r_weight is not None:
                    batch_r_weight = batch_r_weight.to(device)
            
#             batch_y = labels.type(torch.double)

            

#             images = Variable(images).cuda()
#             labels = Variable(labels).cuda()

            curr_model_para = get_model_para_list(model)
            # w_list.append(curr_model_para)
            logits = model(images)
            loss_1 =criterion(logits,labels,batch_r_weight)
            example_loss[indexes] = loss_1.cpu()
#             for pi, cl in zip(indexes, loss_1):
#                 example_loss[pi] = cl.cpu().data.item()

            globals_loss += loss_1.sum().cpu().data.item()

            loss_1 = loss_1.mean()
            optimizer1.zero_grad()
            loss_1.backward()
            
            curr_grad = get_all_vectorized_parameters1(model.parameters())
            # grad_list.append(curr_grad)
            
            
            optimizer1.step()
        example_loss=example_loss - example_loss.mean()
        moving_loss_dic=moving_loss_dic+example_loss

    ordered_list, sorted_ids = torch.sort(moving_loss_dic.detach(), descending = True)
#     loss_1_sorted = moving_loss_dic[ind_1_sorted]
# 
# #         remember_rate = 1 - forget_rate
# #         num_remember = int(remember_rate * len(loss_1_sorted))
#     num_remember = X.shape[0] - removed_count
# 
# 
#     noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)
#     mask = np.ones_like(noise_or_not,dtype=np.float32)
#     mask[ind_1_sorted[num_remember:]]=0
# 
#     top_accuracy_rm=int(0.9 * len(loss_1_sorted))
#     top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

#         print(("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy))


    if random_ids_multi_epochs is None:
        random_ids_multi_epochs = curr_random_ids_multi_epochs
    return moving_loss_dic, ordered_list, sorted_ids, random_ids_multi_epochs, w_list, grad_list



def O2U_second_stage_2(model, optimizer1, criterion, epochs, train_dataset, validation_dataset, test_dataset, full_labeled_id_tensor, is_GPU, device, removed_count, batch_size=16, random_ids_multi_epochs = None):
    
    noise_or_not = torch.ones(train_dataset.data.shape[0]).bool()
    
    noise_or_not[full_labeled_id_tensor] = False
    
#     train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=16,
#                                                num_workers=32,
#                                                shuffle=True)
#     optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#     criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
    moving_loss_dic=torch.zeros_like(noise_or_not)
    ndata = train_dataset.lenth
    
    w_list = []
    
    grad_list = []
    
    
    if random_ids_multi_epochs is None:
        curr_random_ids_multi_epochs = []

    for epoch in range(1, epochs):
        # train models
        globals_loss=0
        model.train()
        with torch.no_grad():
            valid_model_dataset(model, validation_dataset, None, batch_size, 'validation', is_GPU, device)
            valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device)
            
#             evaluate_model_test_dataset(test_dataset, model, args, tag = 'validation')
#             accuracy=evaluate(test_loader, network)
        example_loss= torch.zeros_like(noise_or_not,dtype=float)

        t = (epoch % 10 + 1) / float(10)
        lr = (1 - t) * 0.01 + t * 0.001

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        
        if random_ids_multi_epochs is None:
            random_ids = torch.randperm(train_dataset.data.shape[0])
            curr_random_ids_multi_epochs.append(random_ids)
        else:
            random_ids = random_ids_multi_epochs[epoch-1]
        
        X = train_dataset.data[random_ids]
        
        Y = train_dataset.labels[random_ids]
        
#         for i, (images, labels, indexes) in enumerate(train_loader_detection):
        for i in range(0,X.shape[0], batch_size):
                
            end_id = i + batch_size
            
            if end_id >= X.shape[0]:
                end_id = X.shape[0]
    
            images, labels = X[i:end_id], Y[i:end_id]
            
            indexes = random_ids[i:end_id].view(-1)
            
            if is_GPU:
                images = images.to(device)
                labels = labels.to(device)
            
#             batch_y = labels.type(torch.double)

            

#             images = Variable(images).cuda()
#             labels = Variable(labels).cuda()

            curr_model_para = get_model_para_list(model)
            w_list.append(curr_model_para)
            logits = model(images)
            loss_1 =criterion(logits,labels)
#             example_loss[indexes] = loss_1
#             for pi, cl in zip(indexes, loss_1):
#                 example_loss[pi] = cl.cpu().data.item()

#             globals_loss += loss_1.sum().cpu().data.item()

#             loss_1 = loss_1.mean()
            optimizer1.zero_grad()
            loss_1.backward()
            
            curr_grad = get_all_vectorized_parameters1(model.parameters())
            grad_list.append(curr_grad)
            
            
            optimizer1.step()
#         example_loss=example_loss - example_loss.mean()
#         moving_loss_dic=moving_loss_dic+example_loss
# 
#     ordered_list, sorted_ids = torch.sort(moving_loss_dic.detach(), descending = True)
#     loss_1_sorted = moving_loss_dic[ind_1_sorted]
# 
# #         remember_rate = 1 - forget_rate
# #         num_remember = int(remember_rate * len(loss_1_sorted))
#     num_remember = X.shape[0] - removed_count
# 
# 
#     noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)
#     mask = np.ones_like(noise_or_not,dtype=np.float32)
#     mask[ind_1_sorted[num_remember:]]=0
# 
#     top_accuracy_rm=int(0.9 * len(loss_1_sorted))
#     top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

#         print(("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy))


#     if random_ids_multi_epochs is None:
#         random_ids_multi_epochs = curr_random_ids_multi_epochs
#     return moving_loss_dic, ordered_list, sorted_ids, random_ids_multi_epochs, w_list, grad_list


def do_training(args, train_dataset):
#     model = models.Logistic_regression(train_dataset.data.shape[1], 2, bias = False)
    model = models.Binary_Logistic_regression(train_dataset.data.shape[1], bias = True)

#     criterion = 
    
    criterion2 = model.get_loss_function()
    
    
    Y_one_list = (train_dataset.labels > 0.5).type(torch.DoubleTensor)
    
    loss_val2 = criterion2(model(train_dataset.data).view(Y_one_list.shape), Y_one_list)
    
    loss_val1 = torch.mean(model.soft_loss_function(model(train_dataset.data), Y_one_list.type(torch.DoubleTensor)))
    
    print(loss_val1, loss_val2)
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
        
#     criterion, optimizer = hyper_param_func(model.parameters(), args.lr)
    
    args.loss = model.soft_loss_function
    
    args.optim = optimizer
#     model.init_params()
    model = model.to(args.device)
    
    removed_count = 60
    
    model.train_model_with_weight(optimizer, None, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, 0, torch.ones_like(train_dataset.labels), removed_count, None, loss_func=args.loss, integer_label = True)
    
    return model


def do_training_general(args, train_dataset, model_name, num_class=1, r_weight = None, soft = True, valid_dataset = None, test_dataset = None):
#     model = models.Logistic_regression(train_dataset.data.shape[1], 2, bias = False)

#     
    size = 1
        
    for k in range(len(train_dataset.data.shape)-1):
        size *= train_dataset.data.shape[k+1]
    
    if model_name == 'Binary_Logistic_regression':
            
        model = models.Binary_Logistic_regression(size, bias = True)
#         model2 = models.Binary_Logistic_regression(size, bias = True)
    else:
        if model_name == 'Logistic_regression':
            model = models.Logistic_regression(size, num_class, bias = True)
#             model2 = models.Logistic_regression(size, num_class, bias = True)
        else:
            
            model_class = getattr(models, model_name)
            model = model_class(num_class)
#             model2 = model_class(num_class)
#             if model_name == 'LeNet5':
#                 model = models.LeNet5()


#     criterion = 
    
#     criterion2 = model.get_loss_function()
#     
#     
#     Y_one_list = (train_dataset.labels > 0.5).type(torch.DoubleTensor)
#     
#     loss_val2 = criterion2(model(train_dataset.data).view(Y_one_list.shape), Y_one_list)
#     
#     loss_val1 = torch.mean(model.soft_loss_function(model(train_dataset.data), Y_one_list.type(torch.DoubleTensor)))
#     
#     print(loss_val1, loss_val2)
    
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
#     optimizer2 = model2.get_optimizer(args.tlr, args.wd)
        
#     criterion, optimizer = hyper_param_func(model.parameters(), args.lr)
    if args.f1:
        args.loss = model.get_loss_function(f1 = args.f1)
    else:
        if soft:
            args.loss = model.soft_loss_function
        else:
            args.loss = None#model.get_loss_function(reduction = 'none')
    
    args.optim = optimizer
#     model.init_params()
    model = model.to(args.device)
#     model2 = model2.to(args.device)
#     set_model_parameters(model2, models.get_model_para_list(model))
    
    removed_count = 60
    
    if r_weight is None:
        r_weight = torch.ones(train_dataset.data.shape[0], dtype = train_dataset.data.dtype, device = train_dataset.data.device) 
    
    
#     random_ids_multi_epochs = []
#     for i in range(args.epochs):
#         random_ids_multi_epochs.append(torch.tensor(list(range(train_dataset.data.shape[0]))))
    
    
#     model.train_model_with_weight(optimizer, random_ids_multi_epochs, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, 0, r_weight, removed_count, None, loss_func=args.loss, integer_label = True, valid_dataset =valid_dataset, test_dataset = test_dataset)
    
    train_model_with_weight(model, optimizer, None, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, 0, r_weight, removed_count, None, loss_func=args.loss, integer_label = True, valid_dataset =valid_dataset, test_dataset = test_dataset, f1 = args.f1)
    
    return model

def do_training_general_hard(args, train_dataset, valid_dataset, test_dataset, model_name, num_class=1, binary = False):
#     model = models.Logistic_regression(train_dataset.data.shape[1], 2, bias = False)

#     
    size = 1
        
    for k in range(len(train_dataset.data.shape)-1):
        size *= train_dataset.data.shape[k+1]
    
    if model_name == 'Binary_Logistic_regression':
            
        model = models.Binary_Logistic_regression(size, bias = True)
    else:
        if model_name == 'Logistic_regression':
            model = models.Logistic_regression(size, num_class, bias = True)
            
        else:
            
            model_class = getattr(models, model_name)
            model = model_class(num_class, binary=binary)
#             if model_name == 'LeNet5':
#                 model = models.LeNet5()


#     criterion = 
    
#     criterion2 = model.get_loss_function()
#     
#     
#     Y_one_list = (train_dataset.labels > 0.5).type(torch.DoubleTensor)
#     
#     loss_val2 = criterion2(model(train_dataset.data).view(Y_one_list.shape), Y_one_list)
#     
#     loss_val1 = torch.mean(model.soft_loss_function(model(train_dataset.data), Y_one_list.type(torch.DoubleTensor)))
#     
#     print(loss_val1, loss_val2)
    
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
        
#     criterion, optimizer = hyper_param_func(model.parameters(), args.lr)
#     if soft:
#         args.loss = model.soft_loss_function
#     else:
    args.loss = None#model.get_loss_function(reduction = 'none')
    
    args.optim = optimizer
#     model.init_params()
    model = model.to(args.device)
    
    
#     if r_weight is None:
#         r_weight = torch.ones(train_dataset.data.shape[0], dtype = train_dataset.data.dtype, device = train_dataset.data.device) 
    #optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None
    train_model_dataset(model, optimizer, None, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, loss_func=args.loss, val_dataset = valid_dataset, test_dataset = test_dataset, f1 = args.f1)
    
    return model


def do_training_general_hard_DL(args, train_DL, valid_DL, model_name, num_class=1, binary = False, skipped_ids = None):
#     model = models.Logistic_regression(train_dataset.data.shape[1], 2, bias = False)

#     
    
    if model_name == 'Binary_Logistic_regression':
        
        size = 1
        
        for k in range(len(train_DL.dataset.data.shape)-1):
            size *= train_DL.dataset.data.shape[k+1]
        
        model = models.Binary_Logistic_regression(size, bias = True)
    else:
        if model_name == 'Logistic_regression':
            size = 1
        
            for k in range(len(train_DL.dataset.data.shape)-1):
                size *= train_DL.dataset.data.shape[k+1]
            model = models.Logistic_regression(size, num_class, bias = True)
            
        else:
            
            model_class = getattr(models, model_name)
            model = model_class(num_class, binary = binary)
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
        
    args.loss = None#model.get_loss_function(reduction = 'none')
    
    args.optim = optimizer
#     model.init_params()
    model = model.to(args.device)
    
    
#     if r_weight is None:
#         r_weight = torch.ones(train_dataset.data.shape[0], dtype = train_dataset.data.dtype, device = train_dataset.data.device) 
    #optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None
#     model.train_model(optimizer, None, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, loss_func=args.loss)
    
    
    model = train_model(model, optimizer, train_DL, valid_DL, args.epochs, args.GPU, args.device, loss_func = args.loss, skipped_ids = skipped_ids, f1 = args.f1)
    
    return model

def evaluate_model_test_dataset(test_dataset, model, args, tag = 'test'):
    
    args.loss = model.get_loss_function()
    
    if args.model == 'Logistic_regression' or args.model == 'Binary_Logistic_regression':
        test_dataset_copy = models.MyDataset(test_dataset.data.view(test_dataset.data.shape[0], -1), test_dataset.labels)
    
#         valid_models(test_dataset_copy, model, args, tag)
        valid_model_dataset(model, test_dataset_copy, None, args.bz, 'test', args.GPU, args.device, f1 = args.f1)
        if not args.f1:
            valid_model_dataset(model, test_dataset_copy, None, args.bz, 'test', args.GPU, args.device, f1 = True)
    else:
#         valid_models(test_dataset, model, args, tag)
        valid_model_dataset(model, test_dataset, None, args.bz, 'test', args.GPU, args.device, f1 = args.f1)
        if not args.f1:
            valid_model_dataset(model, test_dataset_copy, None, args.bz, 'test', args.GPU, args.device, f1 = True)

def get_most_influential_point(influences, test_ids, removed_count = 100):
    
#     influences_tensor = torch.cat(influences, dim = 0)
    influence_list = []
     
#     curr_train_ids = [] 
    
    for i in range(len(test_ids)):
        
        test_id = test_ids[i]
        
        curr_influence = influences[test_id]
         
#         influence_list.append(curr_influence['influence'][0])
        
        influence_list.append(curr_influence)
         
#         print('test id::', test_id, curr_influence)
        
    ordered_list, sorted_ids = torch.sort(torch.tensor(influence_list), descending = True)
    
#     ordered_list = torch(np.array(influence_list))
#      
#     sorted_ids  = torch.from_numpy(np.argsort(np.array(influence_list)))
    print('sorted list result::')
#     
    np.set_printoptions(threshold=sys.maxsize)
     
#     print(ordered_list)
     
    np.set_printoptions(threshold=sys.maxsize)
    
    test_id_tensor = torch.tensor(test_ids)
#     print(sorted_ids)
     
    return test_ids[sorted_ids[0]], ordered_list, test_id_tensor[sorted_ids]




'''-torch.mean(torch.sum(torch.nn.functional.log_softmax(y, dim=1) * onehot(t, 2), dim=1))'''
# def evaluate_influence_function_repetitive(args, training_dataset, Y_train_full, test_dataset, model, count, loss_func):
#
#     train_ids = list(range(training_dataset.data.shape[0]))
#
#     origin_training_dataset = models.MyDataset(training_dataset.data.clone(), training_dataset.labels.clone())    
#
#
# #     model = do_training(args, training_dataset)
# #     
# #     evaluate_model_test_dataset(test_dataset, model, args)
#
#
#     all_influence_points = []
#
#     for i in range(count):
#
#
#         ptif.init_logging()
#         config = ptif.get_default_config()
#         train_DL = DataLoader(training_dataset, batch_size=args.bz)
#
#         test_DL = DataLoader(test_dataset, batch_size=args.bz)
#
#
#
#         config['test_sample_num'] = 0
#
#         config['outdir'] = args.output_dir
#
#         if args.GPU:
#             config['gpu'] = args.GPUID
#         else:
#             config['gpu'] = -1
#
#         config['single_train'] = False
#
#         config['loss_func'] = loss_func
#
# #         influences = ptif.calc_img_wise_with_ids(config, model, train_DL, test_DL, test_ids)
#
#
#         influences = ptif.calc_img_wise_multi_with_ids(config, model, train_DL, test_DL, train_ids)
#
#         print('influences::')
#
#
#
#         most_influence_point, ordered_list, sorted_train_ids = get_most_influential_point(influences, train_ids)
#
# #         torch.save(sorted_train_ids, )
#         print(sorted_train_ids[0:count])
#
#         all_influence_points.append(most_influence_point)
#
#         print('most influence point::', most_influence_point)
#
#         print(all_influence_points)
#
#         train_ids = [id for id in train_ids if not id == most_influence_point]
#
#
# #         training_dataset.labels[most_influence_point] = Y_train_full[most_influence_point]
#
#         update_training_data = origin_training_dataset.data[torch.tensor(train_ids)]
#
#         update_training_labels = origin_training_dataset.labels[torch.tensor(train_ids)]
#
#         training_dataset.data = update_training_data
#
#         training_dataset.labels = update_training_labels
#
# #         model = do_training(args, training_dataset)
#
#     print(all_influence_points)
#
#     influential_training_data = origin_training_dataset.data[torch.tensor(all_influence_points)]
#
#     influential_training_labels = Y_train_full[torch.tensor(all_influence_points)]
#
#
#     training_dataset.data = torch.cat([training_dataset.data, influential_training_data], dim = 0)
#
#     training_dataset.labels = torch.cat([training_dataset.labels, influential_training_labels], dim = 0)
#
#     print('after cleaning by influence function::')
#
#     model = do_training(args, training_dataset)
#
#     evaluate_model_test_dataset(test_dataset, model, args)


# def evaluate_influence_transductive_repetitve(args, labeled_train_dataset, unlabeled_train_dataset, valid_dataset, dataset_test, model, count, loss_func, model_name, num_class):
#
#     train_ids = list(range(unlabeled_train_dataset.data.shape[0]))
#
#     ptif.init_logging()
#     config = ptif.get_default_config()
#     train_DL = DataLoader(labeled_train_dataset, batch_size=args.bz)
#
#     valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
#
#
#
#     config['test_sample_num'] = 0
#
#     config['outdir'] = args.output_dir
#
#     if args.GPU:
#         config['gpu'] = args.GPUID
#     else:
#         config['gpu'] = -1
#
#     config['single_train'] = False
#
#     config['loss_func'] = loss_func
#
#     influences = ptif.calc_img_wise_multi_with_ids_transductive(config, model, train_DL, valid_DL, train_ids, unlabeled_train_dataset)
#
#     most_influence_point, ordered_list, sorted_train_ids = get_most_influential_point(influences, train_ids)
#
#     return most_influence_point, ordered_list, sorted_train_ids


# def evaluate_influence_function_repetitive2(args, training_dataset, Y_train_full, valid_dataset, dataset_test, model, count, loss_func, model_name, num_class):
#
#     train_ids = list(range(training_dataset.data.shape[0]))
#
#     origin_training_dataset = models.MyDataset(training_dataset.data.clone(), training_dataset.labels.clone())    
#
#
# #     model = do_training(args, training_dataset)
# #     
# #     evaluate_model_test_dataset(test_dataset, model, args)
#
#
#     all_influence_points = []
#
# #     for i in range(count):
#
#
#     ptif.init_logging()
#     config = ptif.get_default_config()
#     train_DL = DataLoader(training_dataset, batch_size=args.bz)
#
#     valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
#
#
#
#     config['test_sample_num'] = 0
#
#     config['outdir'] = args.output_dir
#
#     if args.GPU:
#         config['gpu'] = args.device
#     else:
#         config['gpu'] = None
#
#     config['single_train'] = False
#
#     config['loss_func'] = loss_func
#
# #         influences = ptif.calc_img_wise_with_ids(config, model, train_DL, test_DL, test_ids)
#
#
#     influences = ptif.calc_img_wise_multi_with_ids(config, model, train_DL, valid_DL, train_ids, f1 = args.f1)
#
#     print('influences::')
#
#
#
#     most_influence_point, ordered_list, sorted_train_ids = get_most_influential_point(influences, train_ids)
#
#     sorted_train_ids = sorted_train_ids.cpu()
#
# #         torch.save(sorted_train_ids, )
# #     print(sorted_train_ids[0:count])
#
#     all_influence_points.append(most_influence_point)
#
#     print('most influence point::', most_influence_point)
#
#     print(all_influence_points)
#
#     train_ids = [id for id in train_ids if not id == most_influence_point]
#
#
# #         training_dataset.labels[most_influence_point] = Y_train_full[most_influence_point]
#
# #     update_training_data = origin_training_dataset.data[torch.tensor(train_ids)]
# #      
# #     update_training_labels = origin_training_dataset.labels[torch.tensor(train_ids)]
# #      
# #     training_dataset.data = update_training_data
# #      
# #     training_dataset.labels = update_training_labels
# #         
# # #         model = do_training(args, training_dataset)
# # 
# #     print(all_influence_points)
# #         
# #     influential_training_data = origin_training_dataset.data[sorted_train_ids[0:count]]
# #     
# #     influential_training_labels = Y_train_full[sorted_train_ids[0:count]]
#
# #     sorted_train_ids = torch.randperm(Y_train_full.shape[0])[0:count]
#
# #     r_weight = torch.ones(Y_train_full.shape[0], dtype = Y_train_full.dtype, device = Y_train_full.device)*noisy_weight
# #     
# #     r_weight[sorted_train_ids[0:count]] = 1
# #     
# # #     training_dataset.data = torch.cat([training_dataset.data, influential_training_data], dim = 0)
# #          
# #     training_dataset.labels[sorted_train_ids[0:count]] = Y_train_full[sorted_train_ids[0:count]]
# # 
# #     print('after cleaning by influence function::')
# #     
# #     training_dataset.data = training_dataset.data[~(training_dataset.labels == -1)]
# #     
# #     training_dataset.labels = training_dataset.labels[~(training_dataset.labels == -1)]
# #     
# #     model = do_training_general(args, training_dataset, model_name, num_class, r_weight = r_weight)
# #     
# #     evaluate_model_test_dataset(dataset_test, model, args)
#
#     return influences, ordered_list, sorted_train_ids



# def evaluate_influence_function_repetitive_incremental(args, batch_size, regularization_term, is_GPU, device, full_out_dir, training_dataset, valid_dataset, model, loss_func, optimizer, learning_rate = 0.0002, r_weight = None):
#
#     train_ids = list(range(training_dataset.data.shape[0]))
#
#     origin_training_dataset = models.MyDataset(training_dataset.data.clone(), training_dataset.labels.clone())    
#
#
# #     model = do_training(args, training_dataset)
# #     
# #     evaluate_model_test_dataset(test_dataset, model, args)
#
#
#     all_influence_points = []
#
# #     for i in range(count):
#
#
# #     ptif.init_logging()
#     config = ptif.get_default_config()
#     train_DL = DataLoader(training_dataset, batch_size=batch_size*2, shuffle = True)
#
#     valid_DL = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle = True)
#
#
#
#     config['test_sample_num'] = 0
#
#     config['outdir'] = full_out_dir
#
#     if is_GPU:
#         config['gpu'] = device
#     else:
#         config['gpu'] = None
#
#     config['single_train'] = False
#
#     config['loss_func'] = loss_func
#
#     config['regularization_term'] = regularization_term
# #         influences = ptif.calc_img_wise_with_ids(config, model, train_DL, test_DL, test_ids)
#
#
# #     s_test_vec = ptif.calc_img_wise_multi_with_ids_incremental2(config, model, training_dataset, valid_dataset, train_ids, batch_size, f1 = False)
#
#     # s_test_vec = ptif.calc_img_wise_multi_with_ids_incremental(config, model, train_DL, valid_DL, train_ids, f1 = False)
#
#     curr_valid_r_weights = torch.ones(valid_dataset.data.shape[0], device = device)
#
#     # curr_valid_r_weights = torch.ones(valid_dataset.data.shape[0], device = device)
#
#     valid_grad,_,_ = compute_gradient(model, valid_dataset.data, valid_dataset.labels, torch.nn.CrossEntropyLoss(), curr_valid_r_weights, 0, optimizer, is_GPU, device, random_sampling = False, bz = batch_size, batch_ids = None)
#
#
#     curr_train_r_weights = r_weight# torch.ones(training_dataset.data.shape[0], device = device)
#
#     s_test_vec = compute_conjugate_grad3(model, valid_grad, training_dataset.data, training_dataset.labels, loss_func, optimizer, curr_train_r_weights, 0, is_GPU, device, exp_res = None, learning_rate = args.derived_lr, random_sampling = True, bz = args.derived_bz, running_iter = args.derived_epochs, regularization_rate=args.derived_l2)
#
#     print('influences::')
#
#     return s_test_vec


def evaluate_influence_function_repetitive_incremental2(args, batch_size, regularization_term, is_GPU, device, full_out_dir, training_dataset, valid_dataset, model, loss_func, optimizer, learning_rate = 0.0002, r_weight = None):
    
    train_ids = list(range(training_dataset.data.shape[0]))
    
    origin_training_dataset = models.MyDataset(training_dataset.data.clone(), training_dataset.labels.clone())    
    
    
#     model = do_training(args, training_dataset)
#     
#     evaluate_model_test_dataset(test_dataset, model, args)
    
    
    all_influence_points = []
    
#     for i in range(count):
        
    
#     ptif.init_logging()
    # config = ptif.get_default_config()
    train_DL = DataLoader(training_dataset, batch_size=batch_size*2, shuffle = True)
        
    valid_DL = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle = True)
    
    
    
    # config['test_sample_num'] = 0
    #
    # config['outdir'] = full_out_dir
    #
    # if is_GPU:
    #     config['gpu'] = device
    # else:
    #     config['gpu'] = None
    #
    # config['single_train'] = False
    #
    # config['loss_func'] = loss_func
    #
    # config['regularization_term'] = regularization_term
#         influences = ptif.calc_img_wise_with_ids(config, model, train_DL, test_DL, test_ids)
    
    
#     s_test_vec = ptif.calc_img_wise_multi_with_ids_incremental2(config, model, training_dataset, valid_dataset, train_ids, batch_size, f1 = False)
    
    # s_test_vec = ptif.calc_img_wise_multi_with_ids_incremental(config, model, train_DL, valid_DL, train_ids, f1 = False)
    
    curr_valid_r_weights = torch.ones(valid_dataset.data.shape[0], device = device)
    
    # curr_valid_r_weights = torch.ones(valid_dataset.data.shape[0], device = device)
    
    valid_grad,_,_ = compute_gradient(model, valid_dataset.data, valid_dataset.labels, torch.nn.CrossEntropyLoss(), curr_valid_r_weights, 0, optimizer, is_GPU, device, random_sampling = False, bz = batch_size, batch_ids = None)
    
    
    curr_train_r_weights = r_weight# torch.ones(training_dataset.data.shape[0], device = device)
    
    r = 1e-8
    
    print('before s test vec::', training_dataset.data.shape, torch.unique(curr_train_r_weights))
    
    s_test_vec = compute_conjugate_grad3(model, valid_grad, training_dataset.data, training_dataset.labels, loss_func, optimizer, curr_train_r_weights, 0, is_GPU, device, exp_res = None, learning_rate = args.derived_lr, random_sampling = True, bz = args.derived_bz, running_iter = args.derived_epochs, regularization_rate=args.derived_l2)
    
    print('influences::')

    return s_test_vec
#     most_influence_point, ordered_list, sorted_train_ids = get_most_influential_point(influences, train_ids)
#     
#     sorted_train_ids = sorted_train_ids.cpu()
#     
# #         torch.save(sorted_train_ids, )
# #     print(sorted_train_ids[0:count])
# 
#     all_influence_points.append(most_influence_point)
# 
#     print('most influence point::', most_influence_point)
#     
#     print(all_influence_points)
# 
#     train_ids = [id for id in train_ids if not id == most_influence_point]
#     
#     
# 
#     return influences, ordered_list, sorted_train_ids, s_test_vec

    
# def evaluate_influence_function_repetitive_del(args, training_DL, val_DL, test_DL, training_dataset, valid_dataset, dataset_test, model, count, loss_func, model_name, num_class):
#
#     if training_DL is None:
#         train_ids = list(range(training_dataset.data.shape[0]))
#     else:
#         train_ids = list(range(training_DL.dataset.lenth))
#
#     if training_DL is None:
#         origin_training_dataset = models.MyDataset(training_dataset.data.clone(), training_dataset.labels.clone())    
#
#
# #     model = do_training(args, training_dataset)
# #     
# #     evaluate_model_test_dataset(test_dataset, model, args)
#
#
#     all_influence_points = []
#
# #     for i in range(count):
#
#
#     ptif.init_logging()
#     config = ptif.get_default_config()
#
#     if training_DL is None:
#         train_DL = DataLoader(training_dataset, batch_size=args.bz)
#
#         valid_DL = DataLoader(valid_dataset, batch_size=args.bz)
#
#     else:
#         train_DL = training_DL
#
#         valid_DL = val_DL
#
#     config['test_sample_num'] = 0
#
#     config['outdir'] = args.output_dir
#
#     if args.GPU:
#         config['gpu'] = args.GPUID
#     else:
#         config['gpu'] = -1
#
#     config['single_train'] = False
#
#     config['loss_func'] = loss_func
#
# #         influences = ptif.calc_img_wise_with_ids(config, model, train_DL, test_DL, test_ids)
#     print('valid data::', valid_DL.dataset)
#
#     influences = ptif.calc_img_wise_multi_with_ids(config, model, train_DL, valid_DL, train_ids, f1 = args.f1)
#
#     print('influences::')
#
#
#
#     most_influence_point, ordered_list, sorted_train_ids = get_most_influential_point(influences, train_ids)
#
#     sorted_train_ids = sorted_train_ids.cpu()
#
# #         torch.save(sorted_train_ids, )
#     print(sorted_train_ids[0:count])
#
#     all_influence_points.append(most_influence_point)
#
#     print('most influence point::', most_influence_point)
#
#     print(all_influence_points)
#
#     train_ids = [id for id in train_ids if not id == most_influence_point]
#
# #     sorted_train_ids = torch.randperm(len(train_ids))
#
#
#
#
# #         training_dataset.labels[most_influence_point] = Y_train_full[most_influence_point]
#
# #     update_training_data = origin_training_dataset.data[torch.tensor(train_ids)]
# #      
# #     update_training_labels = origin_training_dataset.labels[torch.tensor(train_ids)]
# #      
# #     training_dataset.data = update_training_data
# #      
# #     training_dataset.labels = update_training_labels
# #         
# # #         model = do_training(args, training_dataset)
# # 
# #     print(all_influence_points)
# #         
# #     influential_training_data = origin_training_dataset.data[sorted_train_ids[0:count]]
# #     
# #     influential_training_labels = Y_train_full[sorted_train_ids[0:count]]
#
# #     sorted_train_ids = torch.randperm(Y_train_full.shape[0])[0:count]
#
# #     r_weight = torch.ones(training_dataset.data.shape[0], dtype = training_dataset.data.dtype, device = training_dataset.data.device)
# #     
# #     r_weight[sorted_train_ids[0:count]] = 1
#
# #     training_dataset.data = training_dataset.data[sorted_train_ids[count:]]
# #           
# #     training_dataset.labels = training_dataset.labels[sorted_train_ids[count:]] 
#
#     print('after cleaning by influence function::')
#
#     if training_DL is None:
#
#         model = do_training_general_hard(args, training_dataset, valid_dataset, dataset_test, model_name, num_class)
#
#         if dataset_test is not None:
#             evaluate_model_test_dataset(dataset_test, model, args)
#         else:
#             evaluate_model_test_dataset(valid_dataset, model, args)
#
#     else:
#
#         train_DL.dataset.skipped_dataset = sorted_train_ids[0:count]
#
#         model = do_training_general_hard_DL(args, train_DL, val_DL, model_name, num_class, skipped_ids = sorted_train_ids[0:count])
#
#
#         if test_DL is not None:
#             valid_model(model, test_DL, None, 'test', args.GPU, args.device, f1 = args.f1)
#
# #         if dataset_test is not None:
# #             evaluate_model_test_dataset(dataset_test, model, args)
# #         else:
# #             evaluate_model_test_dataset(valid_dataset, model, args)
#
#
#     return most_influence_point, ordered_list, sorted_train_ids

def convert_to_binary_classification_model(args, model, training_dataset):
    bin_model = models.Binary_Logistic_regression(model.fc1.in_features,bias=False)
    
    bin_model.fc1.weight.data.copy_(model.fc1.weight[1].data - model.fc1.weight[0].data)
    
    if args.GPU:
        bin_model = bin_model.to(args.device)
    
    mini_data = training_dataset.data[0].type(torch.DoubleTensor)
    
    if args.GPU:
        mini_data = mini_data.to(args.device)
    
    print(F.softmax(model(mini_data)))
    
    print(bin_model(mini_data))    
    
    
    
    
    return bin_model

def clone_model_weights(model):
    if model.fc1.bias is None:
        bin_model = models.Binary_Logistic_regression(model.fc1.in_features,bias=False)
    else:
        bin_model = models.Binary_Logistic_regression(model.fc1.in_features,bias=True)
    
    bin_model.fc1.weight.data.copy_(model.fc1.weight.data)
    
    if model.fc1.bias is not None:
        bin_model.fc1.bias.data.copy_(model.fc1.bias.data)
    
    return bin_model


def evaluate_our_method(model, training_dataset, Y_train_full, test_dataset,  args, removed_count):
    
    #int(dataset_train.data.shape[0]*args.del_ratio)
    
#     bin_model = models.Binary_Logistic_regression(training_dataset.data.shape[1], bias = False)
#     
#     if args.GPU:
#         bin_model = bin_model.to(args.device)
#     
#     
#     optimizer = bin_model.get_optimizer(args.lr, args.wd)
    
    criterion = model.soft_loss_function
    
    gap = args.jump
    
    
#     origin_model = convert_to_binary_classification_model(args, model, training_dataset)

    origin_model = clone_model_weights(model)
    
    
    delta_w, removed_ids = model.optimize_two_steps_delete_perturb(training_dataset, origin_model, args.wd, args.out_epoch_count, args.inner_epoch_count, None, None, args.lr, args.bz, removed_count, args.tlr, args.wd, args.GPU, args.device, gap, args.norm, lower_bound = None, upper_bound = None, epsilon=0, loss_func = criterion, hessian_lr = args.hlr, hessian_eps = args.heps)
    
        
    print('removed_ids::', removed_ids)
    
    
    
    training_dataset.labels[removed_ids] = Y_train_full[removed_ids]
    
    updated_model = do_training(args, training_dataset)
    
    evaluate_model_test_dataset(test_dataset, updated_model, args)
#     bin_model.train    
    
    

def get_train_test_data_loader_lr(train_X, train_Y, test_X, test_Y):
    
    
    dataset_train = models.MyDataset(train_X, train_Y)
    dataset_test = models.MyDataset(test_X, test_Y)
    
#     data_train_loader = DataLoader(dataset_train, batch_size=specified_batch_size, shuffle=True, num_workers=0)
#     data_test_loader = DataLoader(dataset_test, batch_size=specified_batch_size, num_workers=0)

    return dataset_train, dataset_test

def get_train_test_data_loader_by_name_lr(data_preparer, git_ignore_folder, name):
    
    
    function=getattr(models.Data_preparer, "prepare_" + name)
    
    train_X, train_Y, test_X, test_Y = function(data_preparer, git_ignore_folder)
    
    if isinstance(test_X, tuple):
        dataset_train = models.MyDataset(train_X, train_Y)
        dataset_valid = models.MyDataset(list(test_X)[0], list(test_Y)[0])
        dataset_test = models.MyDataset(list(test_X)[1], list(test_Y)[1])
        return dataset_train, dataset_valid, dataset_test 
#         dataset_train, dataset_test = get_train_test_data_loader_lr(train_X, train_Y, test_X, test_Y)
    else:
        dataset_train, dataset_test = get_train_test_data_loader_lr(train_X, train_Y, test_X, test_Y)
    
        return dataset_train, dataset_test

def select_subset_data(dataset_train, label1, label2):
    
    ids1 = (torch.nonzero(dataset_train.labels == label1)).view(-1)
    
    ids2 = (torch.nonzero(dataset_train.labels == label2)).view(-1)
    
    selected_samples = []
    
    selected_labels = []
    
    selected_samples.append(dataset_train.data[ids1])
    
    selected_samples.append(dataset_train.data[ids2])
    
    
    selected_samples_tensor = torch.cat(selected_samples, 0)
    
    selected_labels.append(dataset_train.labels[ids1])
    
    selected_labels.append(dataset_train.labels[ids2])
    
    selected_labels_tensor = torch.cat(selected_labels, 0)
    
    selected_labels_tensor[0:ids1.shape[0]]=0
    
    selected_labels_tensor[ids1.shape[0]:ids2.shape[0]+ids1.shape[0]]=1
    
    perturbed_ids = torch.randperm(selected_samples_tensor.shape[0])
    
    selected_samples_tensor_final = selected_samples_tensor[perturbed_ids]
    
    selected_labels_tensor_final = selected_labels_tensor[perturbed_ids].type(torch.DoubleTensor)
    
    selected_dataset = models.MyDataset(selected_samples_tensor_final, selected_labels_tensor_final)
    
    return selected_dataset


def partition_val_dataset(dataset_train, ratio):
    random_ids = torch.randperm(dataset_train.data.shape[0])
    
    val_count = int(random_ids.shape[0]*ratio)
    
    val_data = dataset_train.data[random_ids[0:val_count]]
    
    val_label = dataset_train.labels[random_ids[0:val_count]]
    
    train_data = dataset_train.data[random_ids[val_count:]]
    
    train_label = dataset_train.labels[random_ids[val_count:]]
    
    val_dataset = models.MyDataset(val_data, val_label)
    
    train_dataset = models.MyDataset(train_data, train_label)
    
    return train_dataset, val_dataset
    

def partition_hard_labeled_data_noisy_data(dataset_train, ratio, args):
    
    small_sample_count = int(dataset_train.data.shape[0]*ratio)
    
    random_ids = torch.randperm(dataset_train.data.shape[0])
    
    selected_small_sample_ids = random_ids[0:small_sample_count]
    
    selected_noisy_sample_ids = random_ids[small_sample_count:] 
    
    selected_small_samples = dataset_train.data[selected_small_sample_ids]
    
    selected_small_labels = dataset_train.labels[selected_small_sample_ids]
    
    selected_noisy_samples = dataset_train.data[selected_noisy_sample_ids]
    
    selected_noisy_origin_labels = dataset_train.labels[selected_noisy_sample_ids]

    small_dataset_train = models.MyDataset(selected_small_samples, selected_small_labels)
    
    return small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels

def partition_hard_labeled_data_noisy_data2(dataset_train, ratio, args):
    
    certain_dataset_train_data_ids = (~(dataset_train.labels == -1))
    
    uncertain_dataset_train_data_ids = (dataset_train.labels == -1)
    
    certain_dataset_train_data = dataset_train.data[certain_dataset_train_data_ids]
    
    certain_dataset_train_labels = dataset_train.labels[certain_dataset_train_data_ids]
    
    small_sample_count = int(certain_dataset_train_data.shape[0]*ratio)
    
    random_ids = torch.randperm(certain_dataset_train_data.shape[0])
    
    selected_small_sample_ids = random_ids[0:small_sample_count]
    
#     selected_noisy_sample_ids = random_ids[small_sample_count:] 
    
    
    selected_certain_sample_ids = torch.nonzero(certain_dataset_train_data_ids)[selected_small_sample_ids].view(-1)
    
    selected_noisy_sample_ids = torch.tensor(list(set(list(range(dataset_train.data.shape[0]))).difference(set(selected_certain_sample_ids.view(-1).tolist()))))
    
#     selected_small_samples = certain_dataset_train_data[selected_small_sample_ids]
#     
#     selected_small_labels = certain_dataset_train_labels[selected_small_sample_ids]
    
    selected_noisy_samples = dataset_train.data[selected_noisy_sample_ids] #torch.cat([certain_dataset_train_data[selected_noisy_sample_ids], dataset_train.data[uncertain_dataset_train_data_ids]], dim = 0)
    
    selected_noisy_origin_labels = dataset_train.labels[selected_noisy_sample_ids]#torch.cat([certain_dataset_train_labels[selected_noisy_sample_ids], dataset_train.labels[uncertain_dataset_train_data_ids]], dim = 0)

    small_dataset_train = models.MyDataset(dataset_train.data[selected_certain_sample_ids], dataset_train.labels[selected_certain_sample_ids])
    
    return small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, selected_certain_sample_ids, selected_noisy_sample_ids



def partition_hard_labeled_data_noisy_data3(dataset_train, ratio, args):
    
    certain_dataset_train_data_ids = (~(dataset_train.labels == -1)).view(-1)
    
    uncertain_dataset_train_data_ids = (dataset_train.labels == -1).view(-1)
    
#     model = models.Logistic_regression(dataset_train.data.shape[0], args.num_class)
    
    
    certain_dataset_train_data = dataset_train.data[certain_dataset_train_data_ids]
    
    certain_dataset_train_labels = dataset_train.labels[certain_dataset_train_data_ids]
    
    certain_dataset_train = models.MyDataset(certain_dataset_train_data, certain_dataset_train_labels)
    
#     print(torch.unique(certain_dataset_train_labels))
    
    model = do_training_general(args, certain_dataset_train, 'Logistic_regression', args.num_class, None, False, certain_dataset_train, certain_dataset_train)
    
    
    all_mislabeled_ids = get_mislabeled_ids(model, certain_dataset_train, args).view(-1)
    
    selected_count = int(all_mislabeled_ids.shape[0]*ratio)
    
    if selected_count <= 0:
        selected_count = 2
    
    mislabeled_ids = all_mislabeled_ids[torch.randperm(all_mislabeled_ids.shape[0])[0:selected_count]]
    
    
    
    model = model.to('cpu')
    
    print(torch.nonzero(model.determine_labels(certain_dataset_train.data[mislabeled_ids]) == certain_dataset_train.labels[mislabeled_ids]))
    
    correct_labeled_ids = torch.tensor(list(set(list(range(certain_dataset_train.data.shape[0]))).difference(set(mislabeled_ids.view(-1).tolist()))))
    
#     model = do_training_general(args, mislabeled_dataset_train, 'Logistic_regression', args.num_class, None, False, dataset_train, dataset_train)
    
    
#     evaluate_model_test_dataset(test_dataset, model, args, tag)
    


    selected_certain_sample_ids = torch.nonzero(certain_dataset_train_data_ids)[mislabeled_ids].view(-1)
    
    selected_noisy_sample_ids = torch.tensor(list(set(list(range(dataset_train.data.shape[0]))).difference(set(selected_certain_sample_ids.view(-1).tolist()))))
    
#     selected_small_samples = certain_dataset_train_data[selected_small_sample_ids]
#     
#     selected_small_labels = certain_dataset_train_labels[selected_small_sample_ids]
    
    selected_noisy_samples = dataset_train.data[selected_noisy_sample_ids] #torch.cat([certain_dataset_train_data[selected_noisy_sample_ids], dataset_train.data[uncertain_dataset_train_data_ids]], dim = 0)
    
    selected_noisy_origin_labels = dataset_train.labels[selected_noisy_sample_ids]#torch.cat([certain_dataset_train_labels[selected_noisy_sample_ids], dataset_train.labels[uncertain_dataset_train_data_ids]], dim = 0)


    small_data = dataset_train.data[selected_certain_sample_ids]
    
    small_labels = dataset_train.labels[selected_certain_sample_ids]

    
    print('small dataset count::', small_labels.shape[0], torch.unique(small_labels))

    for k in range(args.num_class):
        if k not in small_labels:
            sample_id_with_curr_class = torch.nonzero(dataset_train.labels == k).view(-1)
            selected_sample_id_with_curr_class = np.random.choice(sample_id_with_curr_class.shape[0], 1)
            selected_sample_id_with_curr_class = sample_id_with_curr_class[selected_sample_id_with_curr_class]
#             added_sample_with_curr_class = torch.unsqueeze(dataset_train.data[selected_sample_id_with_curr_class], 0)
            added_sample_with_curr_class = dataset_train.data[selected_sample_id_with_curr_class]
#             print(k, small_data.shape, added_sample_with_curr_class.shape, selected_sample_id_with_curr_class, dataset_train.data[selected_sample_id_with_curr_class].shape)
            
            added_sample_label_with_curr_class = dataset_train.labels[selected_sample_id_with_curr_class]
            small_data = torch.cat([small_data, added_sample_with_curr_class], dim = 0)
            small_labels = torch.cat([small_labels, added_sample_label_with_curr_class], dim =0)
            
            print(selected_sample_id_with_curr_class, selected_certain_sample_ids)
            selected_certain_sample_ids = torch.cat([selected_certain_sample_ids, torch.tensor([selected_sample_id_with_curr_class.item()])], dim = 0)
#             print('small dataset count::', selected_noisy_origin_labels.shape[0], torch.unique(selected_noisy_origin_labels))
    
    small_dataset_train = models.MyDataset(small_data, small_labels)
    print(small_data.shape, small_labels.shape, selected_certain_sample_ids.shape)
    
    return small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, selected_certain_sample_ids, selected_noisy_sample_ids

    

#     selected_noisy_samples = torch.cat([certain_dataset_train_data[correct_labeled_ids], dataset_train.data[uncertain_dataset_train_data_ids]], dim = 0)
#     
#     selected_noisy_origin_labels = torch.cat([certain_dataset_train_labels[correct_labeled_ids], dataset_train.labels[uncertain_dataset_train_data_ids]], dim = 0)
# 
# 
# #     selected_noisy_samples = torch.cat([certain_dataset_train[correct_labeled_ids], ])
#     small_dataset_train = models.MyDataset(certain_dataset_train.data[mislabeled_ids], certain_dataset_train.labels[mislabeled_ids].view(-1))
#     
#     return small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, mislabeled_ids, correct_labeled_ids



def labeling_noisy_samples(model, selected_noisy_samples, args, bz, soft = False):
    
    dataset_size = selected_noisy_samples.shape[0]
    
    noisy_labels = []
    
    with torch.no_grad():
        for k in range(0, dataset_size, bz):
            end_id = k + bz
            
            if end_id > dataset_size:
                end_id = dataset_size
            
            
#             curr_rand_ids = random_ids[k: end_id]
            
            X = selected_noisy_samples[k: end_id]
            
            X = X.type(torch.DoubleTensor)

            if args.GPU:
                X = X.to(args.device)

            labels = model.determine_labels(X, soft = soft)
    
            noisy_labels.append(labels)
            
    selected_noisy_labels = torch.cat(noisy_labels, dim = 0)
    
    return selected_noisy_labels


def get_data_class_num_by_name(data_preparer, name):
    
    
    function=getattr(models.Data_preparer, "get_num_class_" + name)
    
    num_class = function(data_preparer)
    
    
#     dataset_train, dataset_test, data_train_loader, data_test_loader = get_train_test_data_loader(Model, training_data, test_data, specified_batch_size)
    
    return num_class

def generate_random_noisy_labels1(dataset_train, args, ratio, soft_label):
    
    small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels = partition_hard_labeled_data_noisy_data(dataset_train, ratio, args)
    
    model = do_training_general(args, small_dataset_train, args.model, num_class = args.num_class)
    
    
    evaluate_model_test_dataset(small_dataset_train, model, args, tag = 'training')
    
    selected_noisy_labels = labeling_noisy_samples(model, selected_noisy_samples, args, args.bz, soft = soft_label)
    
#     full_noisy_samples = torch.cat([small_dataset_train.data, selected_noisy_samples], dim = 0)
#     
#     full_noisy_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_labels.view(-1, small_dataset_train.labels.shape[1])], dim = 0)
#     
#     full_origin_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])], dim = 0)
    
    meta_count = int(selected_noisy_samples.shape[0]*0.1)
    
    rand_ids = torch.randperm(selected_noisy_samples.shape[0])
    
    meta_sample_ids = rand_ids[0:meta_count]
    
    remaining_ids = rand_ids[meta_count:]
    
    
    valid_dataset = models.MyDataset(selected_noisy_samples[meta_sample_ids], selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])[meta_sample_ids])
    
    final_selected_noisy_samples = selected_noisy_samples[remaining_ids]
    
    final_selected_noisy_labels = selected_noisy_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])[remaining_ids]
    
    final_selected_noisy_origin_labels = selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])[remaining_ids]
    
    
    full_noisy_samples = torch.cat([final_selected_noisy_samples], dim = 0)
    
    full_noisy_labels = torch.cat([final_selected_noisy_labels], dim = 0)
    
    full_origin_labels = torch.cat([final_selected_noisy_origin_labels], dim = 0)
    
    
    full_noisy_dataset = models.MyDataset(full_noisy_samples, full_noisy_labels)
    
    
    
    
    

    
    
    
#     full_training_noisy_dataset = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
#     
#     
#     
#     full_training_origin_labels = full_training_origin_labels[remaining_ids]
    
    
    
    return full_noisy_dataset, full_origin_labels, small_dataset_train, valid_dataset
    
    






