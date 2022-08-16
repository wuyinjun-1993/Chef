'''
Created on Dec 4, 2020

'''
import numpy as np

import torch


r=1e-5

iters = 10


import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/real_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/iterative_detect')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')

# import models



try:
    from models.logistic_regression import *
    # from iterative_detect.utils_iters import *
#     from utils.utils import *
    # from models.utils_real import *
except ImportError:
#     from utils import *
    # from utils_real import *
    from logistic_regression import *
    # from utils_iters import *
stop_number = 0.1

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

def compute_model_para_diff(model1_para_list, model2_para_list):
    
    diff = 0
    
    norm1 = 0
    
    norm2 = 0
    
    all_dot = 0
    
    
    for i in range(len(model1_para_list)):
        
        param1 = model1_para_list[i].to('cpu')
        
        param2 = model2_para_list[i].to('cpu')
        
        curr_diff = torch.norm(param1 - param2, p='fro')
        
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        
        
        all_dot += torch.sum(param1*param2)
        
#         print("curr_diff:", i, curr_diff)
        
        diff += curr_diff*curr_diff
        
    print('model difference (l2 norm):', torch.sqrt(diff))
    
    
def f1_loss(y_pred:torch.Tensor, y_true:torch.Tensor, r_weight=None, is_training=False, num_class = 2) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1 or y_true.ndim == 2
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    
#     onehot_coding = onehot(y_true, num_class)
    
#     if y_pred.ndim == 2:
#         y_pred = y_pred.argmax(dim=1)
#     print(y_true.device, y_pred.device)
    if r_weight is None:
        r_weight = torch.ones(y_true.shape[0], dtype = torch.double, device = y_true.device)
    
    
    r_weight = r_weight.type_as(y_pred).view(-1,1)
    
#     y_true = y_true.type(torch.DoubleTensor).view(-1,1)
    if y_true.ndim == 1:
        y_true = y_true.type(torch.LongTensor)
        y_true = onehot(y_true, num_class).type_as(y_pred)
    
    if y_pred.ndim == 2:
        y_pred = y_pred.type_as(r_weight).view(y_true.shape)
    else:
#         y_pred = y_pred.type(torch.DoubleTensor).view(-1,1)
        y_pred = onehot(y_pred, num_class)
        
#     print(y_true.device, y_pred.device, r_weight.device)
#     onehot_coding = torch.ones_like(y_pred)
    tp = torch.sum(y_true * y_pred * r_weight, dim = 0)/torch.sum(r_weight, dim = 0)
    tn = torch.sum((1 - y_true) * (1 - y_pred) * r_weight, dim = 0)/torch.sum(r_weight, dim =0)
    fp = torch.sum((1 - y_true) * y_pred * r_weight, dim = 0)/torch.sum(r_weight, dim = 0)
    fn = torch.sum(y_true * (1 - y_pred) * r_weight, dim = 0)/torch.sum(r_weight, dim = 0)
    
    r_weight_sum_by_class = torch.sum(r_weight*y_true, dim = 0)
    
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1_all = 2* (precision*recall) / (precision + recall + epsilon)
#     print(f1)
#     f1 = torch.tensor([0.4,0.5,0],dtype = torch.double)
    f1 = torch.sum(f1_all* r_weight_sum_by_class)/torch.sum(r_weight_sum_by_class)
#     f1.requires_grad = is_training
    return -f1


def get_all_vectorized_parameters1(para_list, device = None):
    
    res_list = []
    
    i = 0
    
    for param in para_list:
        
        if device is None:
            res_list.append(param.data.cpu().view(-1))
        else:
            res_list.append(param.data.to(device).view(-1))
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def get_vectorized_params(model):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        res_list.append(param.data.to('cpu').detach().view(-1).clone())
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def get_model_para_list(model):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        res_list.append(param.data.cpu().clone())
        
        i += 1
        
    return res_list

def get_model_grad_list(model):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        res_list.append(param.grad.clone())
        
        i += 1
        
    return res_list

def get_vectorized_grads(model, device = None):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        if device is None:
            res_list.append(param.grad.data.detach().to('cpu').view(-1).clone())
        else:
            res_list.append(param.grad.data.detach().to(device).view(-1).clone())
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def get_vectorized_param_grads(para_list, device = None):
    
    res_list = []
    #
    # para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        if device is None:
            res_list.append(param.grad.data.to('cpu').view(-1).clone())
        else:
            res_list.append(param.grad.data.to(device).view(-1).clone())
        
        i += 1
        
    return torch.cat(res_list, 0).view(1,-1)

def get_vectorized_grads_sample_wise(model, w_list, grad_list, m, optimizer, X, Y, device = None):
    
    res_list = []
    
    para_list = model.parameters()
    
    i = 0
    
    loss_func = model.soft_loss_function_reduce
    
    

    for r in range(m):

        curr_grad_list = []

        set_model_parameters(model, w_list[-(m-r)], device)

        for k in range(X.shape[0]):
        
            print('sample::', k)
            
            optimizer.zero_grad()
            
            loss = loss_func(model(X[k:k+1]), Y[k:k+1])
            
            loss.backward()
            
            curr_grad = get_vectorized_grads(model, device)
            
            curr_grad_list.append(curr_grad.view(-1))
        
        grad_list_tensor = torch.stack(curr_grad_list, 0)
        
#         print(torch.norm(grad_list_tensor - grad_list[-(m-r)]))
        
        res_list.append(grad_list_tensor)
    
    
    final_full_grad = torch.stack(res_list, 0)
    
    final_mean_grad = torch.mean(final_full_grad[-1], dim = 0)
    
    optimizer.zero_grad()
        
    loss = loss_func(model(X), Y)
    
    loss.backward()
    
    exp_grad = get_vectorized_grads(model, device)
    
    print(torch.norm(exp_grad.view(-1) - final_mean_grad.view(-1)))
    
    return final_full_grad 

def get_devectorized_parameters(params, full_shape_list, shape_list):
    
    params = params.view(-1)
    
    para_list = []
    
    pos = 0
    
    for i in range(len(full_shape_list)):

        param = 0
        if len(full_shape_list[i]) >= 2:
            
            curr_shape_list = list(full_shape_list[i])

            param = params[pos: pos+shape_list[i]].view(curr_shape_list)
            
        else:
            param = params[pos: pos+shape_list[i]].view(full_shape_list[i])
        
        para_list.append(param)
    
        
        pos += shape_list[i]
    
    return para_list


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



def add_perturbation_on_model_weight(devec_res, model, origin_model_para_list, device,r=1e-5):
    para_list = model.parameters()
    
    i = 0
    
    for param in para_list:
        
        curr_weight = origin_model_para_list[i] + r*devec_res[i]
        
        curr_weight = curr_weight.to(device)
        
        param.data.copy_(curr_weight)
        
        i += 1


def compute_gradient2(model, X, Y, optimizer, curr_weight, removed_count):
    
    model_out = model(X)
    
    theta = get_vectorized_params(model)
    
    term1 = optimizer.param_groups[0]['weight_decay']*theta.view(1,-1) - torch.sum((Y.view(-1,1) - model_out.view(-1,1))*X*curr_weight.view(-1,1), dim=0)/(X.shape[0] - removed_count)
    
    return term1    

def compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, isGPU, device, random_sampling = False, bz = 1, batch_ids = None):
    
    sampled_ids = None
    
    optimizer.zero_grad()
    
    if not random_sampling:


        grad_sum = 0        

        for k in range(0, X.shape[0], bz):
            
            end_id = k + bz
            
            if end_id > X.shape[0]:
                end_id = X.shape[0]
            
            batch_x = X[k:end_id]
            
            batch_Y = Y[k:end_id]
        
            batch_r_weight = curr_r_weights[k:end_id]
            
            if isGPU:
                batch_x = batch_x.to(device)
                batch_Y = batch_Y.to(device)
                batch_r_weight = batch_r_weight.to(device)
            # model = model.cpu()
            #
            # X = X.cpu()
            #
            # Y = Y.cpu()
        
        # curr_r_weights = curr_r_weights.cpu
        # if isGPU:
            # X = X.to(device)
            # Y = Y.to(device)
            # curr_r_weights = curr_r_weights.to(device)
        # print(next(model.parameters()).is_cuda)
        # print(X.device)
        
            optimizer.zero_grad()
        
            model_out = model(batch_x)
            
            loss = torch.sum(loss_func(model_out, batch_Y).view(-1)*batch_r_weight.view(-1))/(end_id - k)
            
            loss.backward()
        
            model_grad = get_vectorized_grads(model, device)
            
            grad_sum += model_grad*(end_id-k)
            # model_param = get_vectorized_params(model)
        
        
        model_grad = grad_sum/X.shape[0]
        model_param = get_all_vectorized_parameters1(list(model.parameters()), device)
        
        
        
        
        
        
        # optimizer.zero_grad()
        #
        # model_out = model(X)
        #
        # loss = torch.sum(loss_func(model_out, Y).view(-1)*curr_r_weights.view(-1))/X.shape[0]
        #
        # loss.backward()
        #
        # model_grad2 = get_vectorized_grads(model)
        #
        # print(torch.norm(model_grad - model_grad2))
        #
        # print('here')
        # if isGPU:
            # model = model.to(device)
            #
    else:
        
        probs = curr_r_weights.view(-1)/torch.sum(curr_r_weights)
        
        # if batch_ids is None:
        #     sampled_ids = torch.from_numpy(np.random.choice(X.shape[0], bz, p=probs.detach().numpy()))
        #
        #     batch_ids = sampled_ids
            
        
            #
        # else:
            # sampled_ids = batch_ids
        
        
        if batch_ids is None:
        
            random_ids = torch.randperm(X.shape[0])
            
            batch_ids = random_ids[0:bz]
        print('batch_ids::', batch_ids[0:10])
        sampled_ids = batch_ids
        
        batch_X = X[batch_ids]
        
        batch_Y = Y[batch_ids].type(torch.DoubleTensor)
        
        batch_r_weight = curr_r_weights[batch_ids]
        
        if isGPU:
            model = model.to(device)
            
            batch_X = batch_X.to(device)
            
            batch_Y = batch_Y.to(device)
            
            batch_r_weight = batch_r_weight.to(device)
        
        # print(list(model.parameters())[0].device, batch_X.device, batch_Y.device, batch_r_weight.device)
        
        model_out = model(batch_X)
        
        # print('curr model grad::', torch.norm(get_vectorized_grads(model)))
        
        loss = torch.mean(loss_func(model_out.view(batch_Y.shape), batch_Y).view(-1)*batch_r_weight.view(-1))
        
            
    
    
    
        loss.backward()
        
    #     optimizer.step()
    
        model_grad = get_vectorized_grads(model, device)
        
        model_param = get_all_vectorized_parameters1(list(model.parameters()), device)
    
    return model_grad, model_param, sampled_ids


def compute_gradient3(model, X, Y, loss_func, curr_r_weights, res_vec_list, optimizer, isGPU, device, random_sampling = False, bz = 1, batch_ids = None):
    
    sampled_ids = None
    
    optimizer.zero_grad()
    
    # if not random_sampling:
    #
    #
    #     grad_sum = 0        
    #
    #     for k in range(0, X.shape[0], bz):
    #
    #         end_id = k + bz
    #
    #         if end_id > X.shape[0]:
    #             end_id = X.shape[0]
    #
    #         batch_x = X[k:end_id]
    #
    #         batch_Y = Y[k:end_id]
    #
    #         batch_r_weight = curr_r_weights[k:end_id]
    #
    #         if isGPU:
    #             batch_x = batch_x.to(device)
    #             batch_Y = batch_Y.to(device)
    #             batch_r_weight = batch_r_weight.to(device)
    #         # model = model.cpu()
    #         #
    #         # X = X.cpu()
    #         #
    #         # Y = Y.cpu()
    #
    #     # curr_r_weights = curr_r_weights.cpu
    #     # if isGPU:
    #         # X = X.to(device)
    #         # Y = Y.to(device)
    #         # curr_r_weights = curr_r_weights.to(device)
    #     # print(next(model.parameters()).is_cuda)
    #     # print(X.device)
    #
    #         optimizer.zero_grad()
    #
    #         model_out = model(batch_x)
    #
    #         loss = torch.sum(loss_func(model_out, batch_Y).view(-1)*batch_r_weight.view(-1))/(end_id - k)
    #
    #         loss.backward()
    #
    #         model_grad = get_vectorized_grads(model)
    #
    #         grad_sum += model_grad*(end_id-k)
    #         # model_param = get_vectorized_params(model)
    #
    #
    #     model_grad = grad_sum/X.shape[0]
    #     model_param = get_vectorized_params(model)
    #
    #
    #
    #
    #
    #
    #     # optimizer.zero_grad()
    #     #
    #     # model_out = model(X)
    #     #
    #     # loss = torch.sum(loss_func(model_out, Y).view(-1)*curr_r_weights.view(-1))/X.shape[0]
    #     #
    #     # loss.backward()
    #     #
    #     # model_grad2 = get_vectorized_grads(model)
    #     #
    #     # print(torch.norm(model_grad - model_grad2))
    #     #
    #     # print('here')
    #     # if isGPU:
    #         # model = model.to(device)
    #         #
    # else:
    #
    #     probs = curr_r_weights.view(-1)/torch.sum(curr_r_weights)
        
        # if batch_ids is None:
        #     sampled_ids = torch.from_numpy(np.random.choice(X.shape[0], bz, p=probs.detach().numpy()))
        #
        #     batch_ids = sampled_ids
            
        
            #
        # else:
            # sampled_ids = batch_ids
        
        
    if batch_ids is None:
    
        random_ids = torch.randperm(X.shape[0])
        
        batch_ids = random_ids[0:bz]
    print('batch_ids::', batch_ids[0:10])
    sampled_ids = batch_ids
    
    batch_X = X[batch_ids]
    
    batch_Y = Y[batch_ids].type(torch.DoubleTensor)
    
    batch_r_weight = curr_r_weights[batch_ids]
    
    if isGPU:
        model = model.to(device)
        
        batch_X = batch_X.to(device)
        
        batch_Y = batch_Y.to(device)
        
        batch_r_weight = batch_r_weight.to(device)
    
    # print(list(model.parameters())[0].device, batch_X.device, batch_Y.device, batch_r_weight.device)
    
    model_out = model(batch_X)
    
    # print('curr model grad::', torch.norm(get_vectorized_grads(model)))
    
    loss = torch.mean(loss_func(model_out.view(batch_Y.shape), batch_Y).view(-1)*batch_r_weight.view(-1))
    
    loss_grad_prod = 0
    
    for k in range(len(list(model.parameters()))):
        curr_param = list(model.parameters())[k]
        
        grad_f, = torch.autograd.grad(loss, curr_param, create_graph=True)
        
        curr_res_vec = res_vec_list[k]
        
        if isGPU:
            curr_res_vec = curr_res_vec.to(device)
        
        
        loss_grad_prod += torch.dot(curr_res_vec.view(-1),grad_f.view(-1))
        
    
    loss_grad_prod.backward()
        # grad_list.append(grad_f)


    # loss.backward()
    
#     optimizer.step()

    model_grad = get_vectorized_grads(model, device)
    #
    # loss_grad = torch.dot()
    
    #
    model_param = get_all_vectorized_parameters1(list(model.parameters()), device)
    
    return model_grad, model_param, sampled_ids
    

def compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = False, bz = 1):
    devec_res = get_devectorized_parameters(res, full_shape_list, shape_list)
#     print('devectorized res::', devec_res)
    
    add_perturbation_on_model_weight(devec_res, model, origin_model_para_list)
    
    
#     curr_model_para_list = get_model_para_list(model)
    
#     print('current model parameters::', curr_model_para_list)
    
    
    
    curr_vec_grad, curr_vec_param ,_= compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device, random_sampling = random_sampling, bz = bz)
    
#     curr_vec_grad2 = compute_gradient2(model, X, Y, optimizer, curr_r_weights, removed_count)
    
#     print('grad diff::', torch.norm(curr_vec_grad - curr_vec_grad2))
#     
#     print('curr grad::', curr_vec_grad)
    
    return curr_vec_grad, curr_vec_param
    

def compute_grad_with_perturbed_weight_mini_batch(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = False, bz = 1, r=1e-5):
    
    origin_vec_grad, origin_vec_param, sampled_ids = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device, random_sampling = random_sampling, bz = bz)
    
    
    devec_res = get_devectorized_parameters(res, full_shape_list, shape_list)
#     print('devectorized res::', devec_res)
    
    add_perturbation_on_model_weight(devec_res, model, origin_model_para_list, device, r = r)
    
    # print(get_vectorized_params(model) - origin_model_para_list)
    
    compute_model_para_diff(list(model.parameters()), origin_model_para_list)
#     curr_model_para_list = get_model_para_list(model)
    
#     print('current model parameters::', curr_model_para_list)
    
    
    
    curr_vec_grad, curr_vec_param, _ = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device, random_sampling = random_sampling, bz = bz, batch_ids = sampled_ids)
    
    print('curr_grad, origin grad::', curr_vec_grad, origin_vec_grad, curr_vec_grad - origin_vec_grad, torch.norm(curr_vec_grad - origin_vec_grad))
    
    hessian_res_prod = (curr_vec_grad - origin_vec_grad)/r + optimizer.param_groups[0]['weight_decay']*res
    
    
#     curr_vec_grad2 = compute_gradient2(model, X, Y, optimizer, curr_r_weights, removed_count)
    
#     print('grad diff::', torch.norm(curr_vec_grad - curr_vec_grad2))
#     
#     print('curr grad::', curr_vec_grad)
    
    return hessian_res_prod, curr_vec_param

def compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = False, bz = 1, r=1e-5, sampled_ids = None):
    
    res_vec_list = get_devectorized_parameters(res, full_shape_list, shape_list)
    
    hessian_vec_prod, origin_vec_param, sampled_ids = compute_gradient3(model, X, Y, loss_func, curr_r_weights, res_vec_list, optimizer, is_GPU, device, random_sampling = random_sampling, bz = bz, batch_ids = sampled_ids)
    
    
#     devec_res = get_devectorized_parameters(res, full_shape_list, shape_list)
# #     print('devectorized res::', devec_res)
#
#     add_perturbation_on_model_weight(devec_res, model, origin_model_para_list, device, r = r)
#
#     # print(get_vectorized_params(model) - origin_model_para_list)
#
#     compute_model_para_diff(list(model.parameters()), origin_model_para_list)
# #     curr_model_para_list = get_model_para_list(model)
#
# #     print('current model parameters::', curr_model_para_list)
#
#
#
#     curr_vec_grad, curr_vec_param, _ = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device, random_sampling = random_sampling, bz = bz, batch_ids = sampled_ids)
#
#     print('curr_grad, origin grad::', curr_vec_grad, origin_vec_grad, curr_vec_grad - origin_vec_grad, torch.norm(curr_vec_grad - origin_vec_grad))
#
#     hessian_res_prod = (curr_vec_grad - origin_vec_grad)/r + optimizer.param_groups[0]['weight_decay']*res
    
    
#     curr_vec_grad2 = compute_gradient2(model, X, Y, optimizer, curr_r_weights, removed_count)
    
#     print('grad diff::', torch.norm(curr_vec_grad - curr_vec_grad2))
#     
#     print('curr grad::', curr_vec_grad)
    
    return hessian_vec_prod, origin_vec_param, sampled_ids


def set_model_parameters(model, param_list, device = None):
    
    i = 0
    
    for param in model.parameters():
        if device is not None:
            param.data = param_list[i].clone().to(device)
        else:
            param.data = param_list[i].clone()
        i += 1


def compute_hessian_inv(model, X, Y, r_weight, removed_count, optimizer):
    sigmoid_res = model(X)
    
    hessian1 = torch.mm(torch.t(X), X*sigmoid_res.view(-1,1)*(1-sigmoid_res.view(-1,1))*(r_weight.cpu().view(-1,1)))

    res = optimizer.param_groups[0]['weight_decay']*torch.eye(X.shape[1], dtype = torch.double, device = X.device) + hessian1/(X.shape[0] - removed_count)
    
    return res

'''loss_func:: loss not reduced, model in CPU'''
def compute_conjugate_grad(model, target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, is_GPU, device, exp_res = None, learning_rate = 1000, eps = 1e-8, random_sampling = False, bz = 100, running_iter = 200, regularization_rate = 0.0,r=1e-5):
    
        # res = torch.rand(target_vec.shape, device = target_vec.device, dtype = target_vec.dtype)
        
        res = torch.zeros_like(target_vec)
    
#         model_out = model(X)
#         
#         loss = torch.sum(loss_func(model_out, Y)*curr_r_weights)/(X.shape[0] - removed_count)
#         
#         optimizer.zero_grad()
#         
#         loss.backward()
#         
#         optimizer.step()
    
        origin_model_para_list = get_model_para_list(model)
    
#         print(origin_model_para_list)
    
        # origin_model_grad, origin_model_para2,_ = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device)
        
#         print(origin_model_grad)
        
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
        
#         lr = (X.shape[0] - removed_count)/10000
        lr = learning_rate
        
        print('learning rate::', lr)
        
        k = 0
#         for k in range(iters):
        while(True):
            
            '''delta f(w+r*res)'''
            # curr_vec_grad, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            
            '''~ hessian*res'''
            # hessian_res_prod2 = (curr_vec_grad - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*res
            
            set_model_parameters(model, origin_model_para_list, device)
            
            # hessian_res_prod, _ = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz, r= r)
            
            hessian_res_prod, _ = compute_grad_with_perturbed_weight_mini_batch(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz, r= r)
            
            set_model_parameters(model, origin_model_para_list, device)
            
            if exp_res is not None:
                expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
                
    #             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
                
                print('hessian vec prod diff::', torch.norm(hessian_res_prod - torch.mm(res.view(1,-1), expected_res)))
            
            # curr_vec_grad2, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            
            # hessian_diff_prod2 = (curr_vec_grad2 - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*(hessian_res_prod - target_vec)
            
            set_model_parameters(model, origin_model_para_list, device)
            
            delta_vec = hessian_res_prod - target_vec
            
            # hessian_diff_prod,_ = compute_grad_with_perturbed_weight_mini_batch(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz)
            
#             set_model_parameters(model, origin_model_para_list)
#             
#             expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
            
#             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
            # if exp_res is not None: 
            #
            print('hessian vec prod diff::', torch.norm(delta_vec))
            
#             if torch.norm(hessian_diff_prod) < 0.00001/(X.shape[0] - removed_count):
#                 break
            
            if torch.norm(delta_vec) < eps or k >= running_iter:# 0.00001/(X.shape[0] - removed_count):
                break
            
            res = res - lr*(delta_vec + regularization_rate*res)
            
            # if exp_res is not None:
                # print('hessian vector prod diff::', k, torch.norm(res - exp_res), torch.norm(hessian_diff_prod), torch.norm(-torch.mm(target_vec - torch.mm(res, expected_res), expected_res) - hessian_diff_prod), 0.0001/(X.shape[0] - removed_count))
                #
                # print(res, exp_res)
                #
            # else:
                # print('hessian vec prod diff::', k, torch.norm(hessian_diff_prod))
            
            k += 1
        
        set_model_parameters(model, origin_model_para_list, device)
        
        if exp_res is not None:
            print('hessian vector prod diff::', torch.norm(res - exp_res))
    
        return res
    


def compute_conjugate_grad3(model, target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, is_GPU, device, exp_res = None, learning_rate = 1000, eps = 1e-8, random_sampling = False, bz = 100, running_iter = 200, regularization_rate = 0.0,r=1e-5):
    
        # res = torch.rand(target_vec.shape, device = target_vec.device, dtype = target_vec.dtype)
        
        res = torch.zeros_like(target_vec)
    
#         model_out = model(X)
#         
#         loss = torch.sum(loss_func(model_out, Y)*curr_r_weights)/(X.shape[0] - removed_count)
#         
#         optimizer.zero_grad()
#         
#         loss.backward()
#         
#         optimizer.step()
    
        origin_model_para_list = get_model_para_list(model)
    
#         print(origin_model_para_list)
    
        # origin_model_grad, origin_model_para2,_ = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device)
        
#         print(origin_model_grad)
        
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
        
#         lr = (X.shape[0] - removed_count)/10000
        lr = learning_rate
        
        print('learning rate::', lr)
        
        k = 0
#         for k in range(iters):
        while(True):
            
            '''delta f(w+r*res)'''
            # curr_vec_grad, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            
            '''~ hessian*res'''
            # hessian_res_prod2 = (curr_vec_grad - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*res
            
            set_model_parameters(model, origin_model_para_list, device)
            
            # hessian_res_prod, _ = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz, r= r)
            
            hessian_res_prod, _, sampled_ids = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz, r= r)
            
            set_model_parameters(model, origin_model_para_list, device)
            
            if exp_res is not None:
                expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
                
    #             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
                
                print('hessian vec prod diff::', torch.norm(hessian_res_prod - torch.mm(res.view(1,-1), expected_res)))
            
            # curr_vec_grad2, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            
            # hessian_diff_prod2 = (curr_vec_grad2 - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*(hessian_res_prod - target_vec)
            
            set_model_parameters(model, origin_model_para_list, device)
            
            delta_vec = hessian_res_prod - target_vec
            
            # hessian_diff_prod2,_,_ = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz, sampled_ids = sampled_ids)
            
#             set_model_parameters(model, origin_model_para_list)
#             
#             expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
            
#             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
            # if exp_res is not None: 
            #
            print('hessian vec prod diff::', torch.norm(delta_vec))#, torch.norm(hessian_diff_prod2))
            
#             if torch.norm(hessian_diff_prod) < 0.00001/(X.shape[0] - removed_count):
#                 break
            
            if torch.norm(delta_vec) < eps or k >= running_iter:# 0.00001/(X.shape[0] - removed_count):
                break
            
            # res = res - lr*(delta_vec + regularization_rate*res)
            
            res = res - lr*(delta_vec + regularization_rate*res)
            
            # if exp_res is not None:
                # print('hessian vector prod diff::', k, torch.norm(res - exp_res), torch.norm(hessian_diff_prod), torch.norm(-torch.mm(target_vec - torch.mm(res, expected_res), expected_res) - hessian_diff_prod), 0.0001/(X.shape[0] - removed_count))
                #
                # print(res, exp_res)
                #
            # else:
                # print('hessian vec prod diff::', k, torch.norm(hessian_diff_prod))
            
            k += 1
        
        set_model_parameters(model, origin_model_para_list, device)
        
        if exp_res is not None:
            print('hessian vector prod diff::', torch.norm(res - exp_res))
    
        return res

def compute_conjugate_grad2(model, target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, is_GPU, device, exp_res = None, learning_rate = 1000, eps = 1e-8, random_sampling = False, bz = 100, running_iter = 200, regularization_rate = 0.0):
    
        # res = torch.rand(target_vec.shape, device = target_vec.device, dtype = target_vec.dtype)
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
        
        print(get_vectorized_params(model))
        
        print(torch.norm(get_vectorized_params(model)))
        
        print(target_vec)
        
        print(torch.norm(target_vec))
        
        origin_model_para_list = get_model_para_list(model)
        
        '''res:x, p_vec:p, residual: r'''
        
        # res = torch.zeros_like(target_vec)
        res = target_vec.clone()
        
        '''r = b - np.dot(A, x)'''
        
        set_model_parameters(model, origin_model_para_list, device)
            
        hessian_res_prod, _ = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz)
        
        residual = target_vec - hessian_res_prod
        
        '''p = r'''
        p_vec = residual
        
        
        ''' rsold = np.dot(np.transpose(r), r)'''
        rsold = torch.dot(residual.view(-1), residual.view(-1))
        
        for k in range(running_iter):
            
            '''Ap = np.dot(A, p)'''
            set_model_parameters(model, origin_model_para_list, device)
            
            hessian_res_prod, _ = compute_grad_with_perturbed_weight_mini_batch2(model, origin_model_para_list, p_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz)

            '''alpha = rsold / np.dot(np.transpose(p), Ap)'''
            alpha = rsold / torch.dot(p_vec.view(-1), hessian_res_prod.view(-1))
            
            '''x = x + np.dot(alpha, p)'''
            res = res + alpha*p_vec#torch.dot(alpha, p_vec.view(-1))
            
            '''r = r - np.dot(alpha, Ap)'''
            residual = residual - alpha*hessian_res_prod#torch.dot(alpha, hessian_res_prod.view(-1))
            
            '''rsnew = np.dot(np.transpose(r), r)'''
            rsnew = torch.dot(residual.view(-1), residual.view(-1))
            
            print('rsnew, diff:: ', rsnew, residual, torch.norm(hessian_res_prod - target_vec))
            
            if torch.sqrt(rsnew) < eps:
                break
            
            '''p = r + (rsnew/rsold)*p'''
            p_vec = residual + (rsnew/rsold)*p_vec
            rsold = rsnew
        
#         model_out = model(X)
#         
#         loss = torch.sum(loss_func(model_out, Y)*curr_r_weights)/(X.shape[0] - removed_count)
#         
#         optimizer.zero_grad()
#         
#         loss.backward()
#         
#         optimizer.step()
    
        
    
#         print(origin_model_para_list)
    
        # origin_model_grad, origin_model_para2,_ = compute_gradient(model, X, Y, loss_func, curr_r_weights, removed_count, optimizer, is_GPU, device)
        
#         print(origin_model_grad)
        
        
#         lr = (X.shape[0] - removed_count)/10000
        # lr = learning_rate
        #
        # print('learning rate::', lr)
        #
        # k = 0
#         for k in range(iters):
        # while(True):
        #
            # '''delta f(w+r*res)'''
            # # curr_vec_grad, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, res, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            #
            # '''~ hessian*res'''
            # # hessian_res_prod2 = (curr_vec_grad - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*res
            #
            #
            # set_model_parameters(model, origin_model_para_list, device)
            #
            # if exp_res is not None:
                # expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
                #
    # #             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
    #
                # print('hessian vec prod diff::', torch.norm(hessian_res_prod - torch.mm(res.view(1,-1), expected_res)))
                #
            # # curr_vec_grad2, curr_vec_param = compute_grad_with_perturbed_weight_single_step(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device)
            #
            # # hessian_diff_prod2 = (curr_vec_grad2 - origin_model_grad)/r + optimizer.param_groups[0]['weight_decay']*(hessian_res_prod - target_vec)
            #
            # set_model_parameters(model, origin_model_para_list, device)
            #
            # delta_vec = hessian_res_prod - target_vec
            #
            # # hessian_diff_prod,_ = compute_grad_with_perturbed_weight_mini_batch(model, origin_model_para_list, hessian_res_prod - target_vec, X, Y, loss_func, optimizer, curr_r_weights, removed_count, full_shape_list, shape_list, total_shape_size, is_GPU, device, random_sampling = random_sampling, bz = bz)
            #
# #             set_model_parameters(model, origin_model_para_list)
# #             
# #             expected_res = compute_hessian_inv(model, X, Y, curr_r_weights, removed_count, optimizer)
            #
# #             print(torch.mm(res.view(1,-1), expected_res), hessian_res_prod)
            # # if exp_res is not None: 
            # #
            # print('hessian vec prod diff::', torch.norm(delta_vec))
            #
# #             if torch.norm(hessian_diff_prod) < 0.00001/(X.shape[0] - removed_count):
# #                 break
            #
            # if torch.norm(delta_vec) < eps or k >= running_iter:# 0.00001/(X.shape[0] - removed_count):
                # break
                #
            # res = res - lr*(delta_vec + regularization_rate*res)
            #
            # # if exp_res is not None:
                # # print('hessian vector prod diff::', k, torch.norm(res - exp_res), torch.norm(hessian_diff_prod), torch.norm(-torch.mm(target_vec - torch.mm(res, expected_res), expected_res) - hessian_diff_prod), 0.0001/(X.shape[0] - removed_count))
                # #
                # # print(res, exp_res)
                # #
            # # else:
                # # print('hessian vec prod diff::', k, torch.norm(hessian_diff_prod))
                #
            # k += 1
        
        set_model_parameters(model, origin_model_para_list, device)
        
        if exp_res is not None:
            print('hessian vector prod diff::', torch.norm(res - exp_res))
    
        return res

def create_random_id_multi_super_iters(num, epochs):
    
    random_ids_multi_super_iterations = []
    
    for i in range(epochs): 
        random_ids_multi_super_iterations.append(torch.randperm(num))
        
    return random_ids_multi_super_iterations


def frozen_model_para(model):
    for param in model.parameters():
        param.requires_grad = False


def is_small_model(total_shape_size):
    return total_shape_size < 1000

def update_optimizer(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def get_grad_all_samples(model, X, Y, optimizer, total_shape_size, theta, regularization_coeff):
    
    term1 = torch.zeros([X.shape[0], total_shape_size], dtype = X.dtype, device = X.device)
    
    for i in range(X.shape[0]):
        optimizer.zero_grad()
        
        model_out = model(X[i].view(1,-1))
        
        full_loss = model.get_loss_function()(model_out.view(Y[i].shape), Y[i].type(torch.DoubleTensor))
        
        full_loss.backward()
        
        term1[i] = regularization_coeff*theta.view(1,-1) + get_vectorized_grads(model)
        
    return term1

def compute_partial_gradient_x_delete(total_shape_size, X, Y, model, origin_model, theta, origin_theta, regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec, is_GPU, device, optimizer, lower_bound = None, upper_bound = None, sample_id=0, hessian_lr = 1000, hessian_eps = 1e-8, hv_random_sampling = False, hv_bz = 1):
#      = X + delta_X

        frozen_model_para(origin_model)
    
        total_count = X.shape[0]
    
        update_X = X.clone()
        
        optimizer.zero_grad()
        
        model_out = model(update_X)
        
        full_loss = model.get_loss_function()(model_out.view(Y.shape), Y.type(torch.DoubleTensor))
#         model.fc1.zero_grad()
#         
        full_loss.backward()
#     
#         expected_loss = model.fc1.weight.grad.clone()
#     
#         model.fc1.zero_grad()
#         
#         if model.fc1.bias is not None:
#             update_X = torch.cat([update_X, torch.ones([update_X.shape[0], 1], dtype = update_X.dtype, device = update_X.device)], dim = 1)
    
#         '''n*1'''
#         sigmoid_res = F.sigmoid(X_Y_theta_prod)
        
        
#         term1 = regularization_coeff*theta.view(1,-1) - Y.view(-1,1)*update_X*(1-sigmoid_res) 
        '''cpu'''
        
#         term1 = regularization_coeff*theta.view(1,-1) + get_vectorized_grads(model)
        term1 = get_grad_all_samples(model, X, Y, optimizer, total_shape_size, theta, regularization_coeff)
        if isinstance(model, Binary_Logistic_regression):
            term1_1 = regularization_coeff*theta.view(1,-1) - (Y.view(-1,1) - model_out.view(-1,1))*update_X
            print(torch.norm(term1 - term1_1))
            
            
        full_grad = torch.zeros_like(r_weight)
    #         diff = expected_loss.view(1,-1) + regularization_coeff*theta.view(1,-1) - torch.sum(term1, dim=0)/X.shape[0]
    #         
    #         print('diff::', torch.norm(diff))
#         if model.is_small_model():
        if isinstance(model, Binary_Logistic_regression) and is_small_model(total_shape_size):
            '''cpu'''
            hessian_res, _ = model.loss_function_hessian_with_weight(update_X, Y, model_out, regularization_coeff, r_weight, removed_count, model)
            
    #         X_Y_theta_prod = Y.view(-1,1)*torch.mm(update_X, theta)
            
    #         X_Y_theta_prod = Y.view(-1,1)*model(update_X)
    #
            '''cpu'''
            hessian_inv = torch.inverse(hessian_res)
            
            
    #         
    #         print(diff)
            
            
    #         '''n*m'''
    #         term1_X = update_X*(sigmoid_res*(sigmoid_res - 1))
    #         
    #         '''n*1'''
    #         
    #         term2_X = (Y.view(-1,1)*(1-sigmoid_res))
            
    #         res_term1 = term1[0:X.shape[0]-1] - term1[X.shape[0]-1:] 
            
    #         '''n*m'''
    #         term1 = X*(sigmoid_res*(sigmoid_res - 1))
    #         
    #         '''n*1'''
    #         
    #         term2 = (Y.view(-1,1)*(1-sigmoid_res))
            
            
    #         full_grad_X = torch.zeros_like(X)
    #         full_grad2 = torch.zeros_like(X)
            
        #     term2 = 
            
            
    #             total_count = global_count
    #         
    #         
    #         for k in range(total_count):
    #             
    #             print('compute gradient::', k)
    #             
    #             curr_term1 = torch.mm(sub_hessian_inv.view(1,-1), torch.mm(term1[k].view(-1,1), theta.view(1,-1)))/X.shape[0]
    #              
    #     #         curr_term1_extra = torch.mm(sub_hessian_inv, torch.abs(torch.mm(term1_extra[i].view(-1,1), theta.view(1,-1))))
    #              
    #     #         print(curr_term1.shape)
    #              
    #             curr_term2 = sub_hessian_inv.view(1,-1)*term2[k]/X.shape[0]
    #     
    #     #         print(curr_term1.shape, curr_term2.shape)
    #             curr_grad = curr_term1 + curr_term2
    #             
    #             full_grad2[k] = curr_grad
            
        
    #         gap = 1000
        
            
    #             theta_copy = theta.view(1, 1, -1).repeat(gap, 1, 1)
            
            '''gap, 1, m'''
    #             if is_l2_norm:
    #             sub_hessian_inv_copy = sub_hessian_inv.view(1,1,-1).repeat(gap, 1, 1)
    #             else:
            '''cpu'''
            sub_hessian_inv_copy = hessian_inv.view(1,hessian_inv.shape[0],hessian_inv.shape[1]).repeat(gap, 1, 1)
        
        if (not norm == 'pos') and (not norm == 'neg') and (not norm == 'loss'):
            theta_grad = get_derivative_norm(theta.view(-1) - origin_theta.view(-1), norm)
            
    #             if norm == 'l2':
            theta_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
        else:
            
            theta_copy = theta.detach().clone()
            
            theta_copy.requires_grad = True
            
            origin_theta_copy = origin_theta.detach().clone() 
            
            origin_theta_copy.requires_grad = False
            
            if norm == 'pos':
                loss = torch.dot(origin_theta_copy.view(-1), theta_copy.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta_copy.view(-1)))
            else:
                if norm == 'neg':
                    loss = -torch.dot(origin_theta_copy.view(-1), theta_copy.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta_copy.view(-1)))
                else:
                    loss_func = model.get_loss_function(reduction='none')

                    optimizer.zero_grad()
#                     if model.fc1.weight.grad is not None:
#                         model.fc1.weight.grad.zero_()
#                     
#                     if model.fc1.bias is not None and model.fc1.bias.grad is not None:
#                         model.fc1.bias.grad.zero_()

#                     origin_model.fc1.requires_grad = False
                    
                    individual_loss_terms = loss_func(model(X).view(Y.shape), Y.type(torch.DoubleTensor)).clone()
                    
                    origin_individual_loss_terms = loss_func(origin_model(X).view(Y.shape), Y.type(torch.DoubleTensor)).clone()
                    
                    r_weight_cpu = r_weight.cpu()
                    
                    print('model_parameter::', get_vectorized_params(model), get_vectorized_params(origin_model))
                    
                    if norm == 'loss':
                        
                        loss1 = torch.sum(individual_loss_terms.view(-1)*r_weight_cpu.view(-1))/(X.shape[0] - removed_count)
                        
#                         print('origin model parameters:', origin_model.fc1.weight, origin_model.fc1.bias)
                        
                        loss2 = torch.mean(loss_func(origin_model(X).view(Y.shape), Y.type(torch.DoubleTensor))*torch.ones_like(r_weight_cpu))
                    else:
                        loss1 = torch.sum(individual_loss_terms.view(-1)*(1-r_weight_cpu).view(-1))/torch.sum(1-r_weight_cpu)
                         
                        loss2 = torch.sum(origin_individual_loss_terms.view(-1)*(1-r_weight_cpu).view(-1))/torch.sum(1-r_weight_cpu)
                    
#                     selected_X, selected_Y = get_selected_data(r_weight, X, Y, removed_count)
#                     
#                     loss2 = torch.mean(loss_func(model(selected_X), selected_Y.type(torch.DoubleTensor)).view(-1))
#                     
#                     loss1 = torch.mean(loss_func(origin_model(selected_X), selected_Y.type(torch.DoubleTensor)))
                    
                    
                    print('loss1::', loss1)
                    
                    print('loss2::', loss2)
                    
                    loss = loss1# - loss2
                    
                    
                    
                    
#                     loss = 
            
            print('loss::', loss)
#             loss = torch.sum(torch.argmax(torch.mm(update_X, theta), 1) == torch.argmax(torch.mm(X, origin_theta_copy), 1))
            
#             loss = torch.sum(F.gumbel_softmax(torch.mm(update_X, theta), hard = True) == F.gumbel_softmax(torch.mm(X, origin_theta_copy), hard = True))
            
#             loss = -torch.sum(torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy)))
            
#             loss = -torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy))
            
            if not norm == 'loss':
                if theta.grad is not None:
                    theta.grad.zero_()
                
                loss.backward()
                
                theta_grad = theta_copy.grad.clone()
                
                theta_grad_copy = -theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
                
                
                del theta_copy
            else:
#                 model.fc1.weight.grad.zero_()
                
                loss.backward()
                
                theta_grad = get_vectorized_grads(model)
                
                theta_grad_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
                
#             theta_copy.grad.zero_()
#             
#             theta_copy.requires_grad = False
#             else:
#                 if norm == ''
        
        '''cpu'''
                
#         if self.is_small_model():
        if isinstance(model, Binary_Logistic_regression) and is_small_model(total_shape_size):
            sub_hessian_inv_theta_prod = torch.bmm(theta_grad_copy, sub_hessian_inv_copy)
        
        else:
            hessian_inv_theta_prod2 = compute_conjugate_grad(model, theta_grad, X, Y, loss_func, optimizer, r_weight, removed_count, is_GPU, device, learning_rate = hessian_lr, eps = hessian_eps, random_sampling = hv_random_sampling, bz = hv_bz)
            
            sub_hessian_inv_theta_prod = hessian_inv_theta_prod2.view(1, 1, -1).repeat(gap, 1,1)
        
#         print(torch.norm(sub_hessian_inv_theta_prod[0] - sub_hessian_inv_theta_prod2[0])) 
        
        
        
        with torch.no_grad():
            for k in range(0, total_count, gap):
                
                end_id = k+gap
                if end_id >= total_count:
                    end_id = total_count
    
                batch_term1 = term1[k:end_id]
                
                if is_GPU:
                    batch_term1 = batch_term1.to(device)
                
#                 batch_term1_X = term1_X[k:end_id]
#                 
#                 batch_term2_X = term2_X[k:end_id]
                
                print('compute gradient::', k, end_id)
                
                '''gap, 1'''
                
#                 batch_term2 = term2[k: end_id]
                '''gap*m*1'''
#                 print(sub_hessian_inv_copy[0: end_id - k].shape, batch_term1.shape, end_id)
                '''cpu'''
                curr_hessian_inv_theta_prod = sub_hessian_inv_theta_prod[0: end_id - k]
                
                if is_GPU:
                    curr_hessian_inv_theta_prod = curr_hessian_inv_theta_prod.to(device)
                    
#                 print(curr_hessian_inv_theta_prod.device, batch_term1.device)
                curr_res_term1 = -torch.bmm(curr_hessian_inv_theta_prod, batch_term1.view(end_id - k, -1,1))/(X.shape[0]-removed_count)
                
#                 curr_term2 = sub_hessian_inv_copy[0: end_id - k]*batch_term2.view(end_id - k, 1, 1)/X.shape[0]
                
#                 curr_grad = (curr_term1 + curr_term2).view(end_id - k, -1)
                '''- from the max problem'''
#                 curr_res_term1 = -torch.bmm(theta_copy[0: end_id - k], curr_res_term1)
                
#                 print(res_term1.shape, full_grad[k:end_id].shape)
#                 print(full_grad.device, curr_res_term1.device, mu_vec.device, lambda_vec.device, gamma_vec.device)
                full_grad[k:end_id] = curr_res_term1.view(-1) + (mu_vec[k:end_id].view(-1) - lambda_vec[k:end_id].view(-1) - gamma_vec)/(X.shape[0] - removed_count)
        
#                 curr_term1_X = torch.bmm(sub_hessian_inv_theta_prod[0: end_id - k], torch.bmm(batch_term1_X.view(end_id - k, -1,1), theta_copy[0: end_id - k]))/X.shape[0]
#                 
#                 curr_term2_X = sub_hessian_inv_theta_prod[0: end_id - k]*batch_term2_X.view(end_id - k, 1, 1)/X.shape[0]
#                 
#                 curr_grad = (curr_term1_X + curr_term2_X).view(end_id - k, -1)
                del batch_term1, curr_hessian_inv_theta_prod
#                 full_grad_X[k:end_id] = curr_grad
    #         print(torch.norm(full_grad2 - full_grad))
        if not (norm == 'loss' or norm == 'rloss'):
            update_r_weight = r_weight - learning_rate*full_grad
        else:
            
            if norm == 'rloss':
                update_r_weight = r_weight - learning_rate*(full_grad - (individual_loss_terms + origin_individual_loss_terms).detach().view(full_grad.shape))
            
            else:
                if is_GPU:
                    individual_loss_terms = individual_loss_terms.to(device)
                
                update_r_weight = r_weight - learning_rate*(full_grad + ((individual_loss_terms)/((X.shape[0] - removed_count))).detach().view(full_grad.shape))
        
                del individual_loss_terms
        
#         if lower_bound is None:
#             
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
#             
#             update_X = X+update_delta_X
#         else:
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_X[update_X > upper_bound] = upper_bound[update_X > upper_bound]
#             
#             update_X[update_X < lower_bound] = lower_bound[update_X < lower_bound]
            
#             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
            
#             update_X = X+update_delta_X
#         update_r_weight[-1] = X.shape[0] - removed_count - torch.sum(update_r_weight[0:-1])
#         update_delta_r_weight = torch.clamp(r_weight - update_r_weight, 0, 1)
#         
#         update_r_weight = r_weight-update_delta_r_weight

        
        
        
        sorted_weight,sorted_ids = torch.sort(update_r_weight, descending=True)
        
        removed_ids = sorted_ids.view(-1)[X.shape[0]-removed_count:]
        
        remaining_ids = sorted_ids.view(-1)[0:X.shape[0]-removed_count] 
        
        integer_r_weight = torch.zeros_like(r_weight)
        
        integer_r_weight[remaining_ids] = 1
        
#         removed_ids = torch.nonzero(update_r_weight < ((X.shape[0] - removed_count)/X.shape[0])) 
        
        
        
        
        return integer_r_weight, removed_ids, update_r_weight, mu_vec, lambda_vec, gamma_vec, update_X, sorted_weight











def optimize_two_steps_delete_perturb(model, dataset_train, origin_model, regularization_coeff, out_epoch_count, inner_epoch_count, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, learning_rate, batch_size, removed_count, training_learning_rate, wd, is_GPU, device, gap, norm, lower_bound = None, upper_bound = None, epsilon=0, loss_func = None, hessian_lr = 1000, hessian_eps = 1e-8, hv_random_sampling = False, hv_bz = 1):
#     delta_X = torch.zeros_like(X)

    optimizer = model.get_optimizer(training_learning_rate, wd)

    if random_ids_multi_super_iterations is None:
        random_ids_multi_super_iterations = create_random_id_multi_super_iters(dataset_train.data.shape[0], inner_epoch_count)

#     origin_weight = origin_model.fc1.weight.detach()
#     
#     origin_bias = None
#     
#     if origin_model.fc1.bias is not None:
#         origin_bias = origin_model.fc1.bias.detach()
#     print('origin model parameter::', origin_weight, origin_bias)

    origin_param = get_vectorized_params(origin_model)
    
    decay_threshold = 0.01
    
    if is_GPU:
        origin_param = origin_param.to(device)
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    origin_param_list = get_devectorized_parameters(origin_param, full_shape_list, shape_list)
    

    X = dataset_train.data
    
    Y = dataset_train.labels

#         if is_GPU:
#             X = X.to(device)
#              
#             Y = Y.to(device)
        
#             origin_weight = origin_weight.to(device)
#             
#             origin_bias = 
    
    
    removed_ids = torch.randperm(X.shape[0])[0:X.shape[0] - removed_count]
    
    '''GPU'''
    r_weight = torch.rand(X.shape[0], device= origin_param.device, dtype = X.dtype)
    
#     r_weight[:] = 0.5#(X.shape[0] - removed_count)/X.shape[0]
#         
#         
#         r_weight[:]=1
#         r_weight[1000:1060]=0
#         self.set_parameters(origin_weight, origin_bias)
#         self.train_model_with_weight(optimizer, random_ids_multi_super_iterations, X.type(torch.DoubleTensor), Y.type(torch.DoubleTensor), batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, r_weight, removed_count, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias))
    
    
    print('r_weight count::', torch.sum(r_weight))
#         update_X = X + delta_X
    set_model_parameters(model, origin_param_list)
    
    prev_r_weight = r_weight.clone()
    
    '''GPU'''
    mu_vec = torch.rand(X.shape[0],device = origin_param.device, dtype = X.dtype)
    
    '''GPU'''
    lambda_vec = torch.rand(X.shape[0],device = origin_param.device, dtype = X.dtype)
    
    '''GPU'''
    gamma_vec = torch.rand(1,device = origin_param.device, dtype = X.dtype)
    
    d_epoch_count = 1
    
    learning_rate2 = 0.0001
    
#         learning_rate3 = 0.0005
    
    warmup_epochs = 100
    
    upper_bound_tensor = None
    
    lower_bound_tensor = None
    
    delta_X = torch.rand((X.shape[0], X.shape[1]), device = X.device, dtype = X.dtype)
    
    if lower_bound is None:
        delta_X = (delta_X*2 - 1)*epsilon
    else:
        
        upper_bound_tensor = dataset_train.upper_bound
        
        lower_bound_tensor = dataset_train.lower_bound
        
        if is_GPU:
            upper_bound_tensor = upper_bound_tensor.to(device)
        
            lower_bound_tensor = lower_bound_tensor.to(device)
            
        upper_eps = (upper_bound_tensor - X)*delta_X

        lower_eps = (X - lower_bound_tensor)*delta_X
        
        delta_X[delta_X > 0] = upper_eps[delta_X > 0]
        
        delta_X[delta_X <= 0] = lower_eps[delta_X <= 0] 
    
    
    delta_w = 0
    
#         update_X = X + delta_X
    
    X = X.type(torch.DoubleTensor)
    
#         update_X = update_X.type(torch.DoubleTensor)
    
    for o_epoch in range(out_epoch_count):
        
        print('outer loop epoch::', o_epoch)
        for d_epoch in range(d_epoch_count):
            
        
            print('dual update epoch::', d_epoch)
    #         lr.theta.requires_grad = False
#                 with torch.no_grad():
                
            '''X, update_X, Y, theta, regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec'''
            
            if o_epoch >= 100:
                print('here')
            
            
            if is_GPU:
                model = model.to('cpu')
                origin_model = origin_model.to('cpu')
            
            updated_r_weight, removed_ids, origin_update_r_weight, mu_vec, lambda_vec, gamma_vec, new_update_X, sorted_weight = compute_partial_gradient_x_delete(total_shape_size, X, Y, model, origin_model, get_vectorized_params(model), get_vectorized_params(origin_model), regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec, is_GPU, device, optimizer, lower_bound_tensor, upper_bound_tensor, hessian_lr = hessian_lr, hessian_eps = hessian_eps, hv_random_sampling = hv_random_sampling, hv_bz = hv_bz)
        
            if is_GPU:
                model = model.to(device)
                origin_model = origin_model.to(device)
            
            print('removed_ids count::', removed_ids.view(-1)[0:100])
            
            print('removed_ids weight::', sorted_weight.view(-1)[0:100])
            
#                 print('X difference', torch.max(new_update_X - update_X), torch.min(new_update_X - update_X), torch.max(new_update_X - X), torch.min(new_update_X - X))
            
            print(torch.max(origin_update_r_weight), torch.min(origin_update_r_weight))
            
            print('r weight update::', torch.norm(origin_update_r_weight.type(torch.double) - r_weight), torch.norm(updated_r_weight.type(torch.double) - prev_r_weight.type(torch.double)))

#                 update_X = new_update_X
#             print('X difference', torch.max(new_update_X - update_X), torch.min(new_update_X - update_X), torch.max(new_update_X - X), torch.min(new_update_X - X))
    
    #         
        
#                 self.set_parameters(origin_weight, origin_bias)
            
#                 self.lr.theta.requires_grad = True
#             update_theta = self.model_update_standard_lib_logistic_regression(inner_epoch_count, dataset_train, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, removed_ids.view(-1), batch_size, training_learning_rate, is_GPU, device, regularization_coeff)
#                 '''random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, learning_rate, optimizer, criterion, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count'''
#                 self.train_model_with_weight(random_ids_multi_super_iterations, update_X, Y, batch_size, training_learning_rate, None, None, inner_epoch_count, is_GPU, device, regularization_coeff, origin_update_r_weight, removed_count)
        
            '''optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count'''
            set_model_parameters(model, origin_param_list)

            update_optimizer(optimizer, training_learning_rate)
             
            origin_update_r_weight_new = origin_update_r_weight.clamp(min=0, max=1)
#                 origin_update_r_weight_new = origin_update_r_weight
             
            model.train_model_with_weight(optimizer, random_ids_multi_super_iterations, X, Y, batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, origin_update_r_weight_new, removed_count, get_vectorized_params(model), loss_func = loss_func)
         
            print('model parameter::', get_vectorized_params(model))
         
            print('model param diff::', torch.norm(get_vectorized_params(model) - origin_param))
#             self.compute_model_para_diff(norm,origin_weight, origin_bias)
#                 update_theta = update_theta/torch.norm(update_theta)
#             update_X = new_update_X

            prev_r_weight = updated_r_weight

            r_weight = origin_update_r_weight_new
            
#                 self.set_parameters(update_theta.detach().clone())
#                 self.compute_model_para_diff(norm,origin_weight, origin_bias)

        
        mu_vec = mu_vec + learning_rate*(origin_update_r_weight.view(-1) - 1)/(X.shape[0] - removed_count)
    
        lambda_vec = lambda_vec + learning_rate*(-origin_update_r_weight.view(-1))/(X.shape[0] - removed_count)
        
        gamma_vec = gamma_vec + learning_rate*(X.shape[0] - removed_count - torch.sum(origin_update_r_weight))/(X.shape[0] - removed_count)
        
        mu_vec[mu_vec < 0] = 0
        
        lambda_vec[lambda_vec < 0] = 0
        
        print('learning rate::', learning_rate)
        
        print('mu vec::', torch.max(mu_vec), torch.min(mu_vec))
            
        print('lambda vec::', torch.max(lambda_vec), torch.min(lambda_vec))
        
        print('gamma::', gamma_vec)
        
        print('r_weight::', torch.max(origin_update_r_weight), torch.min(origin_update_r_weight))
        
        if torch.abs(X.shape[0] - removed_count - torch.sum(origin_update_r_weight)).item() < stop_number and torch.abs(torch.max(origin_update_r_weight) - 1) <= 0.001 and torch.abs(torch.min(origin_update_r_weight)) <= 0.001:
            break
        if torch.abs(X.shape[0] - removed_count - torch.sum(origin_update_r_weight)).item() < stop_number and torch.abs(torch.max(origin_update_r_weight) - 1) <= decay_threshold and torch.abs(torch.min(origin_update_r_weight)) <= decay_threshold:
            learning_rate = learning_rate/2
            decay_threshold = decay_threshold/2
        
#             if torch.abs(X.shape[0] - removed_count - torch.sum(r_weight)).item() < 5:
#                 learning_rate = learning_rate/3*2
#                 if learning_rate < 0.2:
#                     learning_rate = 0.2
        
#             self.set_parameters(origin_weight, origin_bias)
#                 
# #             self.lr.theta.requires_grad = True
#             
#             
#             self.train_model_with_weight(optimizer, random_ids_multi_super_iterations, update_X, Y, batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, updated_r_weight, removed_count, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias))
# 
#             print('model parameter::', self.fc1.weight, self.fc1.bias)
#             
# #             final_update_theta = self.train_model_with_weight(random_ids_multi_super_iterations, update_X, Y, batch_size, training_learning_rate, None, None, inner_epoch_count, is_GPU, device, regularization_coeff, updated_r_weight, removed_count)
#             
# #             final_update_theta = final_update_theta/torch.norm(final_update_theta)
#             
#             self.compute_model_para_diff(norm, origin_weight, origin_bias)
        
        
#             if norm == '2':
#                 print('theta difference::', torch.norm(final_update_theta - origin_weight))
#             else:
#                 if norm == 'inf':
#                     print('theta difference::', torch.norm(final_update_theta - origin_weight, float('inf')))
#             
#             
#                 else:
#                     if norm == 'pos':
#                     
#                         curr_delta_w = torch.dot(final_update_theta.view(-1), origin_weight.view(-1))/(torch.norm(final_update_theta.view(-1)) * torch.norm(origin_theta.view(-1)))
#                     else:
#                         curr_delta_w = -torch.dot(final_update_theta.view(-1), origin_weight.view(-1))/(torch.norm(final_update_theta.view(-1)) * torch.norm(origin_theta.view(-1)))
#         
#                     print('theta difference::', curr_delta_w)
        
        print('here')
        
    return delta_w, removed_ids
    