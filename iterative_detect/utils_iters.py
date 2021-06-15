'''
Created on Jan 6, 2021

'''

import os, sys

import torch
import torch.functional as F
from collections import deque  

import nvgpu


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/real_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Reweight_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
from sklearn.metrics import roc_auc_score
from sklearn import metrics

try:
    from models.util_func import *
    from utils.utils import *
    from real_examples.utils_real import *
    from models.util_func import *
    from Reweight_examples.utils_reweight import *
    from Reweight_examples.generate_prob_labels import *
except ImportError:
    from util_func import *
    from utils import *
    from utils_real import *
    from util_func import *
    from utils_reweight import *
    from generate_prob_labels import *

import models

label_change_threshold = 1e-5

small_constant = 1e-6


incremental_suffix = '_incre'

def get_entry_grad_with_labels(updated_batch_y, curr_entry_grad):
    return -(torch.mm(updated_batch_y.view(1,-1),curr_entry_grad.view(curr_entry_grad.shape[0]*curr_entry_grad.shape[1], -1))/updated_batch_y.shape[0]).view(-1)

def check_existing_files(file):
    return os.path.exists(file)

def store_iteration_count(iter_count, full_out_dir):
    torch.save(iter_count, full_out_dir + '/sample_wise_grad_iter')

def compare_iteration_count(curr_iter_count, full_out_dir):
    prev_iter_count = torch.load(full_out_dir + '/sample_wise_grad_iter')
    
    return curr_iter_count - 1 > (prev_iter_count + 1)

def check_iteration_count_file_existence(full_out_dir):    
    return check_existing_files(full_out_dir + '/sample_wise_grad_iter')

def remove_iteration_count_file(full_out_dir):
    if check_iteration_count_file_existence(full_out_dir):
        os.remove(full_out_dir + '/sample_wise_grad_iter')

def load_iteration_count(full_out_dir):
    iter_count = torch.load(full_out_dir + '/sample_wise_grad_iter')
    return iter_count

def load_sample_wise_grad_list(full_out_dir, iter_count):
    prev_sample_wise_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count))
    
    curr_sample_wise_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count + 1))
    
    return prev_sample_wise_grad_list, curr_sample_wise_grad_list
    


def remove_different_version_dataset(full_out_dir, model_name):
    
    data_version_file = full_out_dir + '/full_training_noisy_dataset_v*'
    
    influence_ids_files = full_out_dir + '/' + model_name + '_influence_removed_ids_v*'
    
    influence_weights_files = full_out_dir + '/' + model_name + '_influence_removed_ids_weight_v*'
    
    model_files = full_out_dir + '/model_v*'
    
    w_list_files = full_out_dir + '/w_list_v*'
    
    grad_list_files = full_out_dir + '/grad_list_v*'
    
    existing_labeled_id_files = full_out_dir + '/full_existing_labeled_id_tensor_v*'
    
    influence_files = full_out_dir + '/' + model_name + '_influence_v*'
    
    file_head_list = [influence_files, data_version_file, influence_ids_files, influence_weights_files, model_files, w_list_files, grad_list_files, existing_labeled_id_files]
    
    for file_head in file_head_list:
        for filename in glob.glob(file_head):
            os.remove(filename)
    if os.path.exists(full_out_dir + '/sample_wise_grad_iter'):
        os.remove(full_out_dir + '/sample_wise_grad_iter')
    if os.path.exists(full_out_dir + '/o2u_random_ids_multi_epochs'):
        os.remove(full_out_dir + '/o2u_random_ids_multi_epochs')
    
#     for filename in glob.glob(influence_ids_files):
#         os.remove(filename)
# 
#     for filename in glob.glob(influence_weights_files):
#         os.remove(filename)
# 
#     for filename in glob.glob(model_files):
#         os.remove(filename)


def compute_single_max_mu(init_w, iterative_times, model, loss_func, optimizer, is_GPU, device, curr_X, curr_Y, full_shape_list, shape_list, total_shape_size, model_param, model_grad):
    
    vectorized_w = init_w.view(1,-1)#get_all_vectorized_parameters1(, device)
    
    vectorized_model_param = get_all_vectorized_parameters1(model_param, device)
    
#     t1 = time.time()
    
    for k in range(iterative_times):
#         
        
        curr_w_after_perturb = vectorized_model_param + small_constant* vectorized_w
        
        set_model_parameters(model, get_devectorized_parameters(curr_w_after_perturb, full_shape_list, shape_list), device)
        
        curr_loss = loss_func(model(curr_X), curr_Y)
        
        optimizer.zero_grad()
        
        curr_loss.backward()
        
        curr_grad = get_vectorized_grads(model, device)
        
        hessian_w_prod = (curr_grad - model_grad)/(small_constant)
        
        updated_vectorized_w = hessian_w_prod/torch.norm(hessian_w_prod)
        
        print(torch.norm(updated_vectorized_w - vectorized_w))
        
        if torch.norm(updated_vectorized_w - vectorized_w) < 1e-6:
            vectorized_w = updated_vectorized_w
            break
        
        
        vectorized_w = updated_vectorized_w
    
    curr_w_after_perturb = vectorized_model_param + small_constant* vectorized_w
        
    set_model_parameters(model, get_devectorized_parameters(curr_w_after_perturb, full_shape_list, shape_list), device)
    
    curr_loss = loss_func(model(curr_X), curr_Y)
    
    optimizer.zero_grad()
    
    curr_loss.backward()
    
    curr_grad = get_vectorized_grads(model, device)
    
    hessian_w_prod = (curr_grad - model_grad)/(small_constant)
    
    
    max_u = torch.dot(hessian_w_prod.view(-1), vectorized_w.view(-1))/torch.dot(vectorized_w.view(-1), vectorized_w.view(-1))
    
#     t2 = time.time()
#     
#     print('time::', t2 - t1)
    
    return max_u  
        
def compute_single_max_mu_class_wise(init_w, iterative_times, model, loss_func, optimizer, is_GPU, device, curr_X, curr_Y, full_shape_list, shape_list, total_shape_size, model_param, model_grad, class_id):
    
    vectorized_w = init_w.view(1,-1)#get_all_vectorized_parameters1(, device)
    
    vectorized_model_param = get_all_vectorized_parameters1(model_param, device)
    
#     t1 = time.time()
    
    for k in range(iterative_times):
#         
        
        curr_w_after_perturb = vectorized_model_param + small_constant* vectorized_w
        
        set_model_parameters(model, get_devectorized_parameters(curr_w_after_perturb, full_shape_list, shape_list), device)
        
        model_out = -F.log_softmax(model(curr_X), dim = 1)
        
#         for p in range(num_class):
        optimizer.zero_grad()
        model_out.view(-1)[class_id].backward()
    
#         curr_loss = loss_func(model(curr_X), curr_Y)
#         
#         optimizer.zero_grad()
#         
#         curr_loss.backward()
    
        curr_grad = get_vectorized_grads(model, device)
        
        hessian_w_prod = (curr_grad - model_grad)/(small_constant)
        
        if torch.norm(hessian_w_prod).item() == 0:
            updated_vectorized_w = hessian_w_prod
        else:
            updated_vectorized_w = hessian_w_prod/torch.norm(hessian_w_prod)
        
        print(torch.norm(updated_vectorized_w - vectorized_w))
        
        if torch.isnan(torch.norm(updated_vectorized_w - vectorized_w)):
            print('here')
        
        if torch.norm(updated_vectorized_w - vectorized_w) < 1e-6:
            vectorized_w = updated_vectorized_w
            break
        
        
        vectorized_w = updated_vectorized_w
    
    curr_w_after_perturb = vectorized_model_param + small_constant* vectorized_w
        
    set_model_parameters(model, get_devectorized_parameters(curr_w_after_perturb, full_shape_list, shape_list), device)
    
#     curr_loss = loss_func(model(curr_X), curr_Y)
    model_out = -F.log_softmax(model(curr_X), dim = 1)
    optimizer.zero_grad()
    
    model_out.view(-1)[class_id].backward()
    
    curr_grad = get_vectorized_grads(model, device)
    
    hessian_w_prod = (curr_grad - model_grad)/(small_constant)
    
    
    max_u = torch.dot(hessian_w_prod.view(-1), vectorized_w.view(-1))/torch.dot(vectorized_w.view(-1), vectorized_w.view(-1))
    
#     t2 = time.time()
#     
#     print('time::', t2 - t1)
    
    return max_u  

def compute_max_mu_sample_wise(dataset_train, model, model_param, loss_func, optimizer, is_GPU, device):
    
    max_mu_list = []
    
    iterative_times = 20
    
    set_model_parameters(model, model_param, device)
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_model_param = get_vectorized_params(model)
    
    init_w = torch.rand(vec_model_param.view(-1).shape[0], device = device)
    
    for i in range(dataset_train.data.shape[0]):
        
#         t1 = time.time()
        
        print('sample::', i)
        
        curr_data = dataset_train.data[i:i+1]
        
        curr_label = dataset_train.labels[i:i+1]
        
        if is_GPU:
            curr_data = curr_data.to(device)
            
            curr_label = curr_label.to(device)
#         t5 = time.time()
        curr_loss = loss_func(model(curr_data), curr_label)
        
        optimizer.zero_grad()
        
        curr_loss.backward()
        
        curr_grad = get_vectorized_grads(model, device)
        
#         t3 = time.time()
        
        max_u = compute_single_max_mu(init_w.clone(), iterative_times, model, loss_func, optimizer, is_GPU, device, curr_data, curr_label, full_shape_list, shape_list, total_shape_size, model_param, curr_grad)
        
#         t4 = time.time()
        
        max_mu_list.append(max_u.cpu())
        
#         t2 = time.time()
        
#         print('time::', t2 - t1, t4 -t3, t5 - t1)
        
#         print('here')
        
    return max_mu_list
        

def compute_max_mu_sample_class_wise(dataset_train, model, model_param, loss_func, optimizer, is_GPU, device, num_class):
    
    max_mu_list = []
    
    iterative_times = 20
    
    set_model_parameters(model, model_param, device)
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    vec_model_param = get_vectorized_params(model)
    
    init_w = torch.rand(vec_model_param.view(-1).shape[0], device = device)
    
    for i in range(dataset_train.data.shape[0]):
        
        t1 = time.time()
        
        print('sample::', i)
        
        curr_data = dataset_train.data[i:i+1]
        
        curr_label = dataset_train.labels[i:i+1]
        
        if is_GPU:
            curr_data = curr_data.to(device)
             
            curr_label = curr_label.to(device)
#         t5 = time.time()


        curr_max_u_list = []
        
        if i >= 2750:
            print('here')

        for p in range(num_class):
            
            
#         curr_loss = loss_func(, curr_label)
            model_out =  -F.log_softmax(model(curr_data), dim = 1)
            
            curr_loss = loss_func(model(curr_data), curr_label)
            
            print('pred diff::', torch.norm(torch.sum(model_out.view(-1)*curr_label.view(-1)) - curr_loss))
            
            optimizer.zero_grad()
        
            model_out.view(-1)[p].backward()
        
            curr_grad = get_vectorized_grads(model, device)
        
            t3 = time.time()
        
            max_u = compute_single_max_mu_class_wise(init_w.clone(), iterative_times, model, loss_func, optimizer, is_GPU, device, curr_data, curr_label, full_shape_list, shape_list, total_shape_size, model_param, curr_grad, p)
        
            curr_max_u_list.append(max_u.cpu())
            t4 = time.time()
        
        max_mu_list.append(curr_max_u_list)
        
        t2 = time.time()
        
        print('time::', t2 - t1, t4 -t3)
        
#         print('here')
        
    return max_mu_list
        


def compute_lower_bound_hessian_w(grad_list, w_list, model, optimizer, loss_func, random_ids_all_epochs, dataset_train, r_weight, batch_size, regularization_rate, is_GPU, device):
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    i = 0
    
    min_hessian_bound = np.inf
    
    max_grad_bound = 0
    
    grad_bound_list = []
    
    mu_bound_list = []
    
    for k in range(len(random_ids_all_epochs)):
        
        random_ids = random_ids_all_epochs[k]
        
        for p in range(0, dataset_train.data.shape[0], batch_size):
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
            curr_rand_ids = random_ids[p:end_id]
            
            batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
            batch_Y = dataset_train.labels[curr_rand_ids]
            
            batch_r_weight = r_weight[curr_rand_ids]
            
            if is_GPU:
                batch_X = batch_X.to(device)
                
                batch_Y = batch_Y.to(device)
                
                batch_r_weight = batch_r_weight.to(device)
            
            curr_grad = get_all_vectorized_parameters1(grad_list[i])
            
            small_perturbed_w = torch.rand(curr_grad.view(-1).shape[0], device=device)
            
            curr_w = get_all_vectorized_parameters1(w_list[i], device)
            
            curr_grad = grad_list[i]
            
            set_model_parameters(model, get_devectorized_parameters(curr_w, full_shape_list, shape_list), device)
            
            optimizer.zero_grad()
            
            curr_loss = torch.mean(loss_func(model(batch_X), batch_Y)*batch_r_weight)
            
            curr_loss.backward()
            
            exp_grad = get_vectorized_grads(model)
            
            print(torch.norm(exp_grad - curr_grad))
            
            
            curr_w_after_perturb = curr_w + small_constant* small_perturbed_w
            
            set_model_parameters(model, get_devectorized_parameters(curr_w_after_perturb, full_shape_list, shape_list), device)
            
            optimizer.zero_grad()
            
            curr_loss = torch.mean(loss_func(model(batch_X), batch_Y)*batch_r_weight)
            
            curr_loss.backward()
            
            grad_after_perturb = get_vectorized_grads(model)
            
            curr_hessian_lb = (torch.norm((grad_after_perturb - curr_grad))/(small_constant*torch.norm(small_perturbed_w))).detach().cpu().item()
        
            min_hessian_bound = min(curr_hessian_lb, min_hessian_bound)
        
            mu_bound_list.append(curr_hessian_lb)
        
            curr_grad_norm = torch.norm(grad_list[i] + regularization_rate*curr_w.cpu()).item()
        
            print('curr grad norm::', curr_grad_norm)
        
            max_grad_bound = max(max_grad_bound, curr_grad_norm)
        
            grad_bound_list.append(curr_grad_norm)
        
            i += 1
            
    return min_hessian_bound + regularization_rate, max_grad_bound, grad_bound_list, mu_bound_list


def compute_w_bound(mu_bound_list,mu, lr, grad_bound_list,n):
    
    w_bound = 0
    
    w_bound1 = 0
    
    mu_bound_prod = 1
    
    for i in range(len(grad_bound_list)):
        
        w_bound1 += (1-mu*lr)**(len(grad_bound_list) - i-1)*grad_bound_list[i]*2/n*lr
        
        w_bound += mu_bound_prod*grad_bound_list[len(grad_bound_list) - i-1]*2/n*lr
        
        mu_bound_prod *= (1-mu_bound_list[len(grad_bound_list) - i-1]*lr)
        
    return w_bound, w_bound1

def update_gradient_incremental(all_entry_grad_list, random_ids_multi_epochs, update_Y, bz, isGPU, device, expected_update_grad_list=None):
    
    w_id = 0
    
    updated_grad_list = []
    
    for j in range(len(random_ids_multi_epochs)):
        
        random_ids = random_ids_multi_epochs[j]
        
        for k in range(0, update_Y.shape[0], bz):
#     for i in range(len(all_entry_grad_list)):

            end_id = k + bz
            
            if end_id >= update_Y.shape[0]:
                end_id = update_Y.shape[0]

            curr_update_Y = update_Y[random_ids[k:end_id]]
            
            curr_entry_grad = all_entry_grad_list[w_id]
            
            if isGPU:
                curr_update_Y = curr_update_Y.to(device)
                curr_entry_grad = curr_entry_grad.to(device)

            curr_grad = get_entry_grad_with_labels(curr_update_Y, curr_entry_grad)
            
            if expected_update_grad_list is not None:
                curr_exp_grad = expected_update_grad_list[w_id]
                
                print('gap::', w_id, torch.norm(curr_exp_grad - curr_grad))
            
            
            updated_grad_list.append(curr_grad)
            
            w_id += 1
            
    return updated_grad_list

def update_gradient_origin(all_entry_grad_list, random_ids_multi_epochs, X, update_Y, bz, isGPU, device, w_list, model, optimizer):
    w_id = 0
    
    updated_grad_list = []
    
    loss_func = model.soft_loss_function_reduce
    
    for j in range(len(random_ids_multi_epochs)):
        
        random_ids = random_ids_multi_epochs[j]
        
        for k in range(0, update_Y.shape[0], bz):
#     for i in range(len(all_entry_grad_list)):

            end_id = k + bz
            
            if end_id >= update_Y.shape[0]:
                end_id = update_Y.shape[0]

            curr_update_Y = update_Y[random_ids[k:end_id]]
            
            curr_entry_grad = all_entry_grad_list[w_id]
            
            batch_X = X[random_ids[k:end_id]]
            
            
            if isGPU:
                curr_update_Y = curr_update_Y.to(device)
                curr_entry_grad = curr_entry_grad.to(device)
                batch_X = batch_X.to(device)

            set_model_parameters(model, w_list[w_id], device)

            optimizer.zero_grad()

#             print(batch_X.shape, curr_update_Y.shape)

            loss = loss_func(model.forward(batch_X), curr_update_Y)

            loss.backward()

            curr_grad = get_vectorized_grads(model)
            
            updated_grad_list.append(curr_grad)
            
            w_id += 1
            
    return updated_grad_list

def compute_grad_final3(para, hessian_para_prod, grad1, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    
    gradients = None
    
    if gradient_dual is not None:
        
        hessian_para_prod += grad_list_tensor 
        
        hessian_para_prod += beta*para_list_tensor 
        
        gradients = hessian_para_prod*size1
        
        
#         gradients = (hessian_para_prod[i]*size1 - (gradient_dual[i].to('cpu') + beta*para[i])*size2)/(size1 - size2)
        
        gradients -= (gradient_dual + beta*para_list_tensor)*size2
        
        gradients /= (size1 - size2)
            
    else:
        
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        
        gradients = hessian_para_prod
        
        
    delta_para = para - para_list_tensor
    
    delta_grad = hessian_para_prod - (grad_list_tensor + beta*para_list_tensor)
    
    tmp_res = 0
    
    
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradients

    else:
        return False, gradients

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

def update_para_final2(vec_para, gradient_list, alpha, beta, exp_gradient, exp_para, origin_para, is_GPU, device, compare):
    
    exp_grad_list = []
    
    if compare and exp_gradient is not None:
        print("grad_diff::")
           
        compute_para_grad_diff(vec_para, exp_para, gradient_list, exp_gradient, origin_para)
#         compute_model_para_diff(gradient_list, exp_grad_list)
          
            
        print("here!!")
    
    vec_para -= alpha*gradient_list
    

        
        
    return vec_para

def clear_gradients(para_list):
    for param in para_list:
        param.grad.zero_()

def compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer, r_weight = None):
    optimizer.zero_grad()

    output = model(batch_X)

    loss = criterion(output, batch_Y, r_weight)
    
    loss.backward()
    
    return loss


def compute_derivative_one_more_step_diff(model, batch_X, batch_Y, batch_Y2, criterion, optimizer, r_weight = None, r_weight2 = None):
    optimizer.zero_grad()

    output = model(batch_X)

    loss = criterion(output, batch_Y, r_weight)
    
    loss2 = criterion(output, batch_Y2, r_weight2)
    
    diff_loss = loss - loss2
    
    diff_loss.backward()
    
    return diff_loss

def compute_unreduced_loss(model, batch_X, batch_Y, criterion_unreduce):
    loss = criterion_unreduce(model(batch_X), batch_Y)
    return loss

def compute_derivative_one_more_step_unreduced_first(model, batch_X, batch_Y, criterion_unreduce, optimizer):
    optimizer.zero_grad()

    output = model(batch_X)

    loss = criterion_unreduce(output, batch_Y)
    
    full_loss = torch.sum(loss)
    
    full_loss.backward()
    
    return loss



def post_processing_gradien_para_list_all_epochs(para_list_all_epochs, grad_list_all_epochs):
    
#     num = 0
    
    _,_,total_shape_size = get_model_para_shape_list(para_list_all_epochs[0])
        
    
    
    para_list_all_epoch_tensor = torch.zeros([len(para_list_all_epochs), total_shape_size], dtype = torch.double)
    
    grad_list_all_epoch_tensor = torch.zeros([len(grad_list_all_epochs), total_shape_size], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)):
        
        para_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(para_list_all_epochs[i])
        
        grad_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(grad_list_all_epochs[i])
        
    
    
    
    return para_list_all_epoch_tensor, grad_list_all_epoch_tensor

def post_processing_para_list_all_epochs(para_list_all_epochs):
    
#     num = 0
    
    _,_,total_shape_size = get_model_para_shape_list(para_list_all_epochs[0])
        
    
    
    para_list_all_epoch_tensor = torch.zeros([len(para_list_all_epochs), total_shape_size], dtype = torch.double)
    
    for i in range(len(para_list_all_epochs)):
        
        para_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(para_list_all_epochs[i])
        
    
    return para_list_all_epoch_tensor



'''pre-fetch parts of the history parameters and gradients into GPU to save the IO overhead'''
def cache_grad_para_history(full_out_dir, cached_size, is_GPU, device, para_list_all_epochs, gradient_list_all_epochs, updated_gradient_list_all_epochs=None):
#     if para_list_all_epochs is None:
#         para_list_all_epochs = torch.load(os.path.join(full_out_dir, 'para_list_all_epochs'))
#         
#     if gradient_list_all_epochs is None:
#         gradient_list_all_epochs = torch.load(os.path.join(full_out_dir, 'gradient_list_all_epochs'))
#     
#     torch.save(para_list_all_epochs, os.path.join(full_out_dir, 'para_list_all_epochs'))
#     torch.save(gradient_list_all_epochs, os.path.join(full_out_dir, 'para_list_all_epochs'))
    
#     para_list_all_epoch_tensor, grad_list_all_epoch_tensor = post_processing_gradien_para_list_all_epochs(para_list_all_epochs, gradient_list_all_epochs)

    para_list_all_epoch_tensor = post_processing_para_list_all_epochs(para_list_all_epochs)

    grad_list_all_epoch_tensor = torch.stack(gradient_list_all_epochs, dim = 0)
    
    
    updated_grad_list_all_epoch_tensor = None
    if updated_gradient_list_all_epochs is not None:
        updated_grad_list_all_epoch_tensor = torch.stack(updated_gradient_list_all_epochs, dim = 0)  

    end_cached_id = cached_size
    
    if end_cached_id > len(para_list_all_epochs):
        end_cached_id =  len(para_list_all_epochs)
    

    para_list_GPU_tensor = para_list_all_epoch_tensor[0:cached_size]
    
    grad_list_GPU_tensor = grad_list_all_epoch_tensor[0:cached_size]
    
    updated_grad_list_GPU_tensor = None
    
    if updated_gradient_list_all_epochs is not None:
        updated_grad_list_GPU_tensor = updated_grad_list_all_epoch_tensor[0:cached_size]

    if is_GPU:
        para_list_GPU_tensor = para_list_GPU_tensor.to(device)
        
        grad_list_GPU_tensor = grad_list_GPU_tensor.to(device) 
        
        if updated_gradient_list_all_epochs is not None:
            updated_grad_list_GPU_tensor = updated_grad_list_GPU_tensor.to(device)
    
    return grad_list_all_epoch_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor

def obtain_gradients_each_class(w_list, origin_grad_list, model, random_ids_per_epochs, dataset_train, batch_size, num_class, optimizer, is_GPU, device):
    
    w_id = 0
    
    
    all_entry_grad_list = []
    
    for k in range(len(random_ids_per_epochs)):
        
        curr_random_ids = random_ids_per_epochs[k]
        
        for i in range(0, dataset_train.data.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= dataset_train.data.shape[0]:
                    end_id = dataset_train.data.shape[0]
        
                batch_x, batch_y = dataset_train.data[curr_random_ids[i:end_id]], dataset_train.labels[curr_random_ids[i:end_id]]
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                set_model_parameters(model, w_list[w_id])
                
                model_out = F.log_softmax(model(batch_x))
    
                curr_grad_list = []
                
                for x in range(batch_x.shape[0]):
                    curr_sample_grad_list = []
                    for j in range(num_class):
                        optimizer.zero_grad()
                        model_out[x][j].backward(retain_graph=True)
                        grad_list = get_vectorized_grads(model)
                        curr_sample_grad_list.append(grad_list.cpu())
                    
                    curr_grad_list.append(torch.cat(curr_sample_grad_list, dim = 0))
                    
                curr_grad_list_tensor = torch.stack(curr_grad_list, dim = 0)
                all_entry_grad_list.append(curr_grad_list_tensor)
                cal_grad = -torch.sum(curr_grad_list_tensor*batch_y.view(batch_y.shape[0], batch_y.shape[1],1), dim=(0,1))/batch_x.shape[0]
                print(k, i, torch.norm(cal_grad - origin_grad_list[w_id]))
#                     set_model_parameters(model, param_list)
                
                w_id += 1
    
    
    return all_entry_grad_list


def obtain_updated_gradients(w_list, grad_list, model, random_ids_per_epochs, dataset_train, batch_size, num_class, optimizer, all_entry_grad_list, updated_labels, isGPU, device):
    
    w_id = 0
    
    curr_loss = model.soft_loss_function_reduce
    
    for k in range(len(random_ids_per_epochs)):
        
        curr_random_ids = random_ids_per_epochs[k]
        
        for i in range(0, dataset_train.data.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= dataset_train.data.shape[0]:
                    end_id = dataset_train.data.shape[0]
        
                batch_x, batch_y = dataset_train.data[curr_random_ids[i:end_id]], dataset_train.labels[curr_random_ids[i:end_id]]
                
                updated_batch_y = updated_labels[curr_random_ids[i:end_id]]
                
                curr_entry_grad = all_entry_grad_list[w_id]
                
                if isGPU:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    updated_batch_y = updated_batch_y.to(device)
                    curr_entry_grad = curr_entry_grad.to(device)
                
#                 update_batch_y_full = (updated_batch_y).view(batch_y.shape[0], batch_y.shape[1], 1).repeat(1,1,all_entry_grad_list[w_id].shape[-1])
                
#                 print(update_batch_y_full.shape)
                
                optimizer.zero_grad()
                
                set_model_parameters(model, w_list[w_id], device)
                
#                 curr_loss = model.get_loss_function('mean')
                
                t1 = time.time()
                
                loss = curr_loss(model.forward(batch_x), updated_batch_y.type(torch.long)).view(-1)
                
                loss.backward()
                
                exp_grad_list = get_vectorized_grads(model)
                
                t2 = time.time()
                
#                 curr_entry_grad_list = torch.stack(all_entry_grad_list[w_id], dim = 0)
                
#                 curr_grad_list = grad_list[w_id].view(-1) - torch.sum(all_entry_grad_list[w_id]*(updated_batch_y - batch_y).view(batch_y.shape[0], batch_y.shape[1], 1), dim = (0,1)).view(-1)/batch_y.shape[0] 
                
#                 curr_grad_list2 = - torch.sum(all_entry_grad_list[w_id]*update_batch_y_full, dim = (0,1)).view(-1)/batch_y.shape[0]
                
                curr_grad_list = -(torch.mm(updated_batch_y.view(1,-1),curr_entry_grad.view(curr_entry_grad.shape[0]*curr_entry_grad.shape[1], -1))/batch_y.shape[0]).view(-1)
                
                curr_grad_list = curr_grad_list.cpu()
#                 print(torch.norm(curr_grad_list - curr_grad_list2))
                
                t3 = time.time()
                
                w_id += 1
                
                print(torch.norm(exp_grad_list.view(-1) - curr_grad_list))
                print('time::', t3 - t2, t2 - t1)
                print('here')



def get_model_params_per_epochs(random_ids_multi_super_iterations, w_list, final_model_param, train_dataset, batch_size):
    i = 0
    model_param_per_epoch_list = []
    
#     for i in range(len(w_list)):
    for j in range(len(random_ids_multi_super_iterations)):
        
        print('epoch::', j)
        
        for k in range(0,train_dataset.data.shape[0], batch_size):
            i+=1
        
        if j < len(random_ids_multi_super_iterations) - 1:
            model_param_per_epoch_list.append(w_list[i])
        else:
            model_param_per_epoch_list.append(final_model_param)
        
    return model_param_per_epoch_list

def select_params_early_stop(epochs, random_ids_multi_super_iterations, w_list, final_model_param, model, train_dataset, val_dataset, test_dataset, batch_size, is_GPU, device, no_prov = False):
    
    early_stopping = models.EarlyStopping(patience=len(random_ids_multi_super_iterations), verbose=True, delta = 0)
    
    
    # if_dnn = True
    # capture_prov = False

    if not ((type(model) is models.Logistic_regression) or (type(model) is models.Binary_Logistic_regression)):
        return model, 0
    
    if no_prov:
        return model,0
        # if_dnn = False
        # capture_prov = True
    
    i = 0
#     for i in range(len(w_list)):
    # for j in range(len(random_ids_multi_super_iterations)):
    
    epochs = min(epochs, len(random_ids_multi_super_iterations))
    
    print(epochs, len(random_ids_multi_super_iterations), train_dataset.data.shape[0], batch_size, len(w_list))
    
    for j in range(epochs):
        print('epoch::', j)
        
        for k in range(0,train_dataset.data.shape[0], batch_size):
            i+=1
        
#         if j >= 42:
#             print('here')
        if j < len(random_ids_multi_super_iterations) - 1:
            set_model_parameters(model, w_list[i], device)
        else:
            set_model_parameters(model, final_model_param, device)
        valid_loss = valid_model_dataset(model, val_dataset, None, batch_size, 'valid', is_GPU, device, f1 = False)
        
        valid_model_dataset(model, test_dataset, None, batch_size, 'test', is_GPU, device, f1 = False)
        
        early_stopping(valid_loss, model, j)
        
        if early_stopping.early_stop:
            break
    
    models.set_model_parameters(model, early_stopping.model_param, device)

    return model, early_stopping.model_epoch


def simulate_human_annotations(num_humans, full_training_origin_labels, full_out_dir):
    
    ratio = 0.1
    
    full_lenth = full_training_origin_labels.shape[0]
    
    for k in range(num_humans):
        random_ids = torch.randperm(full_lenth)
        
        perturbed_ids = random_ids[0:int(full_lenth*ratio)]
        
        annotated_labels = full_training_origin_labels.clone()
        
        annotated_labels[perturbed_ids] = 1 - annotated_labels[perturbed_ids] 
        
        torch.save(annotated_labels, full_out_dir + '/human_annotated_labels_' + str(k))
        
    torch.save(num_humans, full_out_dir + '/human_annotated_count')



def initial_train_model(train_dataset, val_dataset, test_dataset, args, binary=False, is_early_stopping = True, random_ids_multi_super_iterations = None, r_weight = None):
    size = 1
        
    for k in range(len(train_dataset.data.shape)-1):
        size *= train_dataset.data.shape[k+1]
    
    if args.model == 'Binary_Logistic_regression':
            
        model = models.Binary_Logistic_regression(size, bias = True)
    else:
        if args.model == 'Logistic_regression':
            model = models.Logistic_regression(size, args.num_class, bias = True)
            
        else:
            
            model_class = getattr(models, args.model)
            model = model_class(train_dataset.data.shape, args.num_class)
    
    
    if_dnn = True
    capture_prov = False

    if (type(model) is models.Logistic_regression) or (type(model) is models.Binary_Logistic_regression):
        if_dnn = False
        capture_prov = True
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
        
    args.optim = optimizer

    model = model.to(args.device)
    
    args.loss = model.soft_loss_function_reduce
    
#     args.loss = None
    print('if dnn::', if_dnn)
    w_list, grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, train_dataset.data, train_dataset.labels, args.bz, args.epochs, args.GPU, args.device, loss_func = args.loss, val_dataset = val_dataset, test_dataset = test_dataset, f1 = False, capture_prov = capture_prov, is_early_stopping =is_early_stopping, r_weight = r_weight)
    
    
    return w_list, grad_list, random_ids_multi_super_iterations, optimizer, model
    

def obtain_updated_labels(full_output_dir, args, X_train, Y_train, Y_train_full, existing_labeled_id_tensor=None, size = None, start = True, iter_count = None, is_incremental = False, method = 'influence', derive_probab_labels = False, train_annotated_label_tensor = None, suffix = None):
    
    if iter_count is None:
        full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids', map_location='cpu')
    else:
        if suffix is not None:
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix, map_location='cpu')
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count), map_location='cpu')
                
        else:
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_v' + str(iter_count) + incremental_suffix, map_location='cpu')
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_v' + str(iter_count), map_location='cpu')
    
    removed_id_tensor = full_removed_id_tensor
    
    origin_labeled_id_tensor = None
    
    if os.path.exists(full_output_dir + '/clean_sample_ids'):
        origin_labeled_id_tensor = torch.load(full_output_dir + '/clean_sample_ids')
        if existing_labeled_id_tensor is not None:
            existing_labeled_id_tensor = torch.cat([origin_labeled_id_tensor.view(-1), existing_labeled_id_tensor.view(-1)], dim = 0)
        else:
            existing_labeled_id_tensor = origin_labeled_id_tensor.clone()
#     removed_id_tensor = torch.load(full_output_dir + '/noisy_sample_ids')
    
    if existing_labeled_id_tensor is not None:
        existing_boolean_tensor = (torch.sum(removed_id_tensor.view(1,-1) == existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()
        
        removed_id_tensor = removed_id_tensor[~existing_boolean_tensor]
        
        removed_id_tensor = removed_id_tensor[0:args.removed_count]
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
        
    else:
        
        selected_random_ids = []
        
        count = 5
        
        for k in range(args.num_class):
            
            curr_count = 0 
            
            for i in range(removed_id_tensor.shape[0]):
                if Y_train_full[removed_id_tensor[i]].item() == k:
                    curr_count += 1
                    selected_random_ids.append(removed_id_tensor[i])
                    
                    if curr_count >= count:
                        break
                     
        
#         removed_id_tensor = removed_id_tensor[0:25]
#         removed_id_tensor = torch.tensor(selected_random_ids)
    
#         removed_id_tensor = removed_id_tensor[0:args.removed_count]
    
        removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
    
    if size is not None:
        removed_id_tensor = removed_id_tensor[removed_id_tensor < size]
    
    
    human_annotated_labels = load_human_annotations(0, full_output_dir, train_annotated_label_tensor)
    
    print('removed_id_size::', removed_id_tensor.shape)
    Y_train_final = Y_train.cpu().detach().clone()
    
    
    final_labeled_id_tensor = removed_id_tensor[~(Y_train_full[removed_id_tensor] == -1)]
    
    if len(Y_train_final.shape) >= 2:
#         Y_train_final[final_labeled_id_tensor] = onehot(Y_train_full[final_labeled_id_tensor].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)
        agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor)
        
        Y_train_final[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]] = onehot(agg_labels[agg_labels.type(torch.LongTensor) !=-1].type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)

    else:
        Y_train_final[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]] = Y_train_full[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]].type(torch.DoubleTensor).view(-1)
    
    
    full_existing_labeled_id_tensor = torch.zeros(X_train.shape[0]).bool()
    
    if existing_labeled_id_tensor is not None:
        
        full_existing_labeled_id_tensor[existing_labeled_id_tensor.type(torch.long)] = True
        
        print('existing labeled_count::', torch.sum(full_existing_labeled_id_tensor))
    
    full_existing_labeled_id_tensor[final_labeled_id_tensor] = True
    
    if derive_probab_labels:
    
    
        small_dataset_train_extended = models.MyDataset(torch.cat([X_train[full_existing_labeled_id_tensor]], dim = 0), torch.cat([Y_train_full[full_existing_labeled_id_tensor].type(torch.LongTensor)], dim = 0))
        
        print('full labeled_count::', torch.sum(full_existing_labeled_id_tensor), small_dataset_train_extended.lenth, torch.unique(Y_train_full[full_existing_labeled_id_tensor]))

    
        final_unlabeled_id_tensor = torch.tensor(list(set(range(Y_train.shape[0])).difference(set(torch.nonzero(full_existing_labeled_id_tensor).view(-1).tolist()))))
        
        dataset_to_be_labeled = models.MyDataset(X_train[final_unlabeled_id_tensor], Y_train_final[final_unlabeled_id_tensor])
        
        print('dataset to be labeled count::', final_unlabeled_id_tensor.shape[0])
        
        if start:
            remaining_AFs = torch.load(full_output_dir + '/remaining_AFs')
            remaining_AFs_list = []
             
            for i in range(remaining_AFs.shape[0]):
                remaining_AFs_list.append(remaining_AFs[i])
            print(len(remaining_AFs_list), remaining_AFs_list[0].shape)
            LPs, num_class = goggles.compute_LPs(remaining_AFs, final_labeled_id_tensor.view(-1).tolist(),small_dataset_train_extended.labels.view(-1).tolist())
            torch.save(LPs, full_output_dir + '/LPs')
            torch.save(num_class, full_output_dir + '/num_class')
        else:
            LPs = torch.load(full_output_dir + '/LPs')
            num_class = torch.load(full_output_dir + '/num_class')
        
        dataset_to_be_labeled, _, _ = label_remaining_dataset2(args,small_dataset_train_extended, True, dataset_to_be_labeled.data, Y_train_full[final_unlabeled_id_tensor], torch.nonzero(full_existing_labeled_id_tensor).view(-1), final_unlabeled_id_tensor, LPs = LPs, num_class = num_class)
        
        Y_train_final[final_unlabeled_id_tensor] = dataset_to_be_labeled.labels
    
    
    print(Y_train_final)
    
    return Y_train_final, torch.nonzero(full_existing_labeled_id_tensor).view(-1)


def obtain_updated_labels_tars(full_output_dir, args, X_train, Y_train, Y_train_full, existing_labeled_id_tensor=None, size = None, start = True, iter_count = None, is_incremental = False, method = 'influence', derive_probab_labels = False, train_annotated_label_tensor = None, suffix = None):
    
    if iter_count is None:
        full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids', map_location='cpu')
    else:
        if suffix is not None:
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix, map_location='cpu')
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count), map_location='cpu')
                
        else:
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_v' + str(iter_count) + incremental_suffix, map_location='cpu')
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids_v' + str(iter_count), map_location='cpu')
    
    removed_id_tensor = full_removed_id_tensor
    
    origin_labeled_id_tensor = None
    
    if os.path.exists(full_output_dir + '/clean_sample_ids'):
        origin_labeled_id_tensor = torch.load(full_output_dir + '/clean_sample_ids')
        if existing_labeled_id_tensor is not None:
            existing_labeled_id_tensor = torch.cat([origin_labeled_id_tensor.view(-1), existing_labeled_id_tensor.view(-1)], dim = 0)
        else:
            existing_labeled_id_tensor = origin_labeled_id_tensor.clone()
#     removed_id_tensor = torch.load(full_output_dir + '/noisy_sample_ids')
    
    if existing_labeled_id_tensor is not None:
        existing_boolean_tensor = (torch.sum(removed_id_tensor.view(1,-1) == existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()
        
        removed_id_tensor = removed_id_tensor[~existing_boolean_tensor]
        
        removed_id_tensor = removed_id_tensor[0:args.removed_count]
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
        
    else:
        
        selected_random_ids = []
        
        count = 5
        
        for k in range(args.num_class):
            
            curr_count = 0 
            
            for i in range(removed_id_tensor.shape[0]):
                if Y_train_full[removed_id_tensor[i]].item() == k:
                    curr_count += 1
                    selected_random_ids.append(removed_id_tensor[i])
                    
                    if curr_count >= count:
                        break
                     
        
#         removed_id_tensor = removed_id_tensor[0:25]
#         removed_id_tensor = torch.tensor(selected_random_ids)
    
#         removed_id_tensor = removed_id_tensor[0:args.removed_count]
    
        removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
    
    if size is not None:
        removed_id_tensor = removed_id_tensor[removed_id_tensor < size]
    
    
    human_annotated_labels = load_human_annotations(0, full_output_dir, train_annotated_label_tensor)
    
    print('removed_id_size::', removed_id_tensor.shape)
    Y_train_final = Y_train.cpu().clone()
    
    
    final_labeled_id_tensor = removed_id_tensor[~(Y_train_full[removed_id_tensor] == -1)]
    
    if len(Y_train_final.shape) >= 2:
#         Y_train_final[final_labeled_id_tensor] = onehot(Y_train_full[final_labeled_id_tensor].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)
        agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor)
        
        Y_train_final[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]] = onehot(agg_labels[agg_labels.type(torch.LongTensor) !=-1].type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)

    else:
        Y_train_final[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]] = Y_train_full[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) !=-1]].type(torch.DoubleTensor).view(-1)
    
    
    full_existing_labeled_id_tensor = torch.zeros(X_train.shape[0]).bool()
    
    if existing_labeled_id_tensor is not None:
        
        full_existing_labeled_id_tensor[existing_labeled_id_tensor.type(torch.long)] = True
        
        print('existing labeled_count::', torch.sum(full_existing_labeled_id_tensor))
    
    full_existing_labeled_id_tensor[final_labeled_id_tensor] = True
    
    if derive_probab_labels:
    
    
        small_dataset_train_extended = models.MyDataset(torch.cat([X_train[full_existing_labeled_id_tensor]], dim = 0), torch.cat([Y_train_full[full_existing_labeled_id_tensor].type(torch.LongTensor)], dim = 0))
        
        print('full labeled_count::', torch.sum(full_existing_labeled_id_tensor), small_dataset_train_extended.lenth, torch.unique(Y_train_full[full_existing_labeled_id_tensor]))

    
        final_unlabeled_id_tensor = torch.tensor(list(set(range(Y_train.shape[0])).difference(set(torch.nonzero(full_existing_labeled_id_tensor).view(-1).tolist()))))
        
        dataset_to_be_labeled = models.MyDataset(X_train[final_unlabeled_id_tensor], Y_train_final[final_unlabeled_id_tensor])
        
        print('dataset to be labeled count::', final_unlabeled_id_tensor.shape[0])
        
        if start:
            remaining_AFs = torch.load(full_output_dir + '/remaining_AFs')
            remaining_AFs_list = []
             
            for i in range(remaining_AFs.shape[0]):
                remaining_AFs_list.append(remaining_AFs[i])
            print(len(remaining_AFs_list), remaining_AFs_list[0].shape)
            LPs, num_class = goggles.compute_LPs(remaining_AFs, final_labeled_id_tensor.view(-1).tolist(),small_dataset_train_extended.labels.view(-1).tolist())
            torch.save(LPs, full_output_dir + '/LPs')
            torch.save(num_class, full_output_dir + '/num_class')
        else:
            LPs = torch.load(full_output_dir + '/LPs')
            num_class = torch.load(full_output_dir + '/num_class')
        
        dataset_to_be_labeled, _, _ = label_remaining_dataset2(args,small_dataset_train_extended, True, dataset_to_be_labeled.data, Y_train_full[final_unlabeled_id_tensor], torch.nonzero(full_existing_labeled_id_tensor).view(-1), final_unlabeled_id_tensor, LPs = LPs, num_class = num_class)
        
        Y_train_final[final_unlabeled_id_tensor] = dataset_to_be_labeled.labels
    
    
    
    
    return Y_train_final, torch.nonzero(full_existing_labeled_id_tensor).view(-1)



def get_removed_ids(sorted_train_ids, suggested_updated_labels, selected_count):
    final_selected_count = 0
    
    final_selected_ids = set()
    
    final_selected_id_tensor = torch.zeros(selected_count)
    
    final_suggested_update_labels = torch.zeros(selected_count)
    
    k = 0
    
    while(final_selected_count < selected_count):
        curr_selected_ids = sorted_train_ids[k*selected_count: (k+1)*selected_count]
    
        suggested_update_labels = suggested_updated_labels[k*selected_count: (k+1)*selected_count]
    
        for p in range(selected_count):
            if not curr_selected_ids[p] in final_selected_ids:
                final_selected_ids.add(curr_selected_ids[p])
                final_selected_id_tensor[final_selected_count] = curr_selected_ids[p]
                final_suggested_update_labels[final_selected_count] = suggested_update_labels[p]
                final_selected_count += 1
        
        k += 1
        
    return final_selected_id_tensor.type(torch.long), final_suggested_update_labels

def load_human_annotations(resolve_conflict, full_out_dir, train_annotated_label_tensor = None):
    
    if train_annotated_label_tensor is None:
    
        num_humans = torch.load(full_out_dir + '/human_annotated_count')
    
        annotated_labels_list = []
        
        if resolve_conflict == 1:
            num_humans = num_humans-1
            
        
        for k in range(num_humans):
    #         random_ids = torch.randperm(full_lenth)
    #         
    #         perturbed_ids = random_ids[0:int(full_lenth*ratio)]
    #         
    #         annotated_labels = full_training_origin_labels.clone()
    #         
    #         annotated_labels[perturbed_ids] = 1 - annotated_labels[perturbed_ids] 
            
            annotated_labels = torch.load(full_out_dir + '/human_annotated_labels_' + str(k))
            
            annotated_labels_list.append(annotated_labels.view(-1))
            
        annotated_labels_tensor = torch.stack(annotated_labels_list, dim = 1)
    
    else:
        num_humans = 3
        
        if resolve_conflict == 1:
            num_humans = num_humans-1
            
        annotated_labels_tensor = train_annotated_label_tensor[:,0:num_humans]
    
    return annotated_labels_tensor

def resolve_conflict_majority_vote(human_annotated_labels, labeled_ids, suggested_labels=None):
    if suggested_labels is not None:
        all_labels = torch.cat([human_annotated_labels[labeled_ids], suggested_labels.type(torch.LongTensor).view(-1,1)], dim =1)
        # final_mode_labels  = torch.mode(all_labels)[0]
        final_mode_labels = obtain_origin_labels(all_labels)
    else:
        # final_mode_labels = torch.mode(human_annotated_labels[labeled_ids])[0]
    
        final_mode_labels = obtain_origin_labels(human_annotated_labels[labeled_ids])
    
    return final_mode_labels
    
    
def obtain_updated_labels_class_wise0(full_output_dir, args, X_train, Y_train, Y_train_full, existing_labeled_id_tensor=None, size = None, start = True, iter_count = None, is_incremental = False, method = 'influence', derive_probab_labels = False, method_flag = '', resolve_conflict=None):
    
#     torch.save(origin_sorted_train_ids_2, full_out_dir + '/' + args.model + '_influence_removed_ids_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(ordered_list_2, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(final_influence_values_2, full_out_dir + '/' + args.model + '_influence_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(s_test_vec_2, full_out_dir + '/' + args.model + '_influence_s_test_vec_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(suggested_updated_labels, full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(remaining_ids, full_out_dir + '/' + args.model + '_influence_remaining_ids_sl_v' + str(iter_count) + incremental_suffix)

    
    if iter_count is None:
        full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids', map_location='cpu')
    else:
        '''full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_sl_v' + str(iter_count) + incremental_suffix'''
#         if resolve_conflict is not None:
        human_annotated_labels = load_human_annotations(0, full_output_dir)
#             if is_incremental:
#                 full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str, map_location='cpu')
#                 suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#             else:
#                 full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count)+ args.resolve_conflict_str, map_location='cpu')
#                 suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count)+ args.resolve_conflict_str)
#         else:
    
        if is_incremental:
            full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count) + incremental_suffix, map_location='cpu')
            suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count) + incremental_suffix)
        else:
            full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count), map_location='cpu')
            suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count))
    
    
    removed_id_tensor = full_removed_id_tensor
    
    origin_labeled_id_tensor = None
    
    if os.path.exists(full_output_dir + '/clean_sample_ids'):
        origin_labeled_id_tensor = torch.load(full_output_dir + '/clean_sample_ids')
        if existing_labeled_id_tensor is not None:
            existing_labeled_id_tensor = torch.cat([origin_labeled_id_tensor.view(-1), existing_labeled_id_tensor.view(-1)], dim = 0).unique()
        else:
            existing_labeled_id_tensor = origin_labeled_id_tensor.clone()
#     removed_id_tensor = torch.load(full_output_dir + '/noisy_sample_ids')
    
    if existing_labeled_id_tensor is not None:
        existing_boolean_tensor = (torch.sum(removed_id_tensor.view(1,-1) == existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()
        
        removed_id_tensor = removed_id_tensor[~existing_boolean_tensor]
        
        removed_id_tensor, suggested_updated_labels = get_removed_ids(removed_id_tensor, suggested_updated_labels, args.removed_count)
        
#         removed_id_tensor = removed_id_tensor[0:args.removed_count]
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
        
    else:
        
        selected_random_ids = []
        
        count = 5
        
        for k in range(args.num_class):
            
            curr_count = 0 
            
            for i in range(removed_id_tensor.shape[0]):
                if Y_train_full[removed_id_tensor[i]].item() == k:
                    curr_count += 1
                    selected_random_ids.append(removed_id_tensor[i])
                    
                    if curr_count >= count:
                        break
                     
        
#         removed_id_tensor = removed_id_tensor[0:25]
#         removed_id_tensor = torch.tensor(selected_random_ids)
    
        removed_id_tensor = removed_id_tensor[0:args.removed_count]
    
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
    
    if size is not None:
        removed_id_tensor = removed_id_tensor[removed_id_tensor < size]
        
    print('removed_id_size::', removed_id_tensor.shape)
    Y_train_final = Y_train.cpu().clone()
    
    
    final_labeled_id_tensor = removed_id_tensor[~(Y_train_full[removed_id_tensor] == -1)]
    
    if len(Y_train_final.shape) >= 2:
#         if resolve_conflict is not None:
            
#             if resolve_conflict == 1:
#                 agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor, suggested_updated_labels)
#             else:
        agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor)
        
        Y_train_final[final_labeled_id_tensor] = onehot(agg_labels.type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)
        
#             non_conflict_ids =  final_labeled_id_tensor[suggested_updated_labels.type(torch.LongTensor) == Y_train_full[final_labeled_id_tensor].type(torch.LongTensor)]
#             
#             Y_train_final[non_conflict_ids] = onehot(Y_train_full[non_conflict_ids].type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)
#             
#             conflict_ids =  final_labeled_id_tensor[~(suggested_updated_labels.type(torch.LongTensor) == Y_train_full[final_labeled_id_tensor].type(torch.LongTensor))]
#             
#             Y_train_final[conflict_ids] = torch.tensor([0.5, 0.5], dtype = Y_train_final.dtype)

        print('Y train difference::', torch.sum((Y_train_full[final_labeled_id_tensor] - agg_labels) == 0), torch.sum((Y_train_full[final_labeled_id_tensor] - suggested_updated_labels.type(torch.LongTensor)) == 0))
#         else:
#             Y_train_final[final_labeled_id_tensor] = onehot(Y_train_full[final_labeled_id_tensor].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)
            
#             Y_train_final[final_labeled_id_tensor] = onehot(suggested_updated_labels.type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)

#         print('output::', Y_train_final[final_labeled_id_tensor][0:10], onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)[0:10])

#             print('Y train difference::', torch.sum(torch.norm(Y_train_final[final_labeled_id_tensor] - onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor), dim = 1) == 0))
    else:
        Y_train_final[final_labeled_id_tensor] = Y_train_full[final_labeled_id_tensor].type(torch.DoubleTensor).view(-1)
        print('Y train difference::', torch.sum(torch.norm(Y_train_final[final_labeled_id_tensor] - onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor), dim = 1) == 0))
    
    full_existing_labeled_id_tensor = torch.zeros(X_train.shape[0]).bool()
    
    if existing_labeled_id_tensor is not None:
        
        full_existing_labeled_id_tensor[existing_labeled_id_tensor] = True
        
        print('existing labeled_count::', torch.sum(full_existing_labeled_id_tensor))
    
    full_existing_labeled_id_tensor[final_labeled_id_tensor] = True
    
    if derive_probab_labels:
    
    
        small_dataset_train_extended = models.MyDataset(torch.cat([X_train[full_existing_labeled_id_tensor]], dim = 0), torch.cat([Y_train_full[full_existing_labeled_id_tensor].type(torch.LongTensor)], dim = 0))
        
        print('full labeled_count::', torch.sum(full_existing_labeled_id_tensor), small_dataset_train_extended.lenth, torch.unique(Y_train_full[full_existing_labeled_id_tensor]))

    
        final_unlabeled_id_tensor = torch.tensor(list(set(range(Y_train.shape[0])).difference(set(torch.nonzero(full_existing_labeled_id_tensor).view(-1).tolist()))))
        
        dataset_to_be_labeled = models.MyDataset(X_train[final_unlabeled_id_tensor], Y_train_final[final_unlabeled_id_tensor])
        
        print('dataset to be labeled count::', final_unlabeled_id_tensor.shape[0])
        
        if start:
            remaining_AFs = torch.load(full_output_dir + '/remaining_AFs')
            remaining_AFs_list = []
             
            for i in range(remaining_AFs.shape[0]):
                remaining_AFs_list.append(remaining_AFs[i])
            print(len(remaining_AFs_list), remaining_AFs_list[0].shape)
            LPs, num_class = goggles.compute_LPs(remaining_AFs, final_labeled_id_tensor.view(-1).tolist(),small_dataset_train_extended.labels.view(-1).tolist())
            torch.save(LPs, full_output_dir + '/LPs')
            torch.save(num_class, full_output_dir + '/num_class')
        else:
            LPs = torch.load(full_output_dir + '/LPs')
            num_class = torch.load(full_output_dir + '/num_class')
        
        dataset_to_be_labeled, _, _ = label_remaining_dataset2(args,small_dataset_train_extended, True, dataset_to_be_labeled.data, Y_train_full[final_unlabeled_id_tensor], torch.nonzero(full_existing_labeled_id_tensor).view(-1), final_unlabeled_id_tensor, LPs = LPs, num_class = num_class)
        
        Y_train_final[final_unlabeled_id_tensor] = dataset_to_be_labeled.labels
    
    
    
    
    return Y_train_final, torch.nonzero(full_existing_labeled_id_tensor).view(-1)


def obtain_updated_labels_class_wise(full_output_dir, args, X_train, Y_train, Y_train_full, existing_labeled_id_tensor=None, size = None, start = True, iter_count = None, is_incremental = False, method = 'influence', derive_probab_labels = False, method_flag = '', resolve_conflict=None, train_annotated_label_tensor = None):
    
#     torch.save(origin_sorted_train_ids_2, full_out_dir + '/' + args.model + '_influence_removed_ids_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(ordered_list_2, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(final_influence_values_2, full_out_dir + '/' + args.model + '_influence_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(s_test_vec_2, full_out_dir + '/' + args.model + '_influence_s_test_vec_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(suggested_updated_labels, full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_sl_v' + str(iter_count) + incremental_suffix)
#     
#     torch.save(remaining_ids, full_out_dir + '/' + args.model + '_influence_remaining_ids_sl_v' + str(iter_count) + incremental_suffix)

    
    if iter_count is None:
        full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids', map_location='cpu')
    else:
        '''full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_sl_v' + str(iter_count) + incremental_suffix'''
        if resolve_conflict is not None:
            
            # if train_annotated_label_tensor is None:
            human_annotated_labels = load_human_annotations(resolve_conflict, full_output_dir, train_annotated_label_tensor)
                
            # else:
                # human_annotated_labels = train_annotated_label_tensor
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str, map_location='cpu')
                suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count)+ args.resolve_conflict_str, map_location='cpu')
                suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count)+ args.resolve_conflict_str)
        else:
    
            if is_incremental:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count) + incremental_suffix, map_location='cpu')
                suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count) + incremental_suffix)
            else:
                full_removed_id_tensor = torch.load(full_output_dir + '/' + args.model + '_' + method + '_removed_ids' + method_flag + '_v' + str(iter_count), map_location='cpu')
                suggested_updated_labels = torch.load(full_output_dir + '/' + args.model + '_influence_suggested_updated_labels' + method_flag + '_v' + str(iter_count))
    
    
    removed_id_tensor = full_removed_id_tensor
    
    origin_labeled_id_tensor = None
    
    if os.path.exists(full_output_dir + '/clean_sample_ids'):
        origin_labeled_id_tensor = torch.load(full_output_dir + '/clean_sample_ids')
        if existing_labeled_id_tensor is not None:
            existing_labeled_id_tensor = torch.cat([origin_labeled_id_tensor.view(-1), existing_labeled_id_tensor.view(-1)], dim = 0).unique()
        else:
            existing_labeled_id_tensor = origin_labeled_id_tensor.clone()
#     removed_id_tensor = torch.load(full_output_dir + '/noisy_sample_ids')
    
    if existing_labeled_id_tensor is not None:
        existing_boolean_tensor = (torch.sum(removed_id_tensor.view(1,-1) == existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()
        
        removed_id_tensor = removed_id_tensor[~existing_boolean_tensor]
        
        removed_id_tensor, suggested_updated_labels = get_removed_ids(removed_id_tensor, suggested_updated_labels, args.removed_count)
        
#         removed_id_tensor = removed_id_tensor[0:args.removed_count]
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
        
    else:
        
        selected_random_ids = []
        
        count = 5
        
        for k in range(args.num_class):
            
            curr_count = 0 
            
            for i in range(removed_id_tensor.shape[0]):
                if Y_train_full[removed_id_tensor[i]].item() == k:
                    curr_count += 1
                    selected_random_ids.append(removed_id_tensor[i])
                    
                    if curr_count >= count:
                        break
                     
        
#         removed_id_tensor = removed_id_tensor[0:25]
#         removed_id_tensor = torch.tensor(selected_random_ids)
    
        removed_id_tensor = removed_id_tensor[0:args.removed_count]
    
#         removed_id_tensor = removed_id_tensor[removed_id_tensor.shape[0] - args.removed_count:]
    
    if size is not None:
        removed_id_tensor = removed_id_tensor[removed_id_tensor < size]
        
    print('removed_id_size::', removed_id_tensor.shape)
    Y_train_final = Y_train.cpu().detach().clone()
    
    
    final_labeled_id_tensor = removed_id_tensor#[~(Y_train_full[removed_id_tensor] == -1)]
    
    if len(Y_train_final.shape) >= 2:
        if resolve_conflict is not None:
            
            if resolve_conflict == 1:
                agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor, suggested_updated_labels)
            else:
                if resolve_conflict == 0:
                    agg_labels = resolve_conflict_majority_vote(human_annotated_labels, final_labeled_id_tensor)
                else:
                    agg_labels = suggested_updated_labels.clone()
            
            Y_train_final[final_labeled_id_tensor[agg_labels.type(torch.LongTensor) != -1]] = onehot(agg_labels[agg_labels.type(torch.LongTensor) != -1].type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)
            
            
            
#             non_conflict_ids =  final_labeled_id_tensor[suggested_updated_labels.type(torch.LongTensor) == Y_train_full[final_labeled_id_tensor].type(torch.LongTensor)]
#             
#             Y_train_final[non_conflict_ids] = onehot(Y_train_full[non_conflict_ids].type(torch.LongTensor),args.num_class).type(torch.DoubleTensor)
#             
#             conflict_ids =  final_labeled_id_tensor[~(suggested_updated_labels.type(torch.LongTensor) == Y_train_full[final_labeled_id_tensor].type(torch.LongTensor))]
#             
#             Y_train_final[conflict_ids] = torch.tensor([0.5, 0.5], dtype = Y_train_final.dtype)

            print('Y train difference::', torch.sum((Y_train_full[final_labeled_id_tensor] - agg_labels) == 0), torch.nonzero((Y_train_full[final_labeled_id_tensor] == suggested_updated_labels.type(torch.LongTensor))).shape[0])
            
            print('Y train difference details::', final_labeled_id_tensor[(Y_train_full[final_labeled_id_tensor] != suggested_updated_labels.type(torch.LongTensor))], suggested_updated_labels[(Y_train_full[final_labeled_id_tensor] != suggested_updated_labels.type(torch.LongTensor))])
        else:
            Y_train_final[final_labeled_id_tensor[Y_train_full[final_labeled_id_tensor]!=-1]] = onehot(Y_train_full[final_labeled_id_tensor[Y_train_full[final_labeled_id_tensor]!=-1]].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)
            
#             Y_train_final[final_labeled_id_tensor] = onehot(suggested_updated_labels.type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)

#         print('output::', Y_train_final[final_labeled_id_tensor][0:10], onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor)[0:10])

            print('Y train difference::', torch.sum(torch.norm(Y_train_final[final_labeled_id_tensor] - onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor), dim = 1) == 0))
            
            # print('ids  with diff labels::', torch.sum(torch.norm(Y_train_final[final_labeled_id_tensor] - onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor), dim = 1) == 0))
    else:
        
        
        
        Y_train_final[final_labeled_id_tensor[Y_train_full[final_labeled_id_tensor]!=-1]] = Y_train_full[final_labeled_id_tensor[Y_train_full[final_labeled_id_tensor] != -1]].type(torch.DoubleTensor).view(-1)
        print('Y train difference::', torch.sum(torch.norm(Y_train_final[final_labeled_id_tensor] - onehot(suggested_updated_labels[0:args.removed_count].type(torch.LongTensor), args.num_class).type(torch.DoubleTensor), dim = 1) == 0))
    
    if is_incremental:
        remaining_ids0 = torch.load(full_output_dir + '/' + args.model + '_influence_remaining_ids' + method_flag + '_v' + str(iter_count)  + incremental_suffix + args.resolve_conflict_str)
        
        final_influence_values0 = torch.load(full_output_dir + '/' + args.model + '_influence' + method_flag + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)[:,remaining_ids0]
    else:
        remaining_ids0 = torch.load(full_output_dir + '/' + args.model + '_influence_remaining_ids' + method_flag + '_v' + str(iter_count) + args.resolve_conflict_str)
        
        final_influence_values0 = torch.load(full_output_dir + '/' + args.model + '_influence' + method_flag + '_v' + str(iter_count) + args.resolve_conflict_str)[:,remaining_ids0]
    
    ordered_list2, sorted_train_ids2 = torch.sort(torch.abs(final_influence_values0.view(-1)), descending = False)
    
    suggested_updated_labels2 = sorted_train_ids2/remaining_ids0.shape[0]
    
    print('Y train difference 2::', torch.nonzero((Y_train_full[final_labeled_id_tensor] - suggested_updated_labels2[0:args.removed_count]) == 0).shape)

    
    full_existing_labeled_id_tensor = torch.zeros(X_train.shape[0]).bool()
    
    if existing_labeled_id_tensor is not None and len(existing_labeled_id_tensor) > 0:
        
        full_existing_labeled_id_tensor[existing_labeled_id_tensor.type(torch.long)] = True
        
        print('existing labeled_count::', torch.sum(full_existing_labeled_id_tensor))
    
    full_existing_labeled_id_tensor[final_labeled_id_tensor] = True
    
    if derive_probab_labels:
    
    
        small_dataset_train_extended = models.MyDataset(torch.cat([X_train[full_existing_labeled_id_tensor]], dim = 0), torch.cat([Y_train_full[full_existing_labeled_id_tensor].type(torch.LongTensor)], dim = 0))
        
        print('full labeled_count::', torch.sum(full_existing_labeled_id_tensor), small_dataset_train_extended.lenth, torch.unique(Y_train_full[full_existing_labeled_id_tensor]))

    
        final_unlabeled_id_tensor = torch.tensor(list(set(range(Y_train.shape[0])).difference(set(torch.nonzero(full_existing_labeled_id_tensor).view(-1).tolist()))))
        
        dataset_to_be_labeled = models.MyDataset(X_train[final_unlabeled_id_tensor], Y_train_final[final_unlabeled_id_tensor])
        
        print('dataset to be labeled count::', final_unlabeled_id_tensor.shape[0])
        
        if start:
            remaining_AFs = torch.load(full_output_dir + '/remaining_AFs')
            remaining_AFs_list = []
             
            for i in range(remaining_AFs.shape[0]):
                remaining_AFs_list.append(remaining_AFs[i])
            print(len(remaining_AFs_list), remaining_AFs_list[0].shape)
            LPs, num_class = goggles.compute_LPs(remaining_AFs, final_labeled_id_tensor.view(-1).tolist(),small_dataset_train_extended.labels.view(-1).tolist())
            torch.save(LPs, full_output_dir + '/LPs')
            torch.save(num_class, full_output_dir + '/num_class')
        else:
            LPs = torch.load(full_output_dir + '/LPs')
            num_class = torch.load(full_output_dir + '/num_class')
        
        dataset_to_be_labeled, _, _ = label_remaining_dataset2(args,small_dataset_train_extended, True, dataset_to_be_labeled.data, Y_train_full[final_unlabeled_id_tensor], torch.nonzero(full_existing_labeled_id_tensor).view(-1), final_unlabeled_id_tensor, LPs = LPs, num_class = num_class)
        
        Y_train_final[final_unlabeled_id_tensor] = dataset_to_be_labeled.labels
    
    
    
    
    return Y_train_final, torch.nonzero(full_existing_labeled_id_tensor).view(-1), final_labeled_id_tensor


def sample_ids_with_changed_labels(updated_labels, origin_labels):
    
    if len(updated_labels.shape) >= 2:
    
        ids_with_changed_labels = (torch.norm(updated_labels - origin_labels, dim = 1).view(-1) > label_change_threshold)
    else:
        ids_with_changed_labels = (torch.abs(updated_labels.view(-1) - origin_labels.view(-1)) > label_change_threshold)
    
    ids_with_unchanged_labels = ~ids_with_changed_labels
    
    return ids_with_changed_labels.view(-1), ids_with_unchanged_labels.view(-1)  

def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, is_GPU, device):
 
 
    zero_mat_dim = k#ids.shape[0]

    curr_S_k = torch.cat(list(S_k_list), dim = 0)
          
    curr_Y_k = torch.cat(list(Y_k_list), dim = 0)
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    
    
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))
    
    if is_GPU:
        R_k = torch.triu(S_k_time_Y_k)
         
     
        L_k = S_k_time_Y_k - (R_k).to(device)
    else:
        R_k = torch.triu(S_k_time_Y_k)
         
     
        L_k = S_k_time_Y_k - R_k
    
#     if is_GPU:
#         
#         R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
#         
#         L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
#         
#     else:
#         R_k = np.triu(S_k_time_Y_k.numpy())
#     
#         L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat



def compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec, is_GPU, device):
    
    
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device =device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
     
    
    torch.mm(curr_Y_k, v_vec, out = p_mat[0:zero_mat_dim])
    
#     torch.sum(curr_S_k_transpose*v_vec*sigma_k, dim = 1, out = p_mat[zero_mat_dim:zero_mat_dim*2])
    
    
    torch.mm(curr_S_k, v_vec*sigma_k, out = p_mat[zero_mat_dim:zero_mat_dim*2])

#     torch.mm(mat, p_mat, out = p_mat)

    p_mat = torch.mm(mat, p_mat)
    
#     print(curr_Y_k_transpose.shape, curr_S_k_transpose.shape, v_vec.shape)
    
    approx_prod = sigma_k*v_vec
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
#     print((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1)).shape)
#     
#     print((torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1)).shape)
#     
#     print(approx_prod.shape)
    
    
    
#     approx_prod -= torch.sum((torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1))*p_mat.view(1, -1), dim = 1).view(-1,1)
    
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    
    
#     approx_prod -= torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
    return approx_prod


def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, v_vec, period, is_GPU, device):
 
 
#     t3  = time.time()
    
#     period_num = int(i/period)
#     
#     
#     ids = torch.tensor(range(m)).view(-1)
#     
#     if period_num > 0:
#         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
# #     else:
# #         ids = torch.cat([ids, period*torch.tensor(range(period_num + 1))], dim = 0)
#     ids = ids - 1
#     
#     ids = ids[ids >= 0]
#     
#     if ids.shape[0] > k:
#         ids = ids[-k:]
    
#     if i-k >= 1:
#         lb = i-k
#         
#         zero_mat_dim = ids.shape[0] + k
#         
#     else:
#         lb = 1
#         
#         zero_mat_dim = ids.shape[0] + i-1
    zero_mat_dim = k#ids.shape[0]
    
    
    
#     curr_S_k = torch.cat([S_k_list[:, ids],S_k_list[:,lb:i]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids],Y_k_list[:,lb:i]], dim=1)

#     print(ids)

#     curr_S_k = torch.cat([S_k_list[:, ids]], dim=1) 
#           
#     curr_Y_k = torch.cat([Y_k_list[:, ids]], dim=1)

#     curr_S_k = S_k_list[:, ids]
#           
#     curr_Y_k = Y_k_list[:, ids]
    
    curr_S_k = torch.t(torch.cat(list(S_k_list), dim = 0))
          
    curr_Y_k = torch.t(torch.cat(list(Y_k_list), dim = 0))
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
        
    else:
        R_k = np.triu(S_k_time_Y_k.numpy())
    
        L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    
    D_k_diag = torch.diag(S_k_time_Y_k)
    
    
    sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    
    p_mat[0:zero_mat_dim] = tmp
    
    p_mat[zero_mat_dim:zero_mat_dim*2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim = 1)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 0)
    
    # print(sigma_k, S_k_time_S_k, mat)

    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
        
        
    
    p_mat = torch.mm(inv_mat, p_mat)
    
    
    approx_prod = sigma_k*v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim = 1), p_mat)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod,zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, inv_mat

def get_triu_3D(x):
    row_idx, col_idx = np.tril_indices(x.shape[2], k = -1)
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    
    x[:, row_idx, col_idx] = 0
    return x

def get_diag_3D(x):
    row_idx, col_idx = np.tril_indices(x.shape[2], k = -1)
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    x[:, row_idx, col_idx] = 0
    
    
    row_idx, col_idx = np.triu_indices(x.shape[2], k = -1)
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    x[:, row_idx, col_idx] = 0
    
    
    return x

def prepare_approx_hessian_vec_prod(S_k_list, Y_k_list, k, combined_mat, is_GPU, device):
    t3  = time.time()
    
    
    zero_mat_dim = k#ids.shape[0]
    
    '''r*m'''
    curr_S_k = torch.cat(list(S_k_list), dim = 0)
    '''r*n*m'''
    curr_Y_k = Y_k_list#torch.stack(list(Y_k_list), dim = 0)

    if is_GPU:
        curr_S_k = curr_S_k.to(device)
    
    '''n*r*m'''
    curr_S_k = curr_S_k.view(1, curr_S_k.shape[0], curr_S_k.shape[1]).repeat(curr_Y_k.shape[1], 1,1)
    
    '''n*m*r'''
    curr_Y_k = torch.transpose(torch.transpose(curr_Y_k, 0, 1), 1, 2)
    
    S_k_time_Y_k = torch.bmm(curr_S_k, curr_Y_k)#torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    '''n*r*r'''
    S_k_time_S_k = torch.bmm(curr_S_k, torch.transpose(curr_S_k, 1, 2))#torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = get_triu_3D(S_k_time_Y_k.clone())        
        L_k = S_k_time_Y_k - R_k.to(device)
        
    else:
        R_k = torch.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - R_k
    
    '''n*r*r'''
    D_k_diag = torch.tril(torch.triu(S_k_time_Y_k))
    '''n*1*1'''
    sigma_k = torch.bmm(curr_Y_k[:,:,-1].view(curr_Y_k.shape[0], 1, curr_Y_k.shape[1]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1))/(torch.bmm(curr_S_k[:,-1].view(curr_S_k.shape[0], 1, curr_S_k.shape[2]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1)))
    
#     if is_GPU:
#         p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double, device = device)
#     else:
#         p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double)
    
    '''nr*1'''
    upper_mat = torch.cat([-D_k_diag, torch.transpose(L_k, 1, 2)], dim = 2)
    
    lower_mat = torch.cat([L_k, sigma_k.view(sigma_k.shape[0], 1, 1)*S_k_time_S_k], dim = 2)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 1)
    
    '''n*2r*2r'''
    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
    
    
    
#     p_mat = torch.bmm(inv_mat, p_mat)
    
    
#     approx_prod = sigma_k.view(sigma_k.shape[0], 1)*v_vec.view(1, -1)
    
    
    combined_mat[0:curr_Y_k.shape[0],:,0:k] = curr_Y_k
    
    res = sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)
    
    combined_mat[0:curr_Y_k.shape[0],:,k:2*k] = res 
    
    
#     print(torch.norm(combined_mat2 - combined_mat[0:curr_Y_k.shape[0]]))
    ''''''
    
#     approx_prod2 = torch.bmm(combined_mat[0:curr_Y_k.shape[0]], p_mat).view(sigma_k.shape[0], -1)
    
#     approx_prod2 = torch.sum(combined_mat[0:curr_Y_k.shape[0]]*p_mat.view(p_mat.shape[0], 1, p_mat.shape[1]), dim = 2)
    
    
#     print(approx_prod2 - approx_prod2_2)
    
#     approx_prod -= approx_prod2
    t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat, combined_mat

# def cal_approx_hessian_vec_prod0_3_sample_wise_incremental(curr_S_k, curr_Y_k, zero_mat_dim, sigma_k, inv_mat, v_vec, combined_mat, is_GPU, device):
#     if is_GPU:
#         p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double, device = device)
#     else:
#         p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double)
#     
#     '''nr*1'''
#     tmp = torch.mm(torch.transpose(curr_Y_k, 1, 2).reshape(curr_Y_k.shape[0]*curr_Y_k.shape[2], curr_Y_k.shape[1]), v_vec.view(-1,1))
#     
#     p_mat[:,0:zero_mat_dim,0] = tmp.view(curr_Y_k.shape[0], curr_Y_k.shape[2])
#     
#     p_mat[:,zero_mat_dim:zero_mat_dim*2,0] = torch.mm(curr_S_k.view(curr_S_k.shape[0]*curr_S_k.shape[1],curr_S_k.shape[2]), v_vec.view(-1,1)).view(curr_S_k.shape[0], curr_S_k.shape[1])*sigma_k.view(curr_S_k.shape[0], 1)
# 
#     p_mat = torch.bmm(inv_mat, p_mat)
#     
#     
#     approx_prod = sigma_k.view(sigma_k.shape[0], 1)*v_vec.view(1, -1)
#     
#     
#     
# #     combined_mat2 = torch.cat([curr_Y_k, sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)], dim = 2)
#     
#     
#     
# #     combined_mat[0:curr_Y_k.shape[0],:,0:zero_mat_dim] = curr_Y_k
# #     
# #     res = sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)
# #     
# #     combined_mat[0:curr_Y_k.shape[0],:,zero_mat_dim:2*zero_mat_dim] = res 
#     
#     
# #     print(torch.norm(combined_mat2 - combined_mat[0:curr_Y_k.shape[0]]))
#     ''''''
#     
#     approx_prod2 = torch.bmm(combined_mat[0:curr_Y_k.shape[0]], p_mat).view(sigma_k.shape[0], -1)
#     
# #     approx_prod2 = torch.sum(combined_mat[0:curr_Y_k.shape[0]]*p_mat.view(p_mat.shape[0], 1, p_mat.shape[1]), dim = 2)
#     
#     
# #     print(approx_prod2 - approx_prod2_2)
#     
#     approx_prod -= approx_prod2

def cal_approx_hessian_vec_prod0_3_sample_wise(S_k_list, Y_k_list, k, v_vec, combined_mat, is_GPU, device):
 
    t3  = time.time()
    
    
    zero_mat_dim = k#ids.shape[0]
    
    '''r*m'''
    curr_S_k = torch.cat(list(S_k_list), dim = 0)
    '''r*n*m'''
    curr_Y_k = Y_k_list#torch.stack(list(Y_k_list), dim = 0)

    if is_GPU:
        curr_S_k = curr_S_k.to(device)
    
#     curr_S_k = S_k_list[:,k:m] 
#          
#     curr_Y_k = Y_k_list[:,k:m] 
    '''n*r*m'''
    curr_S_k = curr_S_k.view(1, curr_S_k.shape[0], curr_S_k.shape[1]).repeat(curr_Y_k.shape[1], 1,1)
    
    
    '''n*m*r'''
    curr_Y_k = torch.transpose(torch.transpose(curr_Y_k, 0, 1), 1, 2)
    
    S_k_time_Y_k = torch.bmm(curr_S_k, curr_Y_k)#torch.mm(torch.t(curr_S_k), curr_Y_k)
    
    '''n*r*r'''
    S_k_time_S_k = torch.bmm(curr_S_k, torch.transpose(curr_S_k, 1, 2))#torch.mm(torch.t(curr_S_k), curr_S_k)
    
    
    if is_GPU:
        
        R_k = get_triu_3D(S_k_time_Y_k.clone())        
        
#         R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
        
        L_k = S_k_time_Y_k - R_k.to(device)
        
    else:
#         R_k = np.triu(S_k_time_Y_k.numpy())
#         R_k2 = get_triu_3D(S_k_time_Y_k.clone())
        R_k = torch.triu(S_k_time_Y_k)
#         print(torch.norm(R_k2 - R_k))
    
        L_k = S_k_time_Y_k - R_k
    
#     D_k_diag = torch.diag(S_k_time_Y_k)
    '''n*r*r'''
#     D_k_diag2 = get_diag_3D(S_k_time_Y_k.clone())
    
    D_k_diag = torch.tril(torch.triu(S_k_time_Y_k))
    
#     print(torch.norm(D_k_diag[0] - torch.diag(torch.diag(S_k_time_Y_k[0]))))
    
#     sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    '''n*1*1'''
    sigma_k = torch.bmm(curr_Y_k[:,:,-1].view(curr_Y_k.shape[0], 1, curr_Y_k.shape[1]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1))/(torch.bmm(curr_S_k[:,-1].view(curr_S_k.shape[0], 1, curr_S_k.shape[2]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1)))
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    if is_GPU:
        p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double)
    
    '''nr*1'''
    tmp = torch.mm(torch.transpose(curr_Y_k, 1, 2).reshape(curr_Y_k.shape[0]*curr_Y_k.shape[2], curr_Y_k.shape[1]), v_vec.view(-1,1))
    
    p_mat[:,0:zero_mat_dim,0] = tmp.view(curr_Y_k.shape[0], curr_Y_k.shape[2])
    
    p_mat[:,zero_mat_dim:zero_mat_dim*2,0] = torch.mm(curr_S_k.view(curr_S_k.shape[0]*curr_S_k.shape[1],curr_S_k.shape[2]), v_vec.view(-1,1)).view(curr_S_k.shape[0], curr_S_k.shape[1])*sigma_k.view(curr_S_k.shape[0], 1)
    
    
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
#     upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
    upper_mat = torch.cat([-D_k_diag, torch.transpose(L_k, 1, 2)], dim = 2)
    
    lower_mat = torch.cat([L_k, sigma_k.view(sigma_k.shape[0], 1, 1)*S_k_time_S_k], dim = 2)
    
    mat = torch.cat([upper_mat, lower_mat], dim = 1)
    
    '''n*2r*2r'''
    mat = np.linalg.inv(mat.cpu().numpy())
        
    inv_mat = torch.from_numpy(mat)
    
    if is_GPU:
        
        inv_mat = inv_mat.to(device)
    
    
    
    p_mat = torch.bmm(inv_mat, p_mat)
    
    
    approx_prod = sigma_k.view(sigma_k.shape[0], 1)*v_vec.view(1, -1)
    
    
    
#     combined_mat2 = torch.cat([curr_Y_k, sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)], dim = 2)
    
    
    
    combined_mat[0:curr_Y_k.shape[0],:,0:k] = curr_Y_k
    
    res = sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)
    
    combined_mat[0:curr_Y_k.shape[0],:,k:2*k] = res 
    
    
#     print(torch.norm(combined_mat2 - combined_mat[0:curr_Y_k.shape[0]]))
    ''''''
    
    approx_prod2 = torch.bmm(combined_mat[0:curr_Y_k.shape[0]], p_mat).view(sigma_k.shape[0], -1)
    
#     approx_prod2 = torch.sum(combined_mat[0:curr_Y_k.shape[0]]*p_mat.view(p_mat.shape[0], 1, p_mat.shape[1]), dim = 2)
    
    
#     print(approx_prod2 - approx_prod2_2)
    
    approx_prod -= approx_prod2
    t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod, zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat, t4 - t3


def cal_approx_hessian_vec_prod0_3_sample_wise_incremental(zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat, curr_combined_mat, v_vec, is_GPU, device):
 
 
#     t3  = time.time()
    
#     zero_mat_dim = k#ids.shape[0]
#     
#     '''r*m'''
#     curr_S_k = torch.cat(list(S_k_list), dim = 0)
#     '''r*n*m'''
#     curr_Y_k = torch.stack(list(Y_k_list), dim = 0)
#     
# #     curr_S_k = S_k_list[:,k:m] 
# #          
# #     curr_Y_k = Y_k_list[:,k:m] 
#     '''n*r*m'''rr_s_k
#     curr_S_k = curr_S_k.view(1, curr_S_k.shape[0], curr_S_k.shape[1]).repeat(curr_Y_k.shape[1], 1,1)
#     
#     '''n*m*r'''
#     curr_Y_k = torch.transpose(torch.transpose(curr_Y_k, 0, 1), 1, 2)
#     
#     S_k_time_Y_k = torch.bmm(curr_S_k, curr_Y_k)#torch.mm(torch.t(curr_S_k), curr_Y_k)
#     
#     '''n*r*r'''
#     S_k_time_S_k = torch.bmm(curr_S_k, torch.transpose(curr_S_k, 1, 2))#torch.mm(torch.t(curr_S_k), curr_S_k)
#     
#     
#     if is_GPU:
#         
#         R_k = get_triu_3D(S_k_time_Y_k.clone())        
#         
# #         R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
#         
#         L_k = S_k_time_Y_k - R_k.to(device)
#         
#     else:
# #         R_k = np.triu(S_k_time_Y_k.numpy())
# #         R_k2 = get_triu_3D(S_k_time_Y_k.clone())
#         R_k = torch.triu(S_k_time_Y_k)
# #         print(torch.norm(R_k2 - R_k))
#     
#         L_k = S_k_time_Y_k - R_k
#     
# #     D_k_diag = torch.diag(S_k_time_Y_k)
#     '''n*r*r'''
# #     D_k_diag2 = get_diag_3D(S_k_time_Y_k.clone())
#     
#     D_k_diag = torch.tril(torch.triu(S_k_time_Y_k))
#     
# #     print(torch.norm(D_k_diag[0] - torch.diag(torch.diag(S_k_time_Y_k[0]))))
#     
# #     sigma_k = torch.mm(Y_k_list[-1],torch.t(S_k_list[-1]))/(torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
#     '''n*1*1'''
#     sigma_k = torch.bmm(curr_Y_k[:,:,-1].view(curr_Y_k.shape[0], 1, curr_Y_k.shape[1]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1))/(torch.bmm(curr_S_k[:,-1].view(curr_S_k.shape[0], 1, curr_S_k.shape[2]), curr_S_k[:,-1].view(curr_S_k.shape[0], curr_S_k.shape[2], 1)))
    
    
    
#     interm = sigma_k*S_k_time_S_k + torch.mm(L_k, torch.mm(torch.diag(torch.pow(D_k_diag, -1)), torch.t(L_k)))
#     
#     J_k = torch.from_numpy(np.linalg.cholesky(interm.detach().numpy())).type(torch.DoubleTensor)
    
    
#     v_vec = S_k_list[:,i-1].view(-1,1)#torch.rand([para_num, 1], dtype = torch.double)
    
#         v_vec = torch.rand([para_num, 1], dtype = torch.double)
#     t1 = time.time()
    if is_GPU:
        p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double, device = device)
    else:
        p_mat = torch.zeros([curr_S_k.shape[0], zero_mat_dim*2, 1], dtype = torch.double)
    
    '''nr*1'''
    tmp = torch.mm(torch.transpose(curr_Y_k, 1, 2).reshape(curr_Y_k.shape[0]*curr_Y_k.shape[2], curr_Y_k.shape[1]), v_vec.view(-1,1))
    
    p_mat[:,0:zero_mat_dim,0] = tmp.view(curr_Y_k.shape[0], curr_Y_k.shape[2])
    
    p_mat[:,zero_mat_dim:zero_mat_dim*2,0] = torch.mm(curr_S_k.view(curr_S_k.shape[0]*curr_S_k.shape[1],curr_S_k.shape[2]), v_vec.view(-1,1)).view(curr_S_k.shape[0], curr_S_k.shape[1])*sigma_k.view(curr_S_k.shape[0], 1)
    
#     t2 = time.time()
    
#     p_mat = torch.cat([torch.mm(torch.t(curr_Y_k), v_vec), torch.mm(torch.t(curr_S_k), v_vec)*sigma_k], dim = 0)
    
    
#     D_k_sqr_root = torch.pow(D_k_diag, 0.5)
#     
#     D_k_minus_sqr_root = torch.pow(D_k_diag, -0.5)
#     
#     upper_mat_1 = torch.cat([-torch.diag(D_k_sqr_root), torch.mm(torch.diag(D_k_minus_sqr_root), torch.t(L_k))], dim = 1)
#     
#     lower_mat_1 = torch.cat([torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double), torch.t(J_k)], dim = 1)
#     
#     
#     mat_1 = torch.cat([upper_mat_1, lower_mat_1], dim = 0)
#     
#     
#     upper_mat_2 = torch.cat([torch.diag(D_k_sqr_root), torch.zeros([zero_mat_dim, zero_mat_dim], dtype = torch.double)], dim = 1)
#     
#     lower_mat_2 = torch.cat([-torch.mm(L_k, torch.diag(D_k_minus_sqr_root)), J_k], dim = 1)
#     
#     mat_2 = torch.cat([upper_mat_2, lower_mat_2], dim = 0)
    
    
    
    
    
#     upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim = 1)
#     upper_mat = torch.cat([-D_k_diag, torch.transpose(L_k, 1, 2)], dim = 2)
#     
#     lower_mat = torch.cat([L_k, sigma_k.view(sigma_k.shape[0], 1, 1)*S_k_time_S_k], dim = 2)
#     
#     mat = torch.cat([upper_mat, lower_mat], dim = 1)
#     
#     '''n*2r*2r'''
#     mat = np.linalg.inv(mat.cpu().numpy())
#         
#     inv_mat = torch.from_numpy(mat)
#     
#     if is_GPU:
#         
#         inv_mat = inv_mat.to(device)
        
        
    
    p_mat2 = torch.bmm(inv_mat, p_mat)
    
    approx_prod2 = torch.bmm(curr_combined_mat, p_mat2).view(sigma_k.shape[0], -1)
    
#     approx_prod3 = torch.sum(curr_combined_mat*p_mat2.view(p_mat2.shape[0],1,p_mat2.shape[1]), -1).view(sigma_k.shape[0], -1)
    
    ''''''
    approx_prod = sigma_k.view(sigma_k.shape[0], 1)*v_vec.view(1, -1) - approx_prod2#torch.bmm(torch.cat([curr_Y_k, sigma_k.view(sigma_k.shape[0], 1, 1)*torch.transpose(curr_S_k, 1, 2)], dim = 2), p_mat).view(sigma_k.shape[0], -1)
    
    
#     t4  = time.time()
    
    
#     print('time1::', t4 - t3)
#     
#     print('key time::', t2 - t1)
    
    
    return approx_prod


# def model_update_origin(max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate, random_ids_multi_super_iterations, batch_size, criterion, optimizer, regularization_coeff, is_GPU, device):

def compute_para_grad_diff(curr_para, exp_curr_para, curr_grad, exp_curr_grad, origin_para = None):
    print('para diff::', torch.norm(curr_para.cpu() - exp_curr_para.cpu()))
    print('grad diff::', torch.norm(curr_grad.cpu() - exp_curr_grad.cpu()))
    if origin_para is not None:
        print('para update::', torch.norm(origin_para.cpu() - exp_curr_para.cpu()))
    


def model_update_deltagrad(max_epoch, period, length, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate, random_ids_multi_super_iterations, batch_size, criterion, optimizer, regularization_coeff, is_GPU, device, exp_updated_w_list = None, exp_updated_grad_list = None):
    '''function to use deltagrad for incremental updates'''
    
    
    para = list(model.parameters())
    
    
    use_standard_way = False
    
    recorded = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     overhead4 = 0
#     
#     overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    

    updated_w_list = []
    
    updated_grad_list = []
    
    
#     res_para_list = []
#     
#     res_grad_list = []
    
#     t5 = time.time()
    
    '''detect which samples are removed from each mini-batch'''
    
    
#     t6 = time.time()
#     
#     overhead3 += (t6  -t5)
    
    '''main for loop of deltagrad'''
    
    i = 0
    
    t1 = 0
    
    t2 = 0
    
    time1 = 0 
    
#     for k in range(max_epoch):
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
#         random_ids_list = random_ids_list_all_epochs[k]
        
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        
        X = dataset_train.data[random_ids]
#         Y = dataset_train.labels[random_ids]
        
        update_labels_curr_epoch = updated_labels[random_ids]
        
        curr_entry_grad_list_epoch = all_entry_grad_list[k]
        
#         updated_grad = get_entry_grad_with_labels(update_labels_curr_epoch, curr_entry_grad_list)
        
#         for p in range(len(random_ids_list)):
        for p in range(0, dataset_train.lenth, batch_size):
            
#             curr_matched_ids = items[2]        
#             curr_matched_ids = random_ids_list[p]
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
            batch_X = X[j:end_id]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
            
            batch_update_labels = update_labels_curr_epoch[j:end_id]
            
            
            
            if is_GPU:
                batch_X = batch_X.to(device)
#                 batch_Y = batch_Y.to(device)
                batch_update_labels = batch_update_labels.to(device)
            t1 = time.time()
            curr_entry_grad_list = curr_entry_grad_list_epoch[j:end_id]
            
            if is_GPU:
                curr_entry_grad_list = curr_entry_grad_list.to(device)
            
#             learning_rate = learning_rate_all_epochs[i]
            
            
            old_lr = learning_rate    
                
            
            
            
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
            
            
                
            if i< curr_init_epochs or use_standard_way == True:
#                 t7 = time.time()
#                 '''explicitly evaluate the gradient'''
#                 curr_rand_ids = random_ids[j:end_id]
#                 
#                 if not removed_batch_empty_list[i]:
#                 
#                     curr_matched_ids2 = get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids)
#                 else:
#                     curr_matched_ids2 = curr_rand_ids
                
#                 t8 = time.time()
#             
#                 overhead4 += (t8 - t7)
#                 
#                 
#                 t5 = time.time()
                
                set_model_parameters(model, para, device)
                
                compute_derivative_one_more_step(model, batch_X, batch_update_labels, criterion, optimizer)
                
                curr_gradients = get_vectorized_grads(model, device)# get_all_vectorized_parameters1(model.get_all_gradient())
                
                compute_para_grad_diff(get_vectorized_params(model), get_all_vectorized_parameters1(exp_updated_w_list[i]), curr_gradients, exp_updated_grad_list[i], para_list_GPU_tensor[cached_id])
                
#                 t6 = time.time()
# 
#                 overhead3 += (t6 - t5)
                
#                 gradient_remaining = 0
#                 if curr_matched_ids_size > 0:
# #                 if not removed_batch_empty_list[i]:
#                     
# #                     t3 = time.time()
#                     
#                     clear_gradients(model.parameters())
#                         
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                 
#                     gradient_remaining = get_vectorized_grads(model)# get_all_vectorized_parameters1(model.get_all_gradient())     
#                     t4 = time.time()
#                 
#                 
#                     overhead2 += (t4  -t3)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para, device)
                
                    if k>0 or (p > 0 and k == 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_para = prev_para.to(device)
                        
                        curr_s_list = (curr_para - prev_para) + 1e-16
                        
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
#                     gradient_full = (expect_gradients*curr_remaining_id_size + gradient_remaining*curr_matched_ids_size)/(curr_remaining_id_size + curr_matched_ids_size)
                    gradient_full = curr_gradients

                    if k>0 or (p > 0 and k == 0):
                        
#                         prev_grad = updated_grad_list_GPU_tensor[cached_id]
                        prev_grad = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list)
                        
#                         if is_GPU:
#                             prev_grad = prev_grad.to(device)
                        
                        Y_k_list.append(gradient_full - prev_grad + regularization_coeff*curr_s_list+ 1e-16)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    

                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*curr_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del curr_gradients
                    
                    if k>0 or (p > 0 and k == 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
                '''use l-bfgs algorithm to evaluate the gradients'''
                
#                 gradient_dual = None
    
#                 if not removed_batch_empty_list[i]:
                set_model_parameters(model, para)
                
                grad1 = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list) 
                
                prev_para = para_list_GPU_tensor[cached_id]
                
#                 grad1 = updated_grad_list_GPU_tensor[cached_id] + regularization_coeff* #get_entry_grad_with_labels(batch_Y - batch_update_labels, curr_entry_grad_list)
                
                if is_GPU:
                    
                    prev_para = prev_para.to(device)
                grad1 = grad1 + regularization_coeff*prev_para
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                     
#                     gradient_dual = model.get_all_gradient()
                    
                with torch.no_grad():
                
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(para, device) - para_list_GPU_tensor[cached_id].to(device)))
                    
                    
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                    exp_gradient, exp_param = exp_updated_grad_list[i], exp_updated_w_list[i]
                    
                    final_grad = torch.t(grad1.view(vec_para_diff.shape) + hessian_para_prod.view(vec_para_diff.shape))
                    
                    
#                     if gradient_dual is not None:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
#                         
#                     else:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    
                    vec_para = update_para_final2(para, final_grad, learning_rate, regularization_coeff, exp_gradient, get_all_vectorized_parameters1(exp_param), para_list_GPU_tensor[cached_id], is_GPU, device)
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                
                
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            
            
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
#                 updated_grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(updated_grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            
            t2 = time.time()
                
            time1 += (t2 - t1)
            
            id_start = id_end
                        
            
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#     print('overhead4::', overhead4)
#     
#     print('overhead5::', overhead5)
    
    print('time 1::', time1)
    
    set_model_parameters(model, para)
        
    return model

def obtain_remaining_ids(full_out_dir, start_iter_count, end_iter_count, is_incremental):
    
    all_ids_with_unchanged_labels = None
    
    for k in range(start_iter_count, end_iter_count):
        
        if k == 0 or (not is_incremental):
            training_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(k))
        else:
            training_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(k) + incremental_suffix)
        
        if k == start_iter_count:
            
            prev_training_dataset = training_dataset
            all_ids_with_unchanged_labels = torch.ones(training_dataset.data.shape[0]).bool()    
            continue
        
#         torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(k-1))
        
        ids_with_changed_labels, ids_with_unchanged_ids = sample_ids_with_changed_labels(prev_training_dataset.labels, training_dataset.labels)
        
        all_ids_with_unchanged_labels = torch.logical_and(all_ids_with_unchanged_labels,ids_with_unchanged_ids)  
#         all_ids_with_unchanged_labels = torch.nonzero(all_ids_with_unchanged_labels_bool) 
        
        prev_training_dataset = training_dataset
    
    return all_ids_with_unchanged_labels, torch.nonzero(all_ids_with_unchanged_labels).view(-1)
    
#     training_dataset1 = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(k))

def obtain_s_y_list(args, remaining_ids, full_out_dir, is_incremental, m = 3, k = 0):
#         for k in range(iteration_count):

        if k == 0 or (not is_incremental):
        
            w_list = torch.load(full_out_dir + '/w_list_v' + str(k))
             
            grad_list = torch.load(full_out_dir + '/grad_list_v' + str(k))
    
#             full_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(k))

        else:
        
            w_list = torch.load(full_out_dir + '/w_list_v' + str(k)  + incremental_suffix)
             
            grad_list = torch.load(full_out_dir + '/grad_list_v' + str(k) + incremental_suffix)
    
        full_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(k))
        
        if is_incremental:
            w1_list = torch.load(full_out_dir + '/w_list_v' + str(k+1) + incremental_suffix)
             
            grad1_list = torch.load(full_out_dir + '/grad_list_v' + str(k+1) + incremental_suffix)
            
#             full_grad1_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(k+1) + incremental_suffix)
        
        else:
            w1_list = torch.load(full_out_dir + '/w_list_v' + str(k+1))
             
            grad1_list = torch.load(full_out_dir + '/grad_list_v' + str(k+1))
            
        full_grad1_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(k+1))
        
        S_k_list = []
         
        Y_k_list = []
        
        mini_Y_k_list = []
        
        mini_Y_k_list2 = []
         
        for r in range(m):
             
            curr_s_k = get_all_vectorized_parameters1(w1_list[-(m-r)]) - get_all_vectorized_parameters1(w_list[-(m-r)])
             
            S_k_list.append(curr_s_k.view(1,-1))
            
            
#             curr_y_k = (full_grad1_list[-(m-r)].view(1,-1) - full_grad_list[-(m-r)].view(1,-1))[remaining_ids] + args.wd*curr_s_k
            curr_y_k = (full_grad1_list[-(m-r)] - full_grad_list[-(m-r)])[remaining_ids] + args.wd*curr_s_k.view(1, -1)
            
#             Y_k_list.append(grad1_list[-(m-r)].view(1,-1) - grad_list[-(m-r)].view(1,-1) + args.wd*curr_s_k)
            Y_k_list.append(curr_y_k)
            
            curr_mini_y_k = (full_grad1_list[-(m-r)] - full_grad_list[-(m-r)])[0].view(1,-1) + args.wd*curr_s_k.view(1,-1)
            
            mini_Y_k_list.append(curr_mini_y_k)
            
            curr_mini_y_k2 = (grad1_list[-(m-r)] - grad_list[-(m-r)]).view(1,-1) + args.wd*curr_s_k.view(1,-1)
            
            mini_Y_k_list2.append(curr_mini_y_k2)
        
        return S_k_list, Y_k_list, mini_Y_k_list, mini_Y_k_list2, full_grad_list, w_list, grad_list
#
#         w2_list = torch.load(full_out_dir + '/w_list_' + str(k+1))
#         
#         grad2_list = torch.load(full_out_dir + '/grad_list_' + str(k+1))
#         
#         vec_para = get_all_vectorized_parameters1(w2_list[-1], args.device) - get_all_vectorized_parameters1(w_list[-1], args.device)
#         
#         hessian_para_prod_0, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, 0,0, m, vec_para.view(-1,1), 0,args.GPU, args.device)
#             
#         expected_grad_diff = grad2_list[-1] - grad_list[-1] + args.wd*vec_para
#         
#         print(torch.norm(hessian_para_prod_0.view(-1) - expected_grad_diff.view(-1)))

def get_influence_sorted(X, batch_size, is_GPU, device, full_grad_tensors, s_test_vec_tensor, full_data_size, sort = True):
    influence_list = []
    
    for k in range(0, X.shape[0], batch_size):
        end_id = k + batch_size
        
        if end_id >= X.shape[0]:
            end_id = X.shape[0]
        
        curr_grad_tensor = full_grad_tensors[k:end_id]
        
        if is_GPU:
            curr_grad_tensor = curr_grad_tensor.to(device)
        
        curr_influences = -torch.mm(curr_grad_tensor, s_test_vec_tensor.view(-1,1))/full_data_size
        
        influence_list.append(curr_influences.cpu())
        
    influence_list_tensor = torch.cat(influence_list, dim = 0)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])
    if sort:
        ordered_list, sorted_train_ids = torch.sort(influence_list_tensor.view(-1), descending=False)
        
        return influence_list_tensor,ordered_list, sorted_train_ids
    else:
        return influence_list_tensor 

'''X,Y,optimizer, loss_func, batch_size, is_GPU, device, full_grad_tensors0, s_test_vec_tensor, full_data_size, regularization_term = regularization_term'''
def get_influence_sorted2(X, Y, optimizer, loss_func, model, batch_size, is_GPU, device, full_grad_tensors, s_test_vec_tensor, full_data_size, sort = True, regularization_term = None):
    influence_list = []
    
    # for k in range(0, X.shape[0], batch_size):
    for k in range(0, X.shape[0]):
        # end_id = k + batch_size
        #
        # if end_id >= X.shape[0]:
        #     end_id = X.shape[0]
        
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        optimizer.zero_grad()
        
        curr_loss = loss_func(model(curr_x), curr_y)
        
        curr_loss.backward()
        
        curr_model_grad = get_vectorized_grads(model, device)
        
        curr_grad_tensor = curr_model_grad#full_grad_tensors[k:end_id]
        
        if regularization_term is not None:
            curr_grad_tensor += regularization_term
        
        if is_GPU:
            curr_grad_tensor = curr_grad_tensor.to(device)
        
        curr_influences = -torch.mm(curr_grad_tensor, s_test_vec_tensor.view(-1,1))/full_data_size
        
        influence_list.append(curr_influences.cpu())
        
    influence_list_tensor = torch.cat(influence_list, dim = 0)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])
    if sort:
        ordered_list, sorted_train_ids = torch.sort(influence_list_tensor.view(-1), descending=False)
        
        return influence_list_tensor,ordered_list, sorted_train_ids
    else:
        return influence_list_tensor 




def pre_compute_class_grad(X, Y, batch_size, is_GPU, device, full_grad_tensors, y_difference, full_data_size, full_grad_tensors1, full_grad_tensors_multi_class, regular_rate = 0.1, regularization_term = None, train_grad = None):
    # influence_list = []
    #
    # record_time = 0
    
    # s_test_vec_tensor = s_test_vec_tensor.cpu()
    #
    # regularization_term = regularization_term.cpu()
    curr_class_grad_list = []
    
    
    for k in range(0, X.shape[0], batch_size):
        end_id = k + batch_size
        
        if end_id >= X.shape[0]:
            end_id = X.shape[0]
        
        '''bath_size * num_class * m'''
        '''num_class, batch_size, num_class'''
        curr_y_difference = y_difference[:,k:end_id]
        
        '''y_difference:: num_class'''
        '''num_class, batch_size, num_class'''
        if full_grad_tensors_multi_class is None:
        
            curr_grad_tensor = full_grad_tensors[k:end_id]    
            '''batch, num_class'''
#         curr_y = Y[k:end_id]
            '''num_class*batch_size * num_class * m'''
            curr_grad_tensor_full = curr_grad_tensor.view(1, curr_grad_tensor.shape[0], curr_grad_tensor.shape[1], curr_grad_tensor.shape[2]).repeat(y_difference.shape[0], 1, 1, 1)
            
        else:
            curr_grad_tensor_full = full_grad_tensors_multi_class[:,k:end_id]
            
            
        curr_sample_wise_grad = full_grad_tensors1[k:end_id]
        # if is_GPU:
        #     torch.cuda.synchronize(device = device)
        #
        # t1 = time.time()
        
        # print(curr_sample_wise_grad.shape, curr_y_difference.shape, curr_grad_tensor_full.shape)
        
        if is_GPU:
            # curr_grad_tensor = curr_grad_tensor.to(device)
            curr_y_difference = curr_y_difference.to(device)
            curr_grad_tensor_full = curr_grad_tensor_full.to(device)
            curr_sample_wise_grad = curr_sample_wise_grad.to(device)
        
        
#         for p in range(y_difference.shape[0]):
#             '''batch_size, m'''
        
        '''num_class*batch_size *m'''
        
        
        # curr_class_grad0 = torch.bmm(curr_y_difference[:,0].reshape(curr_y_difference.shape[0], 1, curr_y_difference.shape[2]), curr_grad_tensor_full[:,0].reshape(curr_grad_tensor_full.shape[0], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        #
        # curr_class_grad0 = curr_class_grad0.view(curr_y_difference.shape[0], -1) + (1 - regular_rate)*(full_grad_tensors1[k:end_id][0] + regularization_term).view(1, -1)
        
        curr_class_grad = torch.bmm(curr_y_difference.reshape(curr_y_difference.shape[0]*curr_y_difference.shape[1], 1, curr_y_difference.shape[2]), curr_grad_tensor_full.reshape(curr_grad_tensor_full.shape[0]*curr_grad_tensor_full.shape[1], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        
        # print(full_grad_tensors1[k:end_id].device, regularization_term.device, curr_class_grad.device)
        
        if regularization_term is None:
            curr_class_grad = curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(curr_sample_wise_grad).view(1, end_id-k, -1)/full_data_size
        else:
            curr_class_grad = curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(curr_sample_wise_grad + regularization_term).view(1, end_id-k, -1)/full_data_size
        
        if train_grad is not None:
            curr_class_grad = curr_class_grad + train_grad.view(1,-1)
        
#         curr_class_grad = torch.sum(curr_y_difference.view(curr_y_difference.shape[0], curr_y_difference.shape[1], curr_y_difference.shape[2], 1)*curr_grad_tensor_full, 2)
#         torch.bmm(torch.transpose(curr_grad_tensor, 1,2))
        
        '''num_class*batch_size'''
            
        curr_class_grad_list.append(curr_class_grad.cpu())  
        
        # curr_influences = (torch.mm(curr_class_grad.view(-1, curr_class_grad.shape[-1]), s_test_vec_tensor.view(-1,1))).view(y_difference.shape[0], end_id  -k)
        #
        # influence_list.append(curr_influences.cpu())
        
        # if is_GPU:
        #     torch.cuda.synchronize(device = device)
        #
        # t2 = time.time()
        #
        # record_time += (t2 - t1)
#         print('curr_grad_tensor_full shape::', curr_grad_tensor_full.shape, curr_y_difference.shape, curr_class_grad.shape)
    return torch.cat(curr_class_grad_list, 1)

def get_influence_sorted_class_wise(X, Y, batch_size, is_GPU, device, full_grad_tensors, y_difference, s_test_vec_tensor, full_data_size, sort = True, full_grad_tensors1 = None, full_grad_tensors_multi_class = None, regular_rate = 0.1, regularization_term = 0, train_prob_case2 = None, train_grad = None, origin_class_wise_grad_tensor = None):
    influence_list = []
    
    record_time = 0
    
    # s_test_vec_tensor = s_test_vec_tensor.cpu()
    #
    # regularization_term = regularization_term.cpu()
    
    for k in range(0, X.shape[0], batch_size):
        end_id = k + batch_size
        
        if end_id >= X.shape[0]:
            end_id = X.shape[0]
        
        '''bath_size * num_class * m'''
        '''num_class, batch_size, num_class'''
        curr_y_difference = y_difference[:,k:end_id]
        
        '''y_difference:: num_class'''
        '''num_class, batch_size, num_class'''
        if full_grad_tensors_multi_class is None:
        
            curr_grad_tensor = full_grad_tensors[k:end_id]    
            '''batch, num_class'''
#         curr_y = Y[k:end_id]
            '''num_class*batch_size * num_class * m'''
            curr_grad_tensor_full = curr_grad_tensor.view(1, curr_grad_tensor.shape[0], curr_grad_tensor.shape[1], curr_grad_tensor.shape[2]).repeat(y_difference.shape[0], 1, 1, 1)
            
        else:
            curr_grad_tensor_full = full_grad_tensors_multi_class[:,k:end_id]
            
            
        curr_sample_wise_grad = full_grad_tensors1[k:end_id]
        # if is_GPU:
        #     torch.cuda.synchronize(device = device)
        #
        # t1 = time.time()
        
        # print(curr_sample_wise_grad.shape, curr_y_difference.shape, curr_grad_tensor_full.shape)
        
        if is_GPU:
            # curr_grad_tensor = curr_grad_tensor.to(device)
            curr_y_difference = curr_y_difference.to(device)
            curr_grad_tensor_full = curr_grad_tensor_full.to(device)
            curr_sample_wise_grad = curr_sample_wise_grad.to(device)
        
        
#         for p in range(y_difference.shape[0]):
#             '''batch_size, m'''
        
        '''num_class*batch_size *m'''
        
        
        # curr_class_grad0 = torch.bmm(curr_y_difference[:,0].reshape(curr_y_difference.shape[0], 1, curr_y_difference.shape[2]), curr_grad_tensor_full[:,0].reshape(curr_grad_tensor_full.shape[0], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        #
        # curr_class_grad0 = curr_class_grad0.view(curr_y_difference.shape[0], -1) + (1 - regular_rate)*(full_grad_tensors1[k:end_id][0] + regularization_term).view(1, -1)
        
        curr_class_grad = torch.bmm(curr_y_difference.reshape(curr_y_difference.shape[0]*curr_y_difference.shape[1], 1, curr_y_difference.shape[2]), curr_grad_tensor_full.reshape(curr_grad_tensor_full.shape[0]*curr_grad_tensor_full.shape[1], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        
        # print(full_grad_tensors1[k:end_id].device, regularization_term.device, curr_class_grad.device)
        
        if regularization_term is None:
            curr_class_grad = (curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(curr_sample_wise_grad).view(1, end_id-k, -1))/full_data_size
            
        else:
            curr_class_grad = (curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(curr_sample_wise_grad + regularization_term).view(1, end_id-k, -1))/full_data_size
        
        if train_grad is not None:
            curr_class_grad = curr_class_grad + train_grad.view(1,-1)
        
#         curr_class_grad = torch.sum(curr_y_difference.view(curr_y_difference.shape[0], curr_y_difference.shape[1], curr_y_difference.shape[2], 1)*curr_grad_tensor_full, 2)
#         torch.bmm(torch.transpose(curr_grad_tensor, 1,2))
        
        '''num_class*batch_size'''
        curr_influences = (torch.mm(curr_class_grad.view(-1, curr_class_grad.shape[-1]), s_test_vec_tensor.view(-1,1))).view(y_difference.shape[0], end_id  -k)
        
        influence_list.append(curr_influences.cpu())
        
        # if is_GPU:
        #     torch.cuda.synchronize(device = device)
        #
        # t2 = time.time()
        #
        # record_time += (t2 - t1)
#         print('curr_grad_tensor_full shape::', curr_grad_tensor_full.shape, curr_y_difference.shape, curr_class_grad.shape)
        
    influence_list_tensor = torch.cat(influence_list, dim = 1)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])
    print('record time::', record_time)
    
    if train_prob_case2 is not None:
        influence_list_tensor = influence_list_tensor*train_prob_case2.view(1,-1)
    
    if sort:
        ordered_list, sorted_train_ids = torch.sort(influence_list_tensor.view(-1), descending=True)
        
        return influence_list_tensor,ordered_list, sorted_train_ids
    else:
        return influence_list_tensor 

def get_influence_sorted_class_wise2_0(X, Y, batch_size, is_GPU, device, full_grad_tensors, y_difference, s_test_vec_tensor, full_data_size, sort = True, full_grad_tensors1 = None, full_grad_tensors_multi_class = None, regular_rate = 0.1, regularization_term = 0):
    influence_list = []
    
    # for k in range(0, X.shape[0], batch_size):
    for k in range(X.shape[0]):
        # end_id = k + batch_size
        #
        # if end_id >= X.shape[0]:
            # end_id = X.shape[0]
        
        '''bath_size * num_class * m'''
        '''num_class, batch_size, num_class'''
        curr_y_difference = y_difference[:,k]
        
        '''y_difference:: num_class'''
        '''num_class, batch_size, num_class'''
        if full_grad_tensors_multi_class is None:
        
            curr_grad_tensor = full_grad_tensors[k]    
            '''batch, num_class'''
#         curr_y = Y[k:end_id]
            '''num_class*batch_size * num_class * m'''
            curr_grad_tensor_full = curr_grad_tensor.view(1, curr_grad_tensor.shape[0], curr_grad_tensor.shape[1], curr_grad_tensor.shape[2]).repeat(y_difference.shape[0], 1, 1, 1)
            
        else:
            curr_grad_tensor_full = full_grad_tensors_multi_class[:,k]
            
        
#         if is_GPU:
#             curr_grad_tensor = curr_grad_tensor.to(device)
#             curr_y_difference = curr_y_difference.to(device)
#             curr_grad_tensor_full = curr_grad_tensor_full.to(device)
        
        
        
#         for p in range(y_difference.shape[0]):
#             '''batch_size, m'''
        
        '''num_class*batch_size *m'''
        
        
        # curr_class_grad0 = torch.bmm(curr_y_difference[:,0].reshape(curr_y_difference.shape[0], 1, curr_y_difference.shape[2]), curr_grad_tensor_full[:,0].reshape(curr_grad_tensor_full.shape[0], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        #
        # curr_class_grad0 = curr_class_grad0.view(curr_y_difference.shape[0], -1) + (1 - regular_rate)*(full_grad_tensors1[k:end_id][0] + regularization_term).view(1, -1)
        
        
        curr_class_grad = torch.bmm(curr_y_difference.reshape(curr_y_difference.shape[0]*curr_y_difference.shape[1], 1, curr_y_difference.shape[2]), curr_grad_tensor_full.reshape(curr_grad_tensor_full.shape[0]*curr_grad_tensor_full.shape[1], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
        
        # print(full_grad_tensors1[k:end_id].device, regularization_term.device, curr_class_grad.device)
        
        curr_class_grad = curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(full_grad_tensors1[k:k+1] + regularization_term.cpu()).view(1, 1, -1)
        
#         curr_class_grad = torch.sum(curr_y_difference.view(curr_y_difference.shape[0], curr_y_difference.shape[1], curr_y_difference.shape[2], 1)*curr_grad_tensor_full, 2)
#         torch.bmm(torch.transpose(curr_grad_tensor, 1,2))
        
        '''num_class*batch_size'''
        curr_influences = (torch.mm(curr_class_grad.view(-1, curr_class_grad.shape[-1]), s_test_vec_tensor.view(-1,1))/full_data_size).view(y_difference.shape[0], 1)
        
        influence_list.append(curr_influences.cpu())
        
#         print('curr_grad_tensor_full shape::', curr_grad_tensor_full.shape, curr_y_difference.shape, curr_class_grad.shape)
        
    influence_list_tensor = torch.cat(influence_list, dim = 1)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])

    
    
    if sort:
        ordered_list, sorted_train_ids = torch.sort(influence_list_tensor.view(-1), descending=True)
        
        return influence_list_tensor,ordered_list, sorted_train_ids
    else:
        return influence_list_tensor 


def get_influence_sorted_class_wise2(model, loss_func, num_class, optimizer, X, Y, is_GPU, device, y_difference, s_test_vec_tensor, full_data_size, curr_w_tensor, regularization_coeff, regularization_term, sort = True, full_grad_tensors_multi_class = None, regular_rate = 0.1, train_prob_case2 = None, train_grad = None):
    influence_list = []
    
    if is_GPU:
        s_test_vec_tensor = s_test_vec_tensor.to(device)
    
    for k in range(0, X.shape[0]):
        # end_id = k + batch_size
        #
        # if end_id >= X.shape[0]:
            # end_id = X.shape[0]
        print('sample::', k)    
        
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        model_out0 = model(curr_x)
        
        model_out = -F.log_softmax(model_out0)
        
        model_grad_all_classes = []
        
        
        model_grad_all_classes = []
        
        for p in range(num_class):
        
            optimizer.zero_grad()
            
#             print(model_out.shape)
            
            model_out.view(-1)[p].backward(retain_graph = True)
        
            curr_model_grad = get_vectorized_grads(model, device) + regularization_coeff*curr_w_tensor.view(1,-1)

            model_grad_all_classes.append(curr_model_grad.view(-1))
        '''num_class * m'''    
        model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        
        full_model_out = loss_func(model_out0, curr_y)
        
        full_model_out.backward(retain_graph = True)
        
        full_model_grad = get_vectorized_grads(model, device)
        
        '''bath_size * num_class * m'''
        '''num_class, batch_size, num_class'''
        curr_y_difference = y_difference[:,k:k+1]
        
        '''y_difference:: num_class'''
        '''num_class, batch_size, num_class'''
        if full_grad_tensors_multi_class is None:
        
            curr_grad_tensor = model_grad_all_classes_tensor.reshape(1, model_grad_all_classes_tensor.shape[0], model_grad_all_classes_tensor.shape[1])
            '''batch, num_class'''
#         curr_y = Y[k:end_id]
            '''num_class*batch_size * num_class * m'''
            curr_grad_tensor_full = curr_grad_tensor.view(1, curr_grad_tensor.shape[0], curr_grad_tensor.shape[1], curr_grad_tensor.shape[2]).repeat(y_difference.shape[0], 1, 1, 1)
            
        else:
            curr_grad_tensor_full = full_grad_tensors_multi_class[:,k:k+1]
            
        
#         if is_GPU:
#             curr_grad_tensor = curr_grad_tensor.to(device)
#             curr_y_difference = curr_y_difference.to(device)
#             curr_grad_tensor_full = curr_grad_tensor_full.to(device)
        
        
        
#         for p in range(y_difference.shape[0]):
#             '''batch_size, m'''
        
        '''num_class*batch_size *m'''
        
        if is_GPU:
            curr_y_difference = curr_y_difference.to(device)
        
        curr_class_grad = torch.bmm(curr_y_difference.reshape(curr_y_difference.shape[0]*curr_y_difference.shape[1], 1, curr_y_difference.shape[2]), curr_grad_tensor_full.reshape(curr_grad_tensor_full.shape[0]*curr_grad_tensor_full.shape[1], curr_grad_tensor_full.shape[2], curr_grad_tensor_full.shape[3]))
#         curr_class_grad = torch.sum(curr_y_difference.view(curr_y_difference.shape[0], curr_y_difference.shape[1], curr_y_difference.shape[2], 1)*curr_grad_tensor_full, 2)
#         torch.bmm(torch.transpose(curr_grad_tensor, 1,2))
        
        curr_class_grad = (curr_class_grad.view(curr_y_difference.shape[0], curr_y_difference.shape[1], -1) + (1 - regular_rate)*(full_model_grad + regularization_term).view(1, 1, -1))/full_data_size
        
        if train_grad is not None:
            curr_class_grad = curr_class_grad + train_grad.view(1,-1)
        
        '''num_class*batch_size'''
        curr_influences = (torch.mm(curr_class_grad.view(-1, curr_class_grad.shape[-1]), s_test_vec_tensor.view(-1,1))).view(y_difference.shape[0], 1)
        
        influence_list.append(curr_influences.cpu())
        
#         print('curr_grad_tensor_full shape::', curr_grad_tensor_full.shape, curr_y_difference.shape, curr_class_grad.shape)
        
    influence_list_tensor = torch.cat(influence_list, dim = 1)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])

    if train_prob_case2 is not None:
        influence_list_tensor = influence_list_tensor.view(-1)*train_prob_case2.view(-1)
    
    if sort:
        ordered_list, sorted_train_ids = torch.sort(influence_list_tensor.view(-1), descending=True)
        
        return influence_list_tensor,ordered_list, sorted_train_ids
    else:
        return influence_list_tensor 


def get_influence_sorted_ac(X, batch_size, is_GPU, device, full_grad_tensors, average_y_change, full_data_size, sort = True):
    
    full_influence_tensor = torch.norm(full_grad_tensors/full_data_size + average_y_change*X, dim = 1)
    
    
#     influence_list = []
#     
#     for k in range(0, X.shape[0], batch_size):
#         end_id = k + batch_size
#         
#         if end_id >= X.shape[0]:
#             end_id = X.shape[0]
#         
#         curr_grad_tensor = full_grad_tensors[k:end_id]
#         
#         if is_GPU:
#             curr_grad_tensor = curr_grad_tensor.to(device)
#         
#         curr_influences = -torch.mm(curr_grad_tensor, s_test_vec_tensor.view(-1,1))/full_data_size
#         
#         influence_list.append(curr_influences.cpu())
#         
#     influence_list_tensor = torch.cat(influence_list, dim = 0)
    
#     for i in range(X.shape[0]):
#         print(influence_list_tensor[i] - influences2[i])
    if sort:
        ordered_list, sorted_train_ids = torch.sort(full_influence_tensor.view(-1), descending=True)
        
        return full_influence_tensor,ordered_list, sorted_train_ids
    else:
        return full_influence_tensor 

def obtain_remaining_ids_incremental(args, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, model_diff, origin_sample_wise_grad, max_mu_list_tensor, full_removed_id_tensor_0, full_existing_labeled_id_tensor, full_origin_influence_list_tensor, prev_s_test_vec):
    
    X = dataset_train.data
    
    t1 = time.time()
    
    s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, 0, is_GPU, device, full_out_dir, dataset_train, valid_dataset, model, loss_func)
    
    t2 = time.time()
    
    s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    t4 = time.time()
    influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor, full_data_size, sort = False)
    
#     influence_list_tensor0 = torch.mm(origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor.view(-1,1))
    
    t3 = time.time() 
#     prev_influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], prev_s_test_vec, full_data_size, sort = False)
    
    full_influence_list_tensor0 = torch.zeros([X.shape[0]], dtype = influence_list_tensor0.dtype)
    
    full_influence_list_tensor0[full_removed_id_tensor_0] = influence_list_tensor0.view(-1)
    
#     delta_influence_list_tensor = torch.norm(model_diff)*torch.norm(s_test_vec_tensor)/full_data_size*max_mu_list_tensor
    
    delta_influence_list_tensor = 2*torch.norm(torch.dot((model_diff).view(-1), (s_test_vec_tensor).view(-1)))/full_data_size*max_mu_list_tensor
    
    
    
    remaining_id_tensor = torch.zeros(dataset_train.data.shape[0]).bool()
    
    remaining_id_tensor[full_removed_id_tensor_0.view(-1)] = True
    
    remaining_id_tensor[full_existing_labeled_id_tensor.view(-1)] = False
    
    remaining_removed_id0 = torch.nonzero(remaining_id_tensor).view(-1)
    
#     remaining_removed_id0 = full_removed_id_tensor_0[~(torch.sum(full_removed_id_tensor_0.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
#        
#     
#     print(torch.norm(torch.sort(remaining_removed_id1)[0].type(torch.double) - torch.sort(remaining_removed_id0)[0].type(torch.double)))
#             remaining_removed_id1 = full_removed_id_tensor1[~(torch.sum(full_removed_id_tensor1.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
    
    sorted_influence_list_tensor,sorted_influence_sample_ids = torch.sort((full_influence_list_tensor0+delta_influence_list_tensor)[remaining_removed_id0].view(-1))
    
    sorted_remaining_removed_id0 = remaining_removed_id0[sorted_influence_sample_ids]
    
    id_set1 = remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0-delta_influence_list_tensor)[remaining_removed_id0[args.removed_count:]].view(-1)) < ((full_influence_list_tensor0 + delta_influence_list_tensor)[remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0-delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count:]].view(-1)) < ((full_influence_list_tensor0 + delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    final_remaining_ids= torch.cat([sorted_remaining_removed_id0[0:args.removed_count].view(-1), sorted_id_set1], dim = 0)
    
    final_remaining_ids = torch.tensor(list(set(final_remaining_ids.tolist()).union(set(remaining_removed_id0[0:args.removed_count].view(-1).tolist()))))
    
    

    print('compute_hessian time::', t2 - t1)
    
    print('other time::', t3 - t4)
    
    return final_remaining_ids, s_test_vec_tensor


def obtain_remaining_ids_incremental_class_wise2(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, model_diff, origin_sample_wise_grad, origin_sample_wise_grad_full, origin_full_sample_wise_grad_tensors, max_mu_list_tensor, max_sample_mu_list_tensor, full_removed_id_tensor_0, full_existing_labeled_id_tensor, full_origin_influence_list_tensor, prev_s_test_vec, y_difference, regular_rate = 0.0, r_weight = None, origin_class_wise_grad_tensor = None, s_test_vec_tensor = None):
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    t1 = time.time()
    
    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)
    
    regularization_term = args.wd*curr_w_tensor
    
    '''args, batch_size, regularization_term, is_GPU, device, full_out_dir, training_dataset, valid_dataset, model, loss_func, optimizer, learning_rate = 0.0002, r_weight = None'''
    # s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, 0, is_GPU, device, full_out_dir, dataset_train, valid_dataset, model, loss_func. optimizer, r_weight = r_weight)
    
    if s_test_vec_tensor is None:
    
        s_test_vec = evaluate_influence_function_repetitive_incremental2(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = args.derived_lr, r_weight = r_weight)
        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    # if args.GPU:
    #         torch.cuda.synchronize(device = device)
    t2 = time.time()
    
    
    # if args.GPU:
    #         torch.cuda.synchronize(device = device)
    t4 = time.time()
#     influence_list_tensor0  = get_influence_sorted_class_wise(X[full_removed_id_tensor_0], Y[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], y_difference[:,full_removed_id_tensor_0], s_test_vec_tensor, full_data_size, sort = False, full_grad_tensors_multi_class = origin_sample_wise_grad_full[:,full_removed_id_tensor_0])
    
    if origin_class_wise_grad_tensor is not None:
        
        # torch.bmm(origin_class_wise_grad_tensor)
        
        influence_list_tensor0_2 = []
        for k in range(origin_class_wise_grad_tensor.shape[0]):
            influence_list_tensor0_2.append(torch.mm(origin_class_wise_grad_tensor[k][full_removed_id_tensor_0], s_test_vec_tensor.view(-1,1).cpu()))
        
        influence_list_tensor0_2_tensor = torch.stack(influence_list_tensor0_2, 0)
        
        influence_list_tensor0 = influence_list_tensor0_2_tensor.view(influence_list_tensor0_2_tensor.shape[0], influence_list_tensor0_2_tensor.shape[1])
    else:
        # influence_list_tensor0_2 = torch.mm(origin_class_wise_grad_tensor, s_test_vec_tensor.cpu().view(-1,1)).view(y_difference.shape[0], X.shape[0])[:,full_removed_id_tensor_0]
        # origin_class_wise_grad_tensor_1 = pre_compute_class_grad(X, Y, batch_size, is_GPU, device, origin_sample_wise_grad, y_difference, X.shape[0], full_grad_tensors1 = origin_full_sample_wise_grad_tensors, full_grad_tensors_multi_class = origin_sample_wise_grad_full, regular_rate = regular_rate, regularization_term = None)
        '''                                                        X, Y, batch_size, is_GPU, device,                                                 full_grad_tensors, y_difference, s_test_vec_tensor, full_data_size, sort = True, full_grad_tensors1 = None, full_grad_tensors_multi_class = None, regular_rate = 0.1, regularization_term = 0, train_prob_case2 = None, train_grad = None'''
        influence_list_tensor0  = get_influence_sorted_class_wise(X[full_removed_id_tensor_0], Y[full_removed_id_tensor_0], full_removed_id_tensor_0.shape[0], is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], y_difference[:,full_removed_id_tensor_0], s_test_vec_tensor, full_data_size, sort = False, full_grad_tensors1 = origin_full_sample_wise_grad_tensors[full_removed_id_tensor_0], full_grad_tensors_multi_class = origin_sample_wise_grad_full[:,full_removed_id_tensor_0], regular_rate = regular_rate, regularization_term = None)
    
#     influence_list_tensor0 = torch.mm(origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor.view(-1,1))
    # if args.GPU:
    #         torch.cuda.synchronize(device = device)
    t3 = time.time() 
#     prev_influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], prev_s_test_vec, full_data_size, sort = False)
    
    full_influence_list_tensor0 = torch.zeros([num_class, X.shape[0]], dtype = influence_list_tensor0.dtype)
    
    full_influence_list_tensor0[:,full_removed_id_tensor_0] = influence_list_tensor0
    
#     delta_influence_list_tensor = torch.norm(model_diff)*torch.norm(s_test_vec_tensor)/full_data_size*max_mu_list_tensor
    
#     delta_influence_list_tensor = 0.5*(torch.norm((model_diff).view(-1) + (s_test_vec_tensor).view(-1))**2)/full_data_size*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference), dim = -1))
    
    if is_GPU:
        model_diff = model_diff.to(device)
        
        # max_mu_list_tensor = max_mu_list_tensor.to(device)
        
        # y_difference = y_difference.to(device)
    
    model_diff_s_vec_dot = (torch.norm((model_diff).view(-1)*torch.norm(s_test_vec_tensor).view(-1))).cpu()
        
    model_diff_s_vec_dot2 = (torch.dot((model_diff).view(-1), s_test_vec_tensor.view(-1))).cpu()
    
    delta_influence_list_tensor_upper0 = model_diff_s_vec_dot*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference)*(y_difference >= 0).type(torch.double), dim = -1))
     
    delta_influence_list_tensor_lower0 = model_diff_s_vec_dot*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference)*(y_difference < 0).type(torch.double), dim = -1))
    
    
    delta_influence_list_tensor_upper = 0.5*model_diff_s_vec_dot*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference).type(torch.double), dim = -1) + (1-regular_rate)*max_sample_mu_list_tensor)
     
    delta_influence_list_tensor_lower = 0.5*model_diff_s_vec_dot*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference).type(torch.double), dim = -1) + (1-regular_rate)*max_sample_mu_list_tensor)
    
    delta_influence_list_tensor_lower = delta_influence_list_tensor_lower/full_data_size
    
    delta_influence_list_tensor_upper = delta_influence_list_tensor_upper/full_data_size
    
    extra_terms = 0.5*model_diff_s_vec_dot2*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*y_difference.type(torch.double), dim = -1) + (1-regular_rate)*max_sample_mu_list_tensor)
    
    extra_terms = extra_terms/full_data_size
    
    delta_influence_list_tensor_upper += extra_terms
    
    delta_influence_list_tensor_upper = delta_influence_list_tensor_upper
    
    delta_influence_list_tensor_lower = extra_terms - delta_influence_list_tensor_lower
    
    
    delta_influence_list_tensor_lower = delta_influence_list_tensor_lower
    
    # regular_term_upper = (1-regular_rate)*0.5*torch.dot((model_diff).view(-1), s_test_vec_tensor.view(-1))/full_data_size*max_sample_mu_list_tensor
    
    
#     delta_influence_list_tensor_upper = 2*torch.abs(torch.dot((model_diff).view(-1),(s_test_vec_tensor).view(-1))/full_data_size)*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference).type(torch.double), dim = -1))
#     
#     delta_influence_list_tensor_lower = 2*torch.abs(torch.dot((model_diff).view(-1),(s_test_vec_tensor).view(-1))/full_data_size)*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference).type(torch.double), dim = -1))
    
    
#     delta_influence_list_tensor = 2*torch.norm(torch.dot((model_diff).view(-1), (s_test_vec_tensor).view(-1)))/full_data_size*max_mu_list_tensor
    
#     delta_influence_list_tensor = 0.5*(torch.norm((model_diff).view(-1) + (s_test_vec_tensor).view(-1)/full_data_size)**2)*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference)*(y_difference >= 0).type(torch.double), dim = -1))
    
#     delta_influence_list_tensor = torch.t(delta_influence_list_tensor)
    
    remaining_id_tensor = torch.zeros(dataset_train.data.shape[0]).bool()
    
    remaining_id_tensor[full_removed_id_tensor_0.view(-1)] = True
    
    remaining_id_tensor[full_existing_labeled_id_tensor.view(-1)] = False
    
    remaining_removed_id0 = torch.nonzero(remaining_id_tensor).view(-1)
    
#     remaining_removed_id0 = full_removed_id_tensor_0[~(torch.sum(full_removed_id_tensor_0.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
#        
#     
#     print(torch.norm(torch.sort(remaining_removed_id1)[0].type(torch.double) - torch.sort(remaining_removed_id0)[0].type(torch.double)))
#             remaining_removed_id1 = full_removed_id_tensor1[~(torch.sum(full_removed_id_tensor1.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
    
    sorted_influence_list_tensor,sorted_influence_sample_ids0 = torch.sort((full_influence_list_tensor0)[:,remaining_removed_id0].view(-1), descending=True)
    
    sorted_influence_sample_ids = sorted_influence_sample_ids0%remaining_removed_id0.shape[0]
    
    sorted_remaining_removed_id0 = remaining_removed_id0[sorted_influence_sample_ids]
    
#     id_set1 = remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0+delta_influence_list_tensor)[remaining_removed_id0[args.removed_count:]].view(-1)) > ((full_influence_list_tensor0 - delta_influence_list_tensor)[remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    lower_bound_tensor = ((full_influence_list_tensor0+delta_influence_list_tensor_lower)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][args.removed_count:].view(-1)
    
    upper_bound_value = torch.max(((full_influence_list_tensor0+delta_influence_list_tensor_upper)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][0:args.removed_count])
    
    
    upper_bound_tensor = ((full_influence_list_tensor0+delta_influence_list_tensor_upper)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][args.removed_count:].view(-1)
    
    lower_bound_value = torch.min(((full_influence_list_tensor0+delta_influence_list_tensor_lower)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][0:args.removed_count])
    
    
#     sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][([sorted_remaining_removed_id0[args.removed_count:]].view(-1)) < ((full_influence_list_tensor0 + delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    # sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][lower_bound_tensor < upper_bound_value]
    
    sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][lower_bound_value < upper_bound_tensor]
    
    final_remaining_ids= torch.cat([sorted_remaining_removed_id0[0:args.removed_count].view(-1), sorted_id_set1], dim = 0)
    
    # final_remaining_ids = torch.tensor(list(set(final_remaining_ids.tolist()).union(set(remaining_removed_id0[0:args.removed_count].view(-1).tolist()))))
    
    

    print('compute_hessian time::', t2 - t1)
    
    print('other time::', t3 - t4)
    
    return final_remaining_ids, s_test_vec_tensor, full_influence_list_tensor0

def obtain_remaining_ids_incremental_class_wise(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, model_diff, origin_sample_wise_grad, origin_sample_wise_grad_full, max_mu_list_tensor, full_removed_id_tensor_0, full_existing_labeled_id_tensor, full_origin_influence_list_tensor, prev_s_test_vec, y_difference):
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    t1 = time.time()
    
    s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, 0, is_GPU, device, full_out_dir, dataset_train, valid_dataset, model, loss_func)
    
    t2 = time.time()
    
    s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    t4 = time.time()
    influence_list_tensor0  = get_influence_sorted_class_wise(X[full_removed_id_tensor_0], Y[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], y_difference[:,full_removed_id_tensor_0], s_test_vec_tensor, full_data_size, sort = False, full_grad_tensors_multi_class = origin_sample_wise_grad_full[:,full_removed_id_tensor_0])
    
#     influence_list_tensor0 = torch.mm(origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor.view(-1,1))
    
    t3 = time.time() 
#     prev_influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], prev_s_test_vec, full_data_size, sort = False)
    
    full_influence_list_tensor0 = torch.zeros([num_class, X.shape[0]], dtype = influence_list_tensor0.dtype)
    
    full_influence_list_tensor0[:,full_removed_id_tensor_0] = influence_list_tensor0
    
#     delta_influence_list_tensor = torch.norm(model_diff)*torch.norm(s_test_vec_tensor)/full_data_size*max_mu_list_tensor
    
#     delta_influence_list_tensor = 0.5*(torch.norm((model_diff).view(-1) + (s_test_vec_tensor).view(-1))**2)/full_data_size*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference), dim = -1))
    
#     delta_influence_list_tensor = 0.5*(torch.norm((model_diff).view(-1) + (s_test_vec_tensor).view(-1)/full_data_size)**2)*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference), dim = -1))
    
    delta_influence_list_tensor = 0.5*(torch.norm((model_diff).view(-1) + (s_test_vec_tensor).view(-1)/full_data_size)**2)*(torch.sum(max_mu_list_tensor.view(1,max_mu_list_tensor.shape[0], max_mu_list_tensor.shape[1])*torch.abs(y_difference)*(y_difference >= 0).type(torch.double), dim = -1))
    
#     delta_influence_list_tensor = torch.t(delta_influence_list_tensor)
    
    remaining_id_tensor = torch.zeros(dataset_train.data.shape[0]).bool()
    
    remaining_id_tensor[full_removed_id_tensor_0.view(-1)] = True
    
    remaining_id_tensor[full_existing_labeled_id_tensor.view(-1)] = False
    
    remaining_removed_id0 = torch.nonzero(remaining_id_tensor).view(-1)
    
#     remaining_removed_id0 = full_removed_id_tensor_0[~(torch.sum(full_removed_id_tensor_0.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
#        
#     
#     print(torch.norm(torch.sort(remaining_removed_id1)[0].type(torch.double) - torch.sort(remaining_removed_id0)[0].type(torch.double)))
#             remaining_removed_id1 = full_removed_id_tensor1[~(torch.sum(full_removed_id_tensor1.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
    
    sorted_influence_list_tensor,sorted_influence_sample_ids0 = torch.sort((full_influence_list_tensor0-delta_influence_list_tensor)[:,remaining_removed_id0].view(-1))
    
    sorted_influence_sample_ids = sorted_influence_sample_ids0%remaining_removed_id0.shape[0]
    
    sorted_remaining_removed_id0 = remaining_removed_id0[sorted_influence_sample_ids]
    
#     id_set1 = remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0+delta_influence_list_tensor)[remaining_removed_id0[args.removed_count:]].view(-1)) > ((full_influence_list_tensor0 - delta_influence_list_tensor)[remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    lower_bound_tensor = ((full_influence_list_tensor0-delta_influence_list_tensor)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][args.removed_count:].view(-1)
    
    upper_bound_value = ((full_influence_list_tensor0+delta_influence_list_tensor)[:,remaining_removed_id0]).view(-1)[sorted_influence_sample_ids0][args.removed_count-1]
    
#     sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][([sorted_remaining_removed_id0[args.removed_count:]].view(-1)) < ((full_influence_list_tensor0 + delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][lower_bound_tensor < upper_bound_value]
    
    final_remaining_ids= torch.cat([sorted_remaining_removed_id0[0:args.removed_count].view(-1), sorted_id_set1], dim = 0)
    
    final_remaining_ids = torch.tensor(list(set(final_remaining_ids.tolist()).union(set(remaining_removed_id0[0:args.removed_count].view(-1).tolist()))))
    
    

    print('compute_hessian time::', t2 - t1)
    
    print('other time::', t3 - t4)
    
    return final_remaining_ids, s_test_vec_tensor

def obtain_remaining_ids_incremental_ac(model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, model_diff, origin_sample_wise_grad, max_mu_list_tensor, full_removed_id_tensor_0, full_existing_labeled_id_tensor, args, full_origin_influence_list_tensor, average_y_change = 0.001):
    
    X = dataset_train.data
    
    X_extend = torch.cat([X, torch.ones([X.shape[0],1], dtype = X.dtype)], dim = 1)
    
#     t1 = time.time()
#     
#     s_test_vec = evaluate_influence_function_repetitive_incremental(batch_size, 0, is_GPU, device, full_out_dir, dataset_train, valid_dataset, model, loss_func)
#     
#     t2 = time.time()
#     
#     s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    t4 = time.time()
#     influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], X.shape[0], is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor, full_data_size, sort = False)
   
    influence_list_tensor0 = torch.norm(origin_sample_wise_grad[full_removed_id_tensor_0]/full_data_size + X_extend[full_removed_id_tensor_0]*average_y_change, dim = 1)
   
    
#     influence_list_tensor0 = torch.mm(origin_sample_wise_grad[full_removed_id_tensor_0], s_test_vec_tensor.view(-1,1))
    
    t3 = time.time() 
#     prev_influence_list_tensor0  = get_influence_sorted(X[full_removed_id_tensor_0], batch_size, is_GPU, device, origin_sample_wise_grad[full_removed_id_tensor_0], prev_s_test_vec, full_data_size, sort = False)
    
    full_influence_list_tensor0 = torch.zeros([X.shape[0]], dtype = influence_list_tensor0.dtype)
    
    full_influence_list_tensor0[full_removed_id_tensor_0] = influence_list_tensor0.view(-1)
    
#     delta_influence_list_tensor = torch.norm(model_diff)*torch.norm(s_test_vec_tensor)/full_data_size*max_mu_list_tensor
    
    delta_influence_list_tensor = torch.norm(model_diff)/full_data_size*max_mu_list_tensor
    
    
    
    remaining_id_tensor = torch.zeros(dataset_train.data.shape[0]).bool()
    
    remaining_id_tensor[full_removed_id_tensor_0.view(-1)] = True
    
    remaining_id_tensor[full_existing_labeled_id_tensor.view(-1)] = False
    
    remaining_removed_id0 = torch.nonzero(remaining_id_tensor).view(-1)
    
#     remaining_removed_id0 = full_removed_id_tensor_0[~(torch.sum(full_removed_id_tensor_0.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
#        
#     
#     print(torch.norm(torch.sort(remaining_removed_id1)[0].type(torch.double) - torch.sort(remaining_removed_id0)[0].type(torch.double)))
#             remaining_removed_id1 = full_removed_id_tensor1[~(torch.sum(full_removed_id_tensor1.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
    
    sorted_influence_list_tensor,sorted_influence_sample_ids = torch.sort((full_influence_list_tensor0-delta_influence_list_tensor)[remaining_removed_id0].view(-1), descending = True)
    
    sorted_remaining_removed_id0 = remaining_removed_id0[sorted_influence_sample_ids]
    
    id_set1 = remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0+delta_influence_list_tensor)[remaining_removed_id0[args.removed_count:]].view(-1)) > ((full_influence_list_tensor0 - delta_influence_list_tensor)[remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    sorted_id_set1 = sorted_remaining_removed_id0[args.removed_count:][((full_influence_list_tensor0+delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count:]].view(-1)) > ((full_influence_list_tensor0 - delta_influence_list_tensor)[sorted_remaining_removed_id0[args.removed_count-1]].view(-1))]
    
    final_remaining_ids= torch.cat([sorted_remaining_removed_id0[0:args.removed_count].view(-1), sorted_id_set1], dim = 0)
    
    final_remaining_ids = torch.tensor(list(set(final_remaining_ids.tolist()).union(set(remaining_removed_id0[0:args.removed_count].view(-1).tolist()))))
    
    

#     print('compute_hessian time::', t2 - t1)
    
    print('other time::', t3 - t4)
    
    return final_remaining_ids    


def origin_compute_sample_wise_gradients2(args, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, train_r_weight  = None):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    # full_grad_tensors0 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    # t1 = time.time()
    #
    # for k in range(X.shape[0]):
    #     curr_x = X[k:k+1]
    #
    #     curr_y = Y[k:k+1]
    #
    #     if is_GPU:
    #         curr_x = curr_x.to(device)
    #
    #         curr_y = curr_y.to(device)
    #
    #     optimizer.zero_grad()
    #
    #     curr_loss = loss_func(model(curr_x), curr_y)
    #
    #     curr_loss.backward()
    #
    #     curr_model_grad = get_vectorized_grads(model, device)
    #
    #     full_grad_tensors0[k] = curr_model_grad.view(-1) + regularization_coeff*curr_w_tensor.view(1,-1)
    #
    # t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
    if s_test_vec_tensor is None:
    
        t3 = time.time()
    
        '''batch_size, regularization_term, is_GPU, device, full_out_dir, training_dataset, valid_dataset, model, loss_func, optimizer, learning_rate = 0.0002, r_weight = None'''
    
        s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = derived_lr, r_weight = train_r_weight)
    
        t4 = time.time()
        
        print('compute_hessian time::', t4 - t3)

        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    
    t5 = time.time()
    
    '''X, Y, optimizer, loss_func, model, batch_size, is_GPU, device, full_grad_tensors, s_test_vec_tensor, full_data_size, sort = True, regularization_term = None'''
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted2(X,Y,optimizer, loss_func, model, batch_size, is_GPU, device, None, s_test_vec_tensor, full_data_size, regularization_term = regularization_term)

    t6 = time.time()

    # print('gradient time::', t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    return influence_list_tensor, X, ordered_list, sorted_train_ids, s_test_vec_tensor

def origin_compute_sample_wise_gradients(args, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, training_dataset= None, train_r_weight = None):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    t1 = time.time()
    
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        optimizer.zero_grad()
        
        curr_loss = loss_func(model(curr_x), curr_y)
        
        curr_loss.backward()
        
        curr_model_grad = get_vectorized_grads(model, device)
        
        full_grad_tensors0[k] = curr_model_grad.view(-1) + regularization_coeff*curr_w_tensor.view(1,-1)

    t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
    if s_test_vec_tensor is None:
    
        t3 = time.time()
    
        '''batch_size, regularization_term, is_GPU, device, full_out_dir, training_dataset, valid_dataset, model, loss_func, optimizer, learning_rate = 0.0002, r_weight = None'''
    
        s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = derived_lr, r_weight = train_r_weight)
    
        t4 = time.time()
        
        print('compute_hessian time::', t4 - t3)

        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    
    t5 = time.time()
    
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted(X, batch_size, is_GPU, device, full_grad_tensors0, s_test_vec_tensor, full_data_size)

    t6 = time.time()

    print('gradient time::', t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids, s_test_vec_tensor

def compute_sample_class_wise_gradient(regularization_coeff, dataset_train, curr_grad_shape,model,optimizer,num_class, loss_func, is_GPU, device):
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_shape], dtype = X.dtype)
    
    full_sample_grad_tensor = torch.zeros([X.shape[0], curr_grad_shape], dtype = X.dtype)
    
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
             
            curr_y = curr_y.to(device)
        
        model_out = -F.log_softmax(model(curr_x))
        
        model_grad_all_classes = []
        
        print('sample wise gradient id::', k)
        
        for p in range(num_class):
        
            optimizer.zero_grad()
            
#             print(model_out.shape)
            
            model_out.view(-1)[p].backward(retain_graph = True)
        
            curr_model_grad = get_vectorized_grads(model)

            model_grad_all_classes.append(curr_model_grad.view(-1).cpu())
        '''num_class * m'''
            
        optimizer.zero_grad()
        
        loss = loss_func(model_out, curr_y)
        
        loss.backward()
        
        curr_model_sample_wise_grad = get_vectorized_grads(model)
        
        full_sample_grad_tensor[k] = curr_model_sample_wise_grad.view(-1).cpu()
        
        model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        
        
        full_grad_tensors0[k] = model_grad_all_classes_tensor# + regularization_coeff*get_all_vectorized_parameters1(list(model.parameters())).view(1,-1)
        
    return full_grad_tensors0, full_sample_grad_tensor


'''num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None'''

def origin_compute_sample_class_wise_gradients2(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, train_prob_case2 = None, train_r_weight = None):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    # full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_list[0].shape[-1]], dtype = X.dtype)
    
    # full_grad_tensors1 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    
    
    if Y_difference is None:
    
        Y_difference = torch.zeros([num_class, X.shape[0], num_class])
        
        for p in range(num_class):
            Y_class = onehot(torch.tensor([p]), num_class)
    #         print(p, Y_class)
            Y_difference[p] = Y_class.view(1,-1) - Y
    # t1 = time.time()
    # for k in range(X.shape[0]):
        # curr_x = X[k:k+1]
        #
        # curr_y = Y[k:k+1]
        #
        # if is_GPU:
            # curr_x = curr_x.to(device)
            #
            # curr_y = curr_y.to(device)
            #
        # model_out = -F.log_softmax(model(curr_x))
        #
        # model_grad_all_classes = []
        #
        # for p in range(num_class):
        #
            # optimizer.zero_grad()
            #
# #             print(model_out.shape)
            #
            # model_out.view(-1)[p].backward(retain_graph = True)
            #
            # curr_model_grad = get_vectorized_grads(model, device)
            #
            # model_grad_all_classes.append(curr_model_grad.view(-1))
        # '''num_class * m'''    
        # model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        #
        #
# #         full_model_out = loss_func(model(curr_x), curr_y)
# #         
# #         optimizer.zero_grad()
# #         
# #         print(torch.sum(model_out.view(-1)*curr_y.view(-1)) - full_model_out)
# #         
# #         full_model_out.backward()
# #         
# #         curr_full_model_grad = get_vectorized_grads(model, device)
# #         
# #         calculated_full_grad = torch.sum(curr_y.view(-1,1)*model_grad_all_classes_tensor, dim = 0)
# #         
# #         print(torch.norm(calculated_full_grad.view(-1) - curr_full_model_grad.view(-1)))
# #         
# #         full_grad_tensors1[k] = curr_full_model_grad.view(-1)
        #
# #         curr_loss = loss_func(, curr_y)
# #         
# #         curr_loss.backward()
        #
# #         curr_model_grad = get_vectorized_grads(model, device)
        #
        # full_grad_tensors0[k] = model_grad_all_classes_tensor + regularization_coeff*curr_w_tensor.view(1,-1)
        #
    t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
    if s_test_vec_tensor is None:
    
        t3 = time.time()
    
        s_test_vec = evaluate_influence_function_repetitive_incremental2(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = derived_lr, r_weight = train_r_weight)
    
        t4 = time.time()
        
        print('compute_hessian time::', t4 - t3)

        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec)
    
#     s_test_vec = s_test_vec.to('cpu')
    
    t5 = time.time()
    
    train_grad,_,_ = compute_gradient(model, X, Y, loss_func, r_weight, 0, optimizer, is_GPU, device, random_sampling = False, bz = batch_size, batch_ids = None)
    
    train_grad = train_grad + regularization_term.view(1,-1)

    '''loss_func, X, Y, is_GPU, device, y_difference, s_test_vec_tensor, full_data_size, curr_w_tensor, regularization_coeff, regularization_term, sort = True, full_grad_tensors_multi_class = None, regular_rate = 0.1'''
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted_class_wise2(model, loss_func, num_class, optimizer, X, Y, is_GPU, device, Y_difference, s_test_vec_tensor, full_data_size, curr_w_tensor, regularization_coeff, regularization_term, regular_rate = regular_rate, train_prob_case2 = train_prob_case2, train_grad = train_grad)

    t6 = time.time()

    # print(calculate_gradient_time_prefix, t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    full_grad_tensors0 = X.clone()

    return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids, s_test_vec_tensor


def compute_sample_wise_gradient(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, train_prob_case2 = None, train_r_weight = None):
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_list[0].shape[-1]], dtype = X.dtype)
    
    full_grad_tensors1 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        model_out0 = model(curr_x)
        
        model_out = -F.log_softmax(model_out0)
        
        model_grad_all_classes = []
        
        # full_model_out = 0
        
        for p in range(num_class):
        
            optimizer.zero_grad()
            
#             print(model_out.shape)
            
            model_out.view(-1)[p].backward(retain_graph = True)
        
            curr_model_grad = get_vectorized_grads(model, device)

            model_grad_all_classes.append(curr_model_grad.view(-1))
        '''num_class * m'''    
        model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        
        optimizer.zero_grad()
        
        full_model_out = loss_func(model_out0, curr_y)
        
        full_model_out.backward(retain_graph = True)
        
        full_model_grad = get_vectorized_grads(model, device)
#


#         optimizer.zero_grad()
#         
#         print(torch.sum(model_out.view(-1)*curr_y.view(-1)) - full_model_out)
#         
#         full_model_out.backward()
#         
#         curr_full_model_grad = get_vectorized_grads(model, device)
#         
#         calculated_full_grad = torch.sum(curr_y.view(-1,1)*model_grad_all_classes_tensor, dim = 0)
#         
#         print(torch.norm(calculated_full_grad.view(-1) - curr_full_model_grad.view(-1)))
#         
#         full_grad_tensors1[k] = curr_full_model_grad.view(-1)
        
#         curr_loss = loss_func(, curr_y)
#         
#         curr_loss.backward()
        
#         curr_model_grad = get_vectorized_grads(model, device)
        
        full_grad_tensors0[k] = model_grad_all_classes_tensor# + regularization_coeff*curr_w_tensor.view(1,-1) + (1-regular_rate)*full_model_grad.view(1,-1)

        full_grad_tensors1[k] = full_model_grad.view(1,-1)

        
        
    return full_grad_tensors0, full_grad_tensors1

def origin_compute_sample_class_wise_gradients(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, train_prob_case2 = None, train_r_weight = None):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_list[0].shape[-1]], dtype = X.dtype)
    
    full_grad_tensors1 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    
    
    if Y_difference is None:
    
        Y_difference = torch.zeros([num_class, X.shape[0], num_class])
        
        for p in range(num_class):
            Y_class = onehot(torch.tensor([p]), num_class)
    #         print(p, Y_class)
            Y_difference[p] = Y_class.view(1,-1) - Y
    t1 = time.time()
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        model_out0 = model(curr_x)
        
        model_out = -F.log_softmax(model_out0)
        
        model_grad_all_classes = []
        
        # full_model_out = 0
        
        for p in range(num_class):
        
            optimizer.zero_grad()
            
#             print(model_out.shape)
            
            model_out.view(-1)[p].backward(retain_graph = True)
        
            curr_model_grad = get_vectorized_grads(model, device)

            model_grad_all_classes.append(curr_model_grad.view(-1))
        '''num_class * m'''    
        model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        
        optimizer.zero_grad()
        
        full_model_out = loss_func(model_out0, curr_y)
        
        full_model_out.backward(retain_graph = True)
        
        full_model_grad = get_vectorized_grads(model, device)
#


#         optimizer.zero_grad()
#         
#         print(torch.sum(model_out.view(-1)*curr_y.view(-1)) - full_model_out)
#         
#         full_model_out.backward()
#         
#         curr_full_model_grad = get_vectorized_grads(model, device)
#         
#         calculated_full_grad = torch.sum(curr_y.view(-1,1)*model_grad_all_classes_tensor, dim = 0)
#         
#         print(torch.norm(calculated_full_grad.view(-1) - curr_full_model_grad.view(-1)))
#         
#         full_grad_tensors1[k] = curr_full_model_grad.view(-1)
        
#         curr_loss = loss_func(, curr_y)
#         
#         curr_loss.backward()
        
#         curr_model_grad = get_vectorized_grads(model, device)
        
        full_grad_tensors0[k] = model_grad_all_classes_tensor# + regularization_coeff*curr_w_tensor.view(1,-1) + (1-regular_rate)*full_model_grad.view(1,-1)

        full_grad_tensors1[k] = full_model_grad.view(1,-1)

    t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
    if s_test_vec_tensor is None:
    
        t3 = time.time()
    
        s_test_vec = evaluate_influence_function_repetitive_incremental2(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = derived_lr, r_weight = train_r_weight)
    
        t4 = time.time()
        
        print('compute_hessian time::', t4 - t3)

        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    
#     s_test_vec = s_test_vec.to('cpu')
    
    t5 = time.time()
    
    train_grad,_,_ = compute_gradient(model, X, Y, loss_func, r_weight, 0, optimizer, is_GPU, device, random_sampling = False, bz = batch_size, batch_ids = None)
    
    train_grad = None#train_grad + regularization_term.view(1,-1)
    
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted_class_wise(X, Y, batch_size, is_GPU, device, full_grad_tensors0, Y_difference, s_test_vec_tensor, full_data_size, full_grad_tensors1 = full_grad_tensors1, regular_rate = regular_rate, regularization_term = regularization_term, train_prob_case2 = train_prob_case2, train_grad = train_grad)

    t6 = time.time()

    print(calculate_gradient_time_prefix, t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids, s_test_vec_tensor

def origin_compute_sample_class_wise_gradients2_0(args, num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_list[0].shape[-1]], dtype = X.dtype)
    
    full_grad_tensors1 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    
    
    if Y_difference is None:
    
        Y_difference = torch.zeros([num_class, X.shape[0], num_class])
        
        for p in range(num_class):
            Y_class = onehot(torch.tensor([p]), num_class)
    #         print(p, Y_class)
            Y_difference[p] = Y_class.view(1,-1) - Y
    t1 = time.time()
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        model_out0 = model(curr_x)
        
        model_out = -F.log_softmax(model_out0)
        
        model_grad_all_classes = []
        
        # full_model_out = 0
        
        for p in range(num_class):
        
            optimizer.zero_grad()
            
#             print(model_out.shape)
            
            model_out.view(-1)[p].backward(retain_graph = True)
        
            curr_model_grad = get_vectorized_grads(model, device)

            model_grad_all_classes.append(curr_model_grad.view(-1))
        '''num_class * m'''    
        model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
        
        
        full_model_out = loss_func(model_out0, curr_y)
        
        full_model_out.backward(retain_graph = True)
        
        full_model_grad = get_vectorized_grads(model, device)
#


#         optimizer.zero_grad()
#         
#         print(torch.sum(model_out.view(-1)*curr_y.view(-1)) - full_model_out)
#         
#         full_model_out.backward()
#         
#         curr_full_model_grad = get_vectorized_grads(model, device)
#         
#         calculated_full_grad = torch.sum(curr_y.view(-1,1)*model_grad_all_classes_tensor, dim = 0)
#         
#         print(torch.norm(calculated_full_grad.view(-1) - curr_full_model_grad.view(-1)))
#         
#         full_grad_tensors1[k] = curr_full_model_grad.view(-1)
        
#         curr_loss = loss_func(, curr_y)
#         
#         curr_loss.backward()
        
#         curr_model_grad = get_vectorized_grads(model, device)
        
        full_grad_tensors0[k] = model_grad_all_classes_tensor# + regularization_coeff*curr_w_tensor.view(1,-1) + (1-regular_rate)*full_model_grad.view(1,-1)

        full_grad_tensors1[k] = full_model_grad.view(1,-1)

    t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
    if s_test_vec_tensor is None:
    
        t3 = time.time()
    
        s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func, optimizer, learning_rate = derived_lr, r_weight = r_weight)
    
        t4 = time.time()
        
        print('compute_hessian time::', t4 - t3)

        s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec)
    
#     s_test_vec = s_test_vec.to('cpu')
    
    t5 = time.time()
    
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted_class_wise2_0(X, Y, batch_size, is_GPU, device, full_grad_tensors0, Y_difference, s_test_vec_tensor, full_data_size, full_grad_tensors1 = full_grad_tensors1, regular_rate = regular_rate, regularization_term = regularization_term)

    t6 = time.time()

    print(calculate_gradient_time_prefix, t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids, s_test_vec_tensor


# def origin_compute_sample_class_wise_gradients_no_delta(num_class, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train, s_test_vec_tensor = None, Y_difference = None):
#     
#     
#     X = dataset_train.data
#     
#     Y = dataset_train.labels
#     
#     full_grad_tensors0 = torch.zeros([X.shape[0], num_class, curr_grad_list[0].shape[-1]], dtype = X.dtype)
#     
#     full_grad_tensors1 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)
# 
#     curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  
# 
#     
#     
#     if Y_difference is None:
#     
#         Y_difference = torch.zeros([num_class, X.shape[0], num_class])
#         
#         for p in range(num_class):
#             Y_class = onehot(torch.tensor([p]), num_class)
#     #         print(p, Y_class)
#             Y_difference[p] = Y_class.view(1,-1) - Y
#     t1 = time.time()
#     for k in range(X.shape[0]):
#         curr_x = X[k:k+1]
#         
#         curr_y = Y[k:k+1]
#         
#         if is_GPU:
#             curr_x = curr_x.to(device)
#             
#             curr_y = curr_y.to(device)
#         
#         model_out = -F.log_softmax(model(curr_x))
#         
#         model_grad_all_classes = []
#         
#         for p in range(num_class):
#         
#             optimizer.zero_grad()
#             
# #             print(model_out.shape)
#             
#             model_out.view(-1)[p].backward(retain_graph = True)
#         
#             curr_model_grad = get_vectorized_grads(model, device)
# 
#             model_grad_all_classes.append(curr_model_grad.view(-1))
#         '''num_class * m'''    
#         model_grad_all_classes_tensor = torch.stack(model_grad_all_classes, dim = 0)
#         
#         
# #         full_model_out = loss_func(model(curr_x), curr_y)
# #         
# #         optimizer.zero_grad()
# #         
# #         print(torch.sum(model_out.view(-1)*curr_y.view(-1)) - full_model_out)
# #         
# #         full_model_out.backward()
# #         
# #         curr_full_model_grad = get_vectorized_grads(model, device)
# #         
# #         calculated_full_grad = torch.sum(curr_y.view(-1,1)*model_grad_all_classes_tensor, dim = 0)
# #         
# #         print(torch.norm(calculated_full_grad.view(-1) - curr_full_model_grad.view(-1)))
# #         
# #         full_grad_tensors1[k] = curr_full_model_grad.view(-1)
#         
# #         curr_loss = loss_func(, curr_y)
# #         
# #         curr_loss.backward()
#         
# #         curr_model_grad = get_vectorized_grads(model, device)
#         
#         full_grad_tensors0[k] = model_grad_all_classes_tensor# + regularization_coeff*curr_w_tensor.view(1,-1)
# 
#     t2 = time.time()
# 
#     regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)
# 
#     optimizer.zero_grad()
#     
#     if s_test_vec_tensor is None:
#     
#         t3 = time.time()
#     
#         s_test_vec = evaluate_influence_function_repetitive_incremental(batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func)
#     
#         t4 = time.time()
#         
#         print('compute_hessian time::', t4 - t3)
# 
#         s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
#     
#     t5 = time.time()
#     
#     influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted_class_wise(X, Y, batch_size, is_GPU, device, full_grad_tensors0, Y_difference, s_test_vec_tensor, full_data_size, full_grad_tensors1 = full_grad_tensors1)
# 
#     t6 = time.time()
# 
#     print(calculate_gradient_time_prefix, t2 - t1)
#     
#     print('calculate_influence time::', t6 - t5)
# 
#     return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids, s_test_vec_tensor


def origin_compute_sample_wise_gradients_ac(model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size, full_data_size, full_dataset_train,  average_y_change = 0.001):
    
    
    X = dataset_train.data
    
    X_extend = torch.cat([X, torch.ones([X.shape[0], 1], dtype = torch.double)], dim = 1)
    
    Y = dataset_train.labels
    
    full_grad_tensors0 = torch.zeros([X.shape[0], curr_grad_list[0].shape[-1]], dtype = X.dtype)

    curr_w_tensor = get_all_vectorized_parameters1(list(model.parameters()), device)  

    t1 = time.time()
    
    for k in range(X.shape[0]):
        curr_x = X[k:k+1]
        
        curr_y = Y[k:k+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        optimizer.zero_grad()
        
        curr_loss = loss_func(model(curr_x), curr_y)
        
        curr_loss.backward()
        
        curr_model_grad = get_vectorized_grads(model, device)
        
        full_grad_tensors0[k] = curr_model_grad.view(-1)# + regularization_coeff*curr_w_tensor.view(1,-1)


    

    t2 = time.time()

    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)

    optimizer.zero_grad()
    
#     if s_test_vec_tensor is None:
#     
#         t3 = time.time()
#     
#         s_test_vec = evaluate_influence_function_repetitive_incremental(batch_size, regularization_term, is_GPU, device, full_out_dir, full_dataset_train, valid_dataset, model, loss_func)
#     
#         t4 = time.time()
#         
#         print('compute_hessian time::', t4 - t3)
# 
#         s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    
    t5 = time.time()
    
    
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted_ac(X_extend, batch_size, is_GPU, device, full_grad_tensors0, average_y_change, full_data_size)

    t6 = time.time()

    print('gradient time::', t2 - t1)
    
    print('calculate_influence time::', t6 - t5)

    return influence_list_tensor, full_grad_tensors0, ordered_list, sorted_train_ids

def incremental_compute_sample_wise_gradients(args, model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, S_k_list, Y_k_list_tensor, prev_full_grad_list, curr_w_list, prev_w_list, remaining_ids,  is_GPU, device, regularization_coeff, m, batch_size, zero_mat_dim, full_S_k, full_Y_k, full_sigma_k, full_inv_mat, full_combined_mat):
    
    
    X = dataset_train.data
    
    Y = dataset_train.labels
#     prev_para = get_all_vectorized_parameters1(w_list[0])
    
#     prev_para = get_all_vectorized_parameters1(torch.load(full_out_dir + '/model_0').parameters())
#     remaining_bool_tensors = torch.zeros(X.shape[0]).bool()
#     
#     remaining_bool_tensors[remaining_ids] = True

    other_ids = torch.tensor(list(set(list(range(X.shape[0]))).difference(set(remaining_ids.tolist()))))

    
    full_grad_tensors = torch.zeros([X.shape[0], Y_k_list_tensor.shape[-1]], dtype = X.dtype)
    
    
#     prev_para = get_all_vectorized_parameters1(torch.load(full_out_dir + '/w_list_' + str(0))[-1])
#     
# #     prev_grad = grad_list[0][remaining_ids]
#     grad_list = torch.load(full_out_dir + '/grad_list_' + str(0))
# 
#     prev_grad = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(0))[-1]
#     
# #     for i in range(2, 4):
#     
#     
#     
#     curr_grad_list = torch.load(full_out_dir + '/grad_list_' + str(i))
    
    para_diff = get_all_vectorized_parameters1(curr_w_list[-1], device) - get_all_vectorized_parameters1(prev_w_list[-1], device)
    
    if is_GPU:
        para_diff = para_diff.to(device)
    
#     hessian_para_prod_mini0, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, mini_Y_k_list2, 0,0, m, para_diff.view(-1,1), 0,is_GPU, device)
#     
#     exp_grad_gap_mini = curr_grad_list[-1] - grad_list[-1] + regularization_coeff*para_diff
#     
#     print(torch.norm(hessian_para_prod_mini0.view(-1) - exp_grad_gap_mini.view(-1)))
    
    
    
    
#     hessian_para_prod_mini, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list_tensor[:,0].view(Y_k_list_tensor.shape[0], 1, Y_k_list_tensor.shape[2]), 0,0, m, para_diff.view(-1,1), 0,is_GPU, device)
    
#     exp_grad_gap_mini = curr_grad[0] - prev_grad[0] + regularization_coeff*para_diff
    
#     print(torch.norm(hessian_para_prod_mini.view(-1) - exp_grad_gap_mini.view(-1)))
    
    
    
    updated_full_grad_list= []
    
    
    
    total_approx_time = 0
    
    add_time = 0
    
    update_time = 0
    
    update_time2 = 0
    
    prev_w_tensor = get_all_vectorized_parameters1(prev_w_list[-1], device)
    
    curr_w_tensor = get_all_vectorized_parameters1(curr_w_list[-1], device)  
    
    combined_mat = torch.zeros([batch_size, Y_k_list_tensor.shape[-1], 2*m], dtype = X.dtype, device = device)
    
    t1 = time.time()
    
    for k in range(0, remaining_ids.shape[0], batch_size):
    
        end_id = k + batch_size
        
        if end_id >= remaining_ids.shape[0]:
            end_id = remaining_ids.shape[0]
            
        print(k, end_id)
        
#         curr_Y_k_list_tensor = Y_k_list_tensor[:,k:end_id]
        
        curr_Y_k_tensor = full_Y_k[k:end_id]
        print(full_Y_k.shape, curr_Y_k_tensor.shape)
        curr_S_k_tensor = full_S_k[k:end_id]
        
        curr_sigma_k = full_sigma_k[k:end_id]
        
        curr_inv_mat = full_inv_mat[k:end_id]
        
        curr_combined_mat = full_combined_mat[k:end_id]
        
        print(curr_sigma_k.shape,curr_inv_mat.shape)
        
        if is_GPU:
#             curr_Y_k_list_tensor = curr_Y_k_list_tensor.to(device)
            curr_Y_k_tensor = curr_Y_k_tensor.to(device)
#             full_S_k = full_S_k.to(device)
            curr_sigma_k = curr_sigma_k.to(device)
            curr_inv_mat = curr_inv_mat.to(device)
            curr_S_k_tensor = curr_S_k_tensor.to(device)
            curr_combined_mat = curr_combined_mat.to(device)
#         if k == 0:
        t5 = time.time()
#         hessian_para_prod_0, zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat, approx_time = cal_approx_hessian_vec_prod0_3_sample_wise(S_k_list, curr_Y_k_list_tensor, m, para_diff.view(-1,1), combined_mat, is_GPU, device)
        
        t6 = time.time()
        
        hessian_para_prod_1 = cal_approx_hessian_vec_prod0_3_sample_wise_incremental(zero_mat_dim, curr_S_k_tensor, curr_Y_k_tensor, curr_sigma_k, curr_inv_mat, curr_combined_mat, para_diff.view(-1,1), is_GPU, device)
        
        t7 = time.time()
        
        update_time2 += (t7 - t6)
        
        update_time += (t6  -t5)
        
#         print(torch.norm(hessian_para_prod_0 - hessian_para_prod_1))
        
#         total_approx_time += approx_time
        
#         else:
#             
#             
#             hessian_para_prod_0 = cal_approx_hessian_vec_prod0_3_sample_wise_incremental(zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat, k, para_diff.view(-1,1), is_GPU, device)
        '''curr_Y_k, curr_S_k, sigma_k, mat'''
#             hessian_para_prod_0_2, zero_mat_dim, curr_S_k, curr_Y_k, sigma_k, inv_mat = cal_approx_hessian_vec_prod0_3_sample_wise(S_k_list, curr_Y_k_list_tensor, m, para_diff.view(-1,1), is_GPU, device)
#             
#             print(torch.norm(hessian_para_prod_0_2 - hessian_para_prod_0))
    
        print(hessian_para_prod_1.shape)
    
        t3 = time.time()
    
        updated_full_grad = prev_full_grad_list[-1][remaining_ids[k:end_id]] + (hessian_para_prod_1).cpu()# + regularization_coeff*prev_w_tensor.view(1,-1)).cpu()
#         updated_full_grad = prev_full_grad_list[-1][k:end_id] + (hessian_para_prod_0).cpu()
        t4 = time.time()
        
        add_time += (t4 - t3)
        
        del hessian_para_prod_1
#         updated_full_grad_list.append(updated_full_grad)
    
        full_grad_tensors[remaining_ids[k:end_id]] = updated_full_grad 
    

    for k in range(other_ids.shape[0]):
        curr_x = X[other_ids[k]:other_ids[k]+1]
        
        curr_y = Y[other_ids[k]:other_ids[k]+1]
        
        if is_GPU:
            curr_x = curr_x.to(device)
            
            curr_y = curr_y.to(device)
        
        optimizer.zero_grad()
        
        curr_loss = loss_func(model(curr_x), curr_y)
        
        curr_loss.backward()
        
        curr_model_grad = get_vectorized_grads(model)
        
        full_grad_tensors[other_ids[k]] = curr_model_grad.view(-1) + regularization_coeff*curr_w_tensor.cpu().view(1,-1)
    
    t2 = time.time()
    
    
    
    
    regularization_term = regularization_coeff*curr_w_tensor.view(1,-1)
    
#     influences2, ordered_list, sorted_train_ids, s_test_vec = evaluate_influence_function_repetitive_incremental(batch_size, regularization_term, is_GPU, device, full_out_dir, dataset_train, full_grad_tensors, valid_dataset, model, loss_func)
    
    s_test_vec = evaluate_influence_function_repetitive_incremental(args, batch_size, regularization_term, is_GPU, device, full_out_dir, dataset_train, valid_dataset, model, loss_func)
    
    s_test_vec_tensor = get_all_vectorized_parameters1(s_test_vec, device)
    
    influence_list_tensor,ordered_list, sorted_train_ids  = get_influence_sorted(X, batch_size, is_GPU, device, full_grad_tensors, s_test_vec_tensor)
    
    print(t2 - t1)
    
    
    
    
    
    print('total approx time::', total_approx_time)
    
    print('add time::', add_time)
    
    print('update time::', update_time)
    print('update time 2::', update_time2)
#     print(hessian_para_prod_0.shape)
#     
#     print('hessian prod diff::', torch.norm(hessian_para_prod_0[0] - hessian_para_prod_mini))

#     exp_grad_gap  = curr_grad - prev_grad + regularization_coeff*para_diff
    
#     print(torch.norm(hessian_para_prod_0.view(-1) - exp_grad_gap.view(-1)))
    


    
    return influence_list_tensor, full_grad_tensors, ordered_list, sorted_train_ids
#     S_k_list = deque()
#     
#     Y_k_list = deque()
#     
#     mini_Y_k_list = deque()
    
#     m = 4
#     
#     for i in range(6):
#         
# #         curr_para = get_all_vectorized_parameters1(w_list[i+1], device)       
#         
#         curr_para = get_all_vectorized_parameters1(torch.load(full_out_dir + '/model_' + str(i+1)).parameters())
#         
#         curr_s_list = (curr_para - prev_para) + 1e-16
#         
#                         
# #         curr_grad = grad_list[i+1][remaining_ids]
#         curr_grad = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(i+1))
#         
#         grad_gap  = curr_grad - prev_grad
#         
#         
#         
#         if i >= m:
#             '''S_k_list, Y_k_list, k, v_vec, is_GPU, device'''
# #             hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3_sample_wise(S_k_list, Y_k_list, m, curr_s_list.view(-1,1), is_GPU, device)
#             
#             '''S_k_list, Y_k_list, i, m, k, v_vec, period, is_GPU, device'''
#             hessian_para_prod_0, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, mini_Y_k_list, 0,0, m, curr_s_list.view(-1,1), 0,is_GPU, device)
#             
#             
#             removed_y_k = Y_k_list.popleft()
#             removed_s_k = S_k_list.popleft()
#             mini_Y_k_list.popleft()
#             del removed_y_k, removed_s_k
#         
# #             print(torch.norm(hessian_para_prod_0.view(-1) - grad_gap[remaining_ids[0]].view(-1)))
#             print(torch.norm(hessian_para_prod_0.view(-1) - torch.mean(grad_gap, dim = 0).view(-1) - regularization_coeff*curr_s_list.view(-1)))
#             
#             print('here')
#         
#         Y_k_list.append(grad_gap)
# 
# #         mini_Y_k_list.append(grad_gap[remaining_ids[0]].view(1,-1))
#         mini_Y_k_list.append(torch.mean(grad_gap, dim = 0).view(1,-1) + regularization_coeff*curr_s_list)
# 
#         S_k_list.append(curr_s_list)
#                 
# #         prev_para =  curr_para
# #         
# #         prev_grad = curr_grad
#         
#         print('here')



def model_update_deltagrad2(max_epoch, period, length, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, learning_rate, random_ids_multi_super_iterations, batch_size, criterion, optimizer, regularization_coeff, is_GPU, device, exp_updated_w_list = None, exp_updated_grad_list = None, compare = True, GPU_measure = False, GPUID = 0, r_weight_old = None, r_weight_new = None):
    '''function to use deltagrad for incremental updates'''
    
    
    para = list(model.parameters())
    
    # if is_GPU and GPU_measure:
        # nvidia_smi.nvmlInit()
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(GPUID)
# #         GPU_utilization_list = []
        # GPU_mem_usage_list = []
        
    use_standard_way = False
    
    recorded = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     overhead4 = 0
#     
#     overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    

    updated_w_list = []
    
    updated_grad_list = []
    
    # if is_GPU and GPU_measure:
        # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # init_memory = res.used
        #
        # print(nvgpu.gpu_info())
        #
        # print('gpu utilization::', res.used)
    
#     res_para_list = []
#     
#     res_grad_list = []
    
#     t5 = time.time()
    
    '''detect which samples are removed from each mini-batch'''
    
    
#     t6 = time.time()
#     
#     overhead3 += (t6  -t5)
    
    '''main for loop of deltagrad'''
    
    i = 0
    
    t1 = 0
    
    t2 = 0
    
    time1 = 0 
    
    time2 = 0
    
#     for k in range(max_epoch):
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
#         random_ids_list = random_ids_list_all_epochs[k]
        
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        
#         X = dataset_train.data[random_ids]
#         Y = dataset_train.labels[random_ids]
#         
#         update_labels_curr_epoch = updated_labels[random_ids]
#         
#         ids_with_changed_ids_curr_epoch = ids_with_changed_ids[random_ids]
        
#         ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids[random_ids]
        
#         curr_entry_grad_list_epoch = all_entry_grad_list[k]
        
#         updated_grad = get_entry_grad_with_labels(update_labels_curr_epoch, curr_entry_grad_list)
        
#         for p in range(len(random_ids_list)):
        for p in range(0, dataset_train.lenth, batch_size):
            
#             curr_matched_ids = items[2]        
#             curr_matched_ids = random_ids_list[p]
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
#             t3 = time.time()
            
            curr_rand_ids = random_ids[j:end_id]
            
#                 batch_ids_with_unchanged_ids_curr_epoch = batch_ids_with_unchanged_ids_curr_epoch.to(device)
#             t1 = time.time()
#             
#             time2 += (t1 - t3)
#             curr_entry_grad_list = curr_entry_grad_list_epoch[j:end_id]
#             
#             if is_GPU:
#                 curr_entry_grad_list = curr_entry_grad_list.to(device)
            
#             learning_rate = learning_rate_all_epochs[i]
            
            
            old_lr = learning_rate    
                
            
            
            
            if (i-curr_init_epochs)%period == 0 or (i <= period):
                
                recorded = 0
                
                use_standard_way = True
            
            # if i< curr_init_epochs or use_standard_way == True:
            #
                # if is_GPU and GPU_measure:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print('gpu utilization::', k,p, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    # print(nvgpu.gpu_info())
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                    
#                     print('gpu utilization::', k,p, res.__dict__)
                
                
                batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
                batch_Y = dataset_train.labels[curr_rand_ids]
                batch_update_labels = updated_labels[curr_rand_ids]
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
                batch_r_weight_new = None
                
                batch_r_weight_old = None
                
                if r_weight_new is not None:
                    batch_r_weight_new = r_weight_new[curr_rand_ids]
                    
                if r_weight_old is not None:
                    batch_r_weight_old = r_weight_old[curr_rand_ids]
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
                    if r_weight_old is not None:
                        batch_r_weight_old = batch_r_weight_old.to(device)
                    if r_weight_new is not None:
                        batch_r_weight_new = batch_r_weight_new.to(device)
                
                
                

                set_model_parameters(model, para, device)

                updated_w_list.append(get_model_para_list(model))
                
                compute_derivative_one_more_step(model, batch_X, batch_update_labels, criterion, optimizer, r_weight = batch_r_weight_new)
                
                curr_gradients = get_vectorized_grads(model, device)# get_all_vectorized_parameters1(model.get_all_gradient())
                
                updated_grad_list.append(curr_gradients.clone().cpu())
                
                if compare:
                    compute_para_grad_diff(get_vectorized_params(model), get_all_vectorized_parameters1(exp_updated_w_list[i]), curr_gradients, exp_updated_grad_list[i], para_list_GPU_tensor[cached_id])

                curr_changed_ids_grad = 0

                prev_changed_ids_grad = 0

                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_update_labels[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer, r_weight = batch_r_weight_new[batch_ids_with_changed_ids_curr_epoch])
                            
                    curr_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_Y[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer, r_weight = batch_r_weight_old[batch_ids_with_changed_ids_curr_epoch])
                    
                    prev_changed_ids_grad = get_vectorized_grads(model, device)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para, device)
                
                    if k>0 or (p > 0 and k == 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_para = prev_para.to(device)
                        
                        curr_s_list = (curr_para - prev_para) + 1e-16
                        
                        # print(curr_para, prev_para, curr_s_list)
                        
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
#                     gradient_full = (expect_gradients*curr_remaining_id_size + gradient_remaining*curr_matched_ids_size)/(curr_remaining_id_size + curr_matched_ids_size)
                    gradient_full = curr_gradients

                    curr_grad_prev_labels = (gradient_full*batch_X.shape[0] - (curr_changed_ids_grad - prev_changed_ids_grad)*torch.sum(batch_ids_with_changed_ids_curr_epoch))/batch_X.shape[0]                


                    if k>0 or (p > 0 and k == 0):
                        
                        
                        prev_grad = grad_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_grad = prev_grad.to(device)
#                         prev_grad = updated_grad_list_GPU_tensor[cached_id]
#                         prev_grad = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list)
                        
#                         if is_GPU:
#                             prev_grad = prev_grad.to(device)
                        
                        Y_k_list.append(curr_grad_prev_labels - prev_grad + regularization_coeff*curr_s_list)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    

                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*curr_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del curr_gradients
                    
                    if k>0 or (p > 0 and k == 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                # if is_GPU and GPU_measure:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    #
# #                     GPU_utilization_list.append(res.gpu)
                    # print(nvgpu.gpu_info())
                    # print('gpu utilization::', k,p, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                
                
#                 del batch_X, batch_Y
            else:
                
                # if is_GPU and GPU_measure:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    #
# #                     GPU_utilization_list.append(res.gpu)
                    # print(nvgpu.gpu_info())
                    # print('gpu utilization::', k,p, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                
                '''use l-bfgs algorithm to evaluate the gradients'''
                
#                 gradient_dual = None
    
#                 if not removed_batch_empty_list[i]:
                set_model_parameters(model, para)
                
                updated_w_list.append(get_model_para_list(model))
                
                grad1 = 0
                
#                 batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
#                 batch_Y = dataset_train.labels[curr_rand_ids]
                
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
                sub_rand_ids = curr_rand_ids[batch_ids_with_changed_ids_curr_epoch]
                
                mini_batch_Y = dataset_train.labels[sub_rand_ids]
#                 batch_update_labels = updated_labels[curr_rand_ids]
                
                mini_batch_update_labels = updated_labels[sub_rand_ids]
                
                updated_batch_X = dataset_train.data[sub_rand_ids]
                
                batch_r_weight_new = None
                
                batch_r_weight_old = None
                
                if r_weight_new is not None:
                    batch_r_weight_new = r_weight_new[sub_rand_ids]
                    
                if r_weight_old is not None:
                    batch_r_weight_old = r_weight_old[sub_rand_ids]
                
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
#                     batch_X = batch_X.to(device)
#                     batch_Y = batch_Y.to(device)
                    mini_batch_Y = mini_batch_Y.to(device)
                    mini_batch_update_labels = mini_batch_update_labels.to(device)
                    updated_batch_X = updated_batch_X.to(device)
#                     batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
                    
                    if r_weight_new is not None:
                        batch_r_weight_new = batch_r_weight_new.to(device)
                        
                    if r_weight_old is not None:
                        batch_r_weight_old = batch_r_weight_old.to(device)
#                 print(torch.norm(updated_batch_X - batch_X[batch_ids_with_changed_ids_curr_epoch]))
                
                
                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    # compute_derivative_one_more_step(model, updated_batch_X, mini_batch_update_labels, criterion, optimizer, batch_r_weight_new)
                    #
                    # curr_changed_ids_grad = get_vectorized_grads(model, device)
                    #
                    # compute_derivative_one_more_step(model, updated_batch_X, mini_batch_Y, criterion, optimizer, batch_r_weight_old)
                    #
                    # prev_changed_ids_grad = get_vectorized_grads(model, device)
                    #
                    # grad1 = curr_changed_ids_grad - prev_changed_ids_grad

                    compute_derivative_one_more_step_diff(model, updated_batch_X, mini_batch_update_labels, mini_batch_Y, criterion, optimizer, batch_r_weight_new, batch_r_weight_old)
                    
                    grad1 = get_vectorized_grads(model, device)
                    
                    # print(torch.norm(grad1_2 - grad1))
                    #
                    # print('here')
                    # compute_derivative_one_more_step(model, updated_batch_X, mini_batch_update_labels - mini_batch_Y, criterion, optimizer)
                    #
                    # grad1 = get_vectorized_grads(model, device)
                    
#                     print(torch.norm(grad1_2 - grad1))
                    
#                     print('here')
                    
                
                prev_grad = grad_list_GPU_tensor[cached_id]
                
                if is_GPU:
                    prev_grad = prev_grad.to(device)
                    
                grad1 = (grad1*torch.sum(batch_ids_with_changed_ids_curr_epoch) + prev_grad*curr_rand_ids.shape[0])/curr_rand_ids.shape[0]
#                 grad1 = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list) 
                
                prev_para = para_list_GPU_tensor[cached_id]
                
#                 grad1 = updated_grad_list_GPU_tensor[cached_id] + regularization_coeff* #get_entry_grad_with_labels(batch_Y - batch_update_labels, curr_entry_grad_list)
                
                if is_GPU:
                    
                    prev_para = prev_para.to(device)
                grad1 = grad1 + regularization_coeff*prev_para
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                     
#                     gradient_dual = model.get_all_gradient()
                    
                with torch.no_grad():
                
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(para, device) - para_list_GPU_tensor[cached_id].to(device)))
                    
                    
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                    exp_gradient, exp_param = None, None
                    
                    if exp_updated_grad_list is not None and exp_updated_w_list is not None and len(exp_updated_grad_list) > 0 and len(exp_updated_w_list) > 0: 
                        exp_gradient, exp_param = exp_updated_grad_list[i], exp_updated_w_list[i]
                    
                    final_grad = torch.t(grad1.view(vec_para_diff.shape) + hessian_para_prod.view(vec_para_diff.shape))
                    
                    
#                     if gradient_dual is not None:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
#                         
#                     else:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    vec_para = get_all_vectorized_parameters1(para, device)
                    
                    curr_updated_grad = final_grad - regularization_coeff*vec_para
                    
                    updated_grad_list.append(curr_updated_grad.clone().cpu())
                    
                    if exp_param is not None:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, get_all_vectorized_parameters1(exp_param), para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    else:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, None, para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                
                # if is_GPU and GPU_measure:
                    # res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print(nvgpu.gpu_info())
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # print('gpu utilization::', k,p, (res.used - init_memory)/ (1024**2))
# #                     GPU_utilization_list.append(res.gpu)
                    #
                    # GPU_mem_usage_list.append((res.used - init_memory)/ (1024**2))
                
#                 del updated_batch_X, mini_batch_Y, mini_batch_update_labels, batch_ids_with_changed_ids_curr_epoch
                
                
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            
            
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
#                 updated_grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(updated_grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            
#             t2 = time.time()
#                 
#             time1 += (t2 - t1)
            
            id_start = id_end
                        
            
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#     print('overhead4::', overhead4)
#     
#     print('overhead5::', overhead5)
    
    print('time 1::', time1)
    
    print('time 2::', time2)
    
    set_model_parameters(model, para)
    
    # if GPU_measure:
        # return model, updated_w_list, updated_grad_list, GPU_mem_usage_list
    # else:
    return model, updated_w_list, updated_grad_list


def model_update_deltagrad2_o2u(max_epoch, period, length, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, learning_rate, random_ids_multi_super_iterations, criterion, criterion_no_reduce, optimizer, regularization_coeff, is_GPU, device, batch_size=16, exp_updated_w_list = None, exp_updated_grad_list = None, compare = True):
    '''function to use deltagrad for incremental updates'''
    
    
    para = list(model.parameters())
    
    
    use_standard_way = False
    
    recorded = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    
    i = 0
    
    S_k_list = deque()
    
    Y_k_list = deque()
    
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     overhead4 = 0
#     
#     overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    

    updated_w_list = []
    
    updated_grad_list = []
    
    
#     res_para_list = []
#     
#     res_grad_list = []
    
#     t5 = time.time()
    
    '''detect which samples are removed from each mini-batch'''
    
    
#     t6 = time.time()
#     
#     overhead3 += (t6  -t5)
    
    '''main for loop of deltagrad'''
    
    i = 0
    
    t1 = 0
    
    t2 = 0
    
    time1 = 0 
    
    time2 = 0
    
    moving_loss_dic=torch.zeros(dataset_train.data.shape[0], dtype = torch.double, device = device)
#     for k in range(max_epoch):
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
#         random_ids_list = random_ids_list_all_epochs[k]
        
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        
#         X = dataset_train.data[random_ids]
#         Y = dataset_train.labels[random_ids]
#         
#         update_labels_curr_epoch = updated_labels[random_ids]
#         
#         ids_with_changed_ids_curr_epoch = ids_with_changed_ids[random_ids]
        
#         ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids[random_ids]
        
#         curr_entry_grad_list_epoch = all_entry_grad_list[k]
        
#         updated_grad = get_entry_grad_with_labels(update_labels_curr_epoch, curr_entry_grad_list)
        example_loss= torch.zeros_like(moving_loss_dic,dtype=torch.double)
#         for p in range(len(random_ids_list)):
        for p in range(0, dataset_train.lenth, batch_size):
            
#             curr_matched_ids = items[2]        
#             curr_matched_ids = random_ids_list[p]
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
#             t3 = time.time()
            
            curr_rand_ids = random_ids[j:end_id]
            
#                 batch_ids_with_unchanged_ids_curr_epoch = batch_ids_with_unchanged_ids_curr_epoch.to(device)
#             t1 = time.time()
#             
#             time2 += (t1 - t3)
#             curr_entry_grad_list = curr_entry_grad_list_epoch[j:end_id]
#             
#             if is_GPU:
#                 curr_entry_grad_list = curr_entry_grad_list.to(device)
            
#             learning_rate = learning_rate_all_epochs[i]
            
            
            old_lr = learning_rate    
                
            
            
            
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
            
            if i< curr_init_epochs or use_standard_way == True:
                
                
                batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
                batch_Y = dataset_train.labels[curr_rand_ids]
                batch_update_labels = updated_labels[curr_rand_ids]
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
                
                
                
                

                set_model_parameters(model, para, device)
                
                updated_w_list.append(get_model_para_list(model))
                
                loss_no_reduced = compute_derivative_one_more_step_unreduced_first(model, batch_X, batch_update_labels, criterion_no_reduce, optimizer)
                
                example_loss[curr_rand_ids] = loss_no_reduced
                
                curr_gradients = get_vectorized_grads(model, device)# get_all_vectorized_parameters1(model.get_all_gradient())
                
                updated_grad_list.append(curr_gradients.clone().cpu())
                
                if compare:
                    compute_para_grad_diff(get_vectorized_params(model), get_all_vectorized_parameters1(exp_updated_w_list[i]), curr_gradients, exp_updated_grad_list[i], para_list_GPU_tensor[cached_id])

                curr_changed_ids_grad = 0

                prev_changed_ids_grad = 0

                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_update_labels[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                            
                    curr_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_Y[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                    
                    prev_changed_ids_grad = get_vectorized_grads(model, device)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para, device)
                
                    if k>0 or (p > 0 and k == 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_para = prev_para.to(device)
                        
                        curr_s_list = (curr_para - prev_para)#+ 1e-16
                        
                        S_k_list.append(curr_s_list)
                        if len(S_k_list) > m:
                            removed_s_k = S_k_list.popleft()
                            
                            del removed_s_k
                        
#                     gradient_full = (expect_gradients*curr_remaining_id_size + gradient_remaining*curr_matched_ids_size)/(curr_remaining_id_size + curr_matched_ids_size)
                    gradient_full = curr_gradients

                    curr_grad_prev_labels = (gradient_full*batch_X.shape[0] - (curr_changed_ids_grad - prev_changed_ids_grad)*torch.sum(batch_ids_with_changed_ids_curr_epoch))/batch_X.shape[0]                


                    if k>0 or (p > 0 and k == 0):
                        
                        
                        prev_grad = grad_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_grad = prev_grad.to(device)
#                         prev_grad = updated_grad_list_GPU_tensor[cached_id]
#                         prev_grad = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list)
                        
#                         if is_GPU:
#                             prev_grad = prev_grad.to(device)
                        
                        Y_k_list.append(curr_grad_prev_labels - prev_grad + regularization_coeff*curr_s_list)
                        
                        if len(Y_k_list) > m:
                            removed_y_k = Y_k_list.popleft()
                            
                            del removed_y_k
                    
                    

                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*curr_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del curr_gradients
                    
                    if k>0 or (p > 0 and k == 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
                '''use l-bfgs algorithm to evaluate the gradients'''
                
#                 gradient_dual = None
    
#                 if not removed_batch_empty_list[i]:
                set_model_parameters(model, para)
                print(i, batch_update_labels.shape, batch_X.shape)
                loss_no_reduced = compute_unreduced_loss(model, batch_X, batch_update_labels, criterion_no_reduce)
                
                example_loss[curr_rand_ids] = loss_no_reduced
                
                
                updated_w_list.append(get_model_para_list(model))
                
                grad1 = 0
                
#                 batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
                batch_Y = dataset_train.labels[curr_rand_ids]
                
                batch_update_labels = updated_labels[curr_rand_ids]
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
                updated_batch_X = dataset_train.data[curr_rand_ids][batch_ids_with_changed_ids_curr_epoch]
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
#                     batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    updated_batch_X = updated_batch_X.to(device)
                    batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
#                 print(torch.norm(updated_batch_X - batch_X[batch_ids_with_changed_ids_curr_epoch]))
                
                
                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    compute_derivative_one_more_step(model, updated_batch_X, batch_update_labels[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                            
                    curr_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    compute_derivative_one_more_step(model, updated_batch_X, batch_Y[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                    
                    prev_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    grad1 = curr_changed_ids_grad - prev_changed_ids_grad
                
                prev_grad = grad_list_GPU_tensor[cached_id]
                
                if is_GPU:
                    prev_grad = prev_grad.to(device)
                    
                grad1 = (grad1*torch.sum(batch_ids_with_changed_ids_curr_epoch) + prev_grad*batch_X.shape[0])/batch_X.shape[0]
#                 grad1 = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list) 
                
                prev_para = para_list_GPU_tensor[cached_id]
                
#                 grad1 = updated_grad_list_GPU_tensor[cached_id] + regularization_coeff* #get_entry_grad_with_labels(batch_Y - batch_update_labels, curr_entry_grad_list)
                
                if is_GPU:
                    
                    prev_para = prev_para.to(device)
                grad1 = grad1 + regularization_coeff*prev_para
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                     
#                     gradient_dual = model.get_all_gradient()
                    
                with torch.no_grad():
                
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(para, device) - para_list_GPU_tensor[cached_id].to(device)))
                    
                    
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
                            
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                
                                
                                mat = mat.to(device)
                            
                    
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    else:
                        
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                    exp_gradient, exp_param = None, None
                    
                    if exp_updated_grad_list is not None and exp_updated_w_list is not None and len(exp_updated_grad_list) > 0 and len(exp_updated_w_list) > 0: 
                        exp_gradient, exp_param = exp_updated_grad_list[i], exp_updated_w_list[i]
                    
                    final_grad = torch.t(grad1.view(vec_para_diff.shape) + hessian_para_prod.view(vec_para_diff.shape))
                    
                    
#                     if gradient_dual is not None:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
#                         
#                     else:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    vec_para = get_all_vectorized_parameters1(para, device)
                    
                    curr_updated_grad = final_grad - regularization_coeff*vec_para
                    
                    updated_grad_list.append(curr_updated_grad.clone().cpu())
                    
                    if exp_param is not None:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, get_all_vectorized_parameters1(exp_param), para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    else:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, None, para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                
                
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            
            
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
#                 updated_grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(updated_grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            
#             t2 = time.time()
#                 
#             time1 += (t2 - t1)
            
            id_start = id_end
                        
        example_loss=example_loss - example_loss.mean()
        moving_loss_dic=moving_loss_dic+example_loss
        
        
    ordered_list, sorted_ids = torch.sort(moving_loss_dic.detach(), descending = True)    
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#     print('overhead4::', overhead4)
#     
#     print('overhead5::', overhead5)
    
    print('time 1::', time1)
    
    print('time 2::', time2)
    
    set_model_parameters(model, para)
    
    return moving_loss_dic, ordered_list, sorted_ids, updated_w_list, updated_grad_list
#     return model, updated_w_list, updated_grad_list



def construct_s_k_y_k_list(init_epochs, period, w_list, grad_list, updated_w_list, updated_grad_list, random_ids_multi_super_iterations, dataset_train, batch_size, m, regularization_rate, is_GPU, device):
    
    S_k_list = []
    
    Y_k_list = []
    
    mat_prime_list = []
    
    sigma_k_list = []
    
    i = 0
    
    explicit_iter_count = 0
    
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        

        for p in range(0, dataset_train.lenth, batch_size):
            
#             curr_matched_ids = items[2]        
#             curr_matched_ids = random_ids_list[p]
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
            t3 = time.time()
            
            curr_rand_ids = random_ids[j:end_id]
            
            t1 = time.time()
            
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
            
            if i< curr_init_epochs or use_standard_way == True:
                
                recorded += 1
                
                if k>0 or (p > 0 and k == 0):
                    
                    curr_para = get_all_vectorized_parameters1(updated_w_list[i])
                
                    prev_para = get_all_vectorized_parameters1(w_list[i])
                    
                    curr_s_list = (curr_para - prev_para)#+ 1e-16
                    
                    S_k_list.append(curr_s_list.clone())
                    
                    curr_grad = updated_grad_list[explicit_iter_count-1]
                    
                    prev_grad = grad_list[i]
                    
                    curr_grad_list = curr_grad - prev_grad + regularization_rate*curr_s_list
                    
                    Y_k_list.append(curr_grad_list.clone())
                    
                    
                    
                    del prev_para
                
                    del curr_para
                
                explicit_iter_count += 1

                
                if recorded >= 1:
                    use_standard_way = False
                
            else:
                
                if (i-curr_init_epochs) % period == 1:
                    print(i)
                    zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list[-m:], Y_k_list[-m:], i, init_epochs, m, is_GPU, device)
                    
                    mat = np.linalg.inv(mat_prime.cpu().numpy())
                    mat = torch.from_numpy(mat)
                    
                    
                    mat_prime_list.append(mat.clone())
                                    
                    sigma_k_list.append(sigma_k.clone())
                
            i = i + 1
            
            j += batch_size
            
            t2 = time.time()
    
    print('explicit_iter_count::', explicit_iter_count)
    
    return S_k_list, Y_k_list, sigma_k_list, mat_prime_list


def model_update_deltagrad3(max_epoch, period, length, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, learning_rate, random_ids_multi_super_iterations, batch_size, criterion, optimizer, regularization_coeff, is_GPU, device, exp_updated_w_list = None, exp_updated_grad_list = None, compare = True, S_k_list = None, Y_k_list = None, mat_prime_list = None, sigma_k_list = None):
    '''function to use deltagrad for incremental updates'''
    
    
    para = list(model.parameters())
    
    
    use_standard_way = False
    
    recorded = 0
    
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
    else:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)
    
    
    i = 0
    
    existing_hist = True
    
    if S_k_list is None:
        S_k_list = []
        existing_hist = False
    
    if Y_k_list is None:
        Y_k_list = []
        existing_hist = False
    
    if not existing_hist:
        mat_prime_list = []
        
        sigma_k_list = []
    
#     overhead2 = 0
#     
#     overhead3 = 0
#     
#     overhead4 = 0
#     
#     overhead5 = 0
    
    old_lr = 0
    
    cached_id = 0
    
    batch_id = 1
    
    
    random_ids_list_all_epochs = []

    removed_batch_empty_list = []
    

    updated_w_list = []
    
    updated_grad_list = []
    
    grad_origin_label_list = []
#     res_para_list = []
#     
#     res_grad_list = []
    
#     t5 = time.time()
    
    '''detect which samples are removed from each mini-batch'''
    
    
#     t6 = time.time()
#     
#     overhead3 += (t6  -t5)
    
    '''main for loop of deltagrad'''
    
    i = 0
    
    t1 = 0
    
    t2 = 0
    
    time1 = 0 
    
    time2 = 0
    
    explicit_iter_count = 0
#     for k in range(max_epoch):
    for k in range(len(random_ids_multi_super_iterations)):
    
        random_ids = random_ids_multi_super_iterations[k]
#         random_ids_list = random_ids_list_all_epochs[k]
        
        id_start = 0
    
        id_end = 0
        
        j = 0
        
        curr_init_epochs = init_epochs
        
#         X = dataset_train.data[random_ids]
#         Y = dataset_train.labels[random_ids]
#         
#         update_labels_curr_epoch = updated_labels[random_ids]
#         
#         ids_with_changed_ids_curr_epoch = ids_with_changed_ids[random_ids]
        
#         ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids[random_ids]
        
#         curr_entry_grad_list_epoch = all_entry_grad_list[k]
        
#         updated_grad = get_entry_grad_with_labels(update_labels_curr_epoch, curr_entry_grad_list)
        
#         for p in range(len(random_ids_list)):


        for p in range(0, dataset_train.lenth, batch_size):
            
#             curr_matched_ids = items[2]        
#             curr_matched_ids = random_ids_list[p]
            
            end_id = p + batch_size
            
            if end_id > dataset_train.lenth:
                end_id = dataset_train.lenth
            
            t3 = time.time()
            
            curr_rand_ids = random_ids[j:end_id]
            
#                 batch_ids_with_unchanged_ids_curr_epoch = batch_ids_with_unchanged_ids_curr_epoch.to(device)
            t1 = time.time()
            
            time2 += (t1 - t3)
#             curr_entry_grad_list = curr_entry_grad_list_epoch[j:end_id]
#             
#             if is_GPU:
#                 curr_entry_grad_list = curr_entry_grad_list.to(device)
            
#             learning_rate = learning_rate_all_epochs[i]
            
            
            old_lr = learning_rate    
                
            
            
            
            if (i-curr_init_epochs)%period == 0:
                
                recorded = 0
                
                use_standard_way = True
            
            if i< curr_init_epochs or use_standard_way == True:
                
                
                batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
                batch_Y = dataset_train.labels[curr_rand_ids]
                batch_update_labels = updated_labels[curr_rand_ids]
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
                
                
                
                

                set_model_parameters(model, para, device)

                updated_w_list.append(get_model_para_list(model))
                
                compute_derivative_one_more_step(model, batch_X, batch_update_labels, criterion, optimizer)
                
                curr_gradients = get_vectorized_grads(model, device)# get_all_vectorized_parameters1(model.get_all_gradient())
                
                updated_grad_list.append(curr_gradients.clone().cpu())
                
                explicit_iter_count += 1
                
                if compare:
                    compute_para_grad_diff(get_vectorized_params(model), get_all_vectorized_parameters1(exp_updated_w_list[i]), curr_gradients, exp_updated_grad_list[i], para_list_GPU_tensor[cached_id])

                curr_changed_ids_grad = 0

                prev_changed_ids_grad = 0

                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_update_labels[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                            
                    curr_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    compute_derivative_one_more_step(model, batch_X[batch_ids_with_changed_ids_curr_epoch], batch_Y[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                    
                    prev_changed_ids_grad = get_vectorized_grads(model, device)
                
                with torch.no_grad():
                               
                
                    curr_para = get_all_vectorized_parameters1(para, device)
                
                    if k>0 or (p > 0 and k == 0):
                        
                        prev_para = para_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_para = prev_para.to(device)
                        
                        curr_s_list = (curr_para - prev_para)#+ 1e-16
                        
                        if not existing_hist:
                            S_k_list.append(curr_s_list.cpu().clone())
#                         if len(S_k_list) > m:
#                             removed_s_k = S_k_list.popleft()
#                             
#                             del removed_s_k
                        
#                     gradient_full = (expect_gradients*curr_remaining_id_size + gradient_remaining*curr_matched_ids_size)/(curr_remaining_id_size + curr_matched_ids_size)
                    gradient_full = curr_gradients

                    curr_grad_prev_labels = (gradient_full*batch_X.shape[0] - (curr_changed_ids_grad - prev_changed_ids_grad)*torch.sum(batch_ids_with_changed_ids_curr_epoch))/batch_X.shape[0]                


                    if k>0 or (p > 0 and k == 0):
                        
                        
                        prev_grad = grad_list_GPU_tensor[cached_id]
                        
                        if is_GPU:
                            prev_grad = prev_grad.to(device)
#                         prev_grad = updated_grad_list_GPU_tensor[cached_id]
#                         prev_grad = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list)
                        
#                         if is_GPU:
#                             prev_grad = prev_grad.to(device)
                        if not existing_hist:
                            curr_Y_k_list = curr_grad_prev_labels - prev_grad + regularization_coeff*curr_s_list
                            Y_k_list.append(curr_Y_k_list.cpu().clone())
                        grad_origin_label_list.append(curr_grad_prev_labels)
#                         if len(Y_k_list) > m:
#                             removed_y_k = Y_k_list.popleft()
#                             
#                             del removed_y_k
                    
                    

                    para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*curr_gradients, full_shape_list, shape_list)
                    
                    recorded += 1
                    
                    
                    del gradient_full
                    
                    del curr_gradients
                    
                    if k>0 or (p > 0 and k == 0):
                        del prev_para
                    
                        del curr_para
                    
                    if recorded >= length:
                        use_standard_way = False
                
                
            else:
                
                '''use l-bfgs algorithm to evaluate the gradients'''
                
#                 gradient_dual = None
    
#                 if not removed_batch_empty_list[i]:
                set_model_parameters(model, para)
                
                updated_w_list.append(get_model_para_list(model))
                
                grad1 = 0
                
#                 batch_X = dataset_train.data[curr_rand_ids]
            
#             batch_Y = dataset_train.labels[random_ids[j:end_id]]
                batch_Y = dataset_train.labels[curr_rand_ids]
                
                batch_update_labels = updated_labels[curr_rand_ids]
                
                batch_ids_with_changed_ids_curr_epoch = ids_with_changed_ids[curr_rand_ids]
                
                updated_batch_X = dataset_train.data[curr_rand_ids][batch_ids_with_changed_ids_curr_epoch]
                
    #             batch_ids_with_unchanged_ids_curr_epoch = ids_with_unchanged_ids_curr_epoch[j:end_id]
                
                if is_GPU:
#                     batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    updated_batch_X = updated_batch_X.to(device)
                    batch_update_labels = batch_update_labels.to(device)
                    batch_ids_with_changed_ids_curr_epoch = batch_ids_with_changed_ids_curr_epoch.to(device)
#                 print(torch.norm(updated_batch_X - batch_X[batch_ids_with_changed_ids_curr_epoch]))
                
                
                if torch.sum(batch_ids_with_changed_ids_curr_epoch).item() > 0:
                    compute_derivative_one_more_step(model, updated_batch_X, batch_update_labels[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                            
                    curr_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    compute_derivative_one_more_step(model, updated_batch_X, batch_Y[batch_ids_with_changed_ids_curr_epoch], criterion, optimizer)
                    
                    prev_changed_ids_grad = get_vectorized_grads(model, device)
                    
                    grad1 = curr_changed_ids_grad - prev_changed_ids_grad
                
                prev_grad = grad_list_GPU_tensor[cached_id]
                
                if is_GPU:
                    prev_grad = prev_grad.to(device)
                    
                grad1 = (grad1*torch.sum(batch_ids_with_changed_ids_curr_epoch) + prev_grad*batch_X.shape[0])/batch_X.shape[0]
#                 grad1 = get_entry_grad_with_labels(batch_update_labels, curr_entry_grad_list) 
                
                prev_para = para_list_GPU_tensor[cached_id]
                
#                 grad1 = updated_grad_list_GPU_tensor[cached_id] + regularization_coeff* #get_entry_grad_with_labels(batch_Y - batch_update_labels, curr_entry_grad_list)
                
                if is_GPU:
                    
                    prev_para = prev_para.to(device)
                grad1 = grad1 + regularization_coeff*prev_para
#                     compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer)
#                     
#                     gradient_dual = model.get_all_gradient()
                    
                with torch.no_grad():
                
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(para, device) - para_list_GPU_tensor[cached_id].to(device)))
                    
                    if not existing_hist:
                        if (i-curr_init_epochs)//period >= 1:
                            if (i-curr_init_epochs) % period == 1:
                                zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list[-m:], Y_k_list[-m:], i, init_epochs, m, is_GPU, device)
#                                 print(len(S_k_list), i)
                                mat = np.linalg.inv(mat_prime.cpu().numpy())
                                mat = torch.from_numpy(mat)
                                
                                mat_prime_list.append(mat.cpu().clone())
                                
                                sigma_k_list.append(sigma_k.cpu().clone())
                                
                                if is_GPU:
                                    
                                    
                                    mat = mat.to(device)
                                
                        
                            hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                            
                        else:
                            
                            hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list[-m:], Y_k_list[-m:], i, init_epochs, m, vec_para_diff, period, is_GPU, device)
                            
                            if (i-curr_init_epochs)//period == 0 and (i-curr_init_epochs) % period == 1:
#                                 mat = np.linalg.inv(mat_prime.cpu().numpy())
#                                 mat = torch.from_numpy(mat)
                                
                                mat_prime_list.append(mat_prime.cpu().clone())
                                
                                sigma_k_list.append(sigma_k.cpu().clone())
                            
#                             mat_prime_list.append(mat.clone())
#                                 
#                             sigma_k_list.append(sigma_k.clone())
                    else:
                        
                        curr_Y_k = Y_k_list[explicit_iter_count-1-m:explicit_iter_count-1]
                        
                        curr_S_k = S_k_list[explicit_iter_count-1-m:explicit_iter_count-1]
                        
#                         print(explicit_iter_count-curr_init_epochs, i, len(sigma_k_list))
                        
                        sigma_k = sigma_k_list[explicit_iter_count-curr_init_epochs-1]
                        
                        mat = mat_prime_list[explicit_iter_count-curr_init_epochs-1]
                        
                        if is_GPU:
                            curr_Y_k = curr_Y_k.to(device)
                            curr_S_k = curr_S_k.to(device)
                            sigma_k = sigma_k.to(device)
                            mat = mat.to(device)
                        
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(m, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        
                    exp_gradient, exp_param = None, None
                    
                    if exp_updated_grad_list is not None and exp_updated_w_list is not None and len(exp_updated_grad_list) > 0 and len(exp_updated_w_list) > 0: 
                        exp_gradient, exp_param = exp_updated_grad_list[i], exp_updated_w_list[i]
                    
                    final_grad = torch.t(grad1.view(vec_para_diff.shape) + hessian_para_prod.view(vec_para_diff.shape))
                    
                    
#                     if gradient_dual is not None:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
#                         
#                     else:
#                         is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    
                    vec_para = get_all_vectorized_parameters1(para, device)
                    
                    curr_updated_grad = final_grad - regularization_coeff*vec_para
                    
                    updated_grad_list.append(curr_updated_grad.clone().cpu())
                    
                    if exp_param is not None:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, get_all_vectorized_parameters1(exp_param), para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    else:
                        vec_para = update_para_final2(vec_para, final_grad, learning_rate, regularization_coeff, exp_gradient, None, para_list_GPU_tensor[cached_id], is_GPU, device, compare=compare)
                    
                    
                    para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                
                
            i = i + 1
            
            j += batch_size
            
            
            cached_id += 1
            
            
            
            
            if cached_id%cached_size == 0:
                
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                
                if GPU_tensor_end_id > para_list_all_epoch_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epoch_tensor.shape[0] 
#                 print("end_tensor_id::", GPU_tensor_end_id)
                
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(grad_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
#                 updated_grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(updated_grad_list_all_epoch_tensor[batch_id*cached_size:GPU_tensor_end_id])
                
                batch_id += 1
                
                cached_id = 0
            
            
            t2 = time.time()
                
            time1 += (t2 - t1)
            
            id_start = id_end
                        
            
#     print('overhead::', overhead)
#     
#     print('overhead2::', overhead2)
#     
#     print('overhead3::', overhead3)
#     
#     print('overhead4::', overhead4)
#     
#     print('overhead5::', overhead5)
    
    print('time 1::', time1)
    
    print('time 2::', time2)
    
    print('explicit_iter_count 0::', explicit_iter_count)
    
    set_model_parameters(model, para)
    
    if existing_hist:
        return model, updated_w_list, updated_grad_list, None, None, None, None, grad_origin_label_list
    else:
        return model, updated_w_list, updated_grad_list, S_k_list, Y_k_list, mat_prime_list, sigma_k_list, grad_origin_label_list
  
if __name__ == '__main__':

#     args = {'bs':100, 'lr':0.5, 'n_epochs':100, 'device':'cpu'}
    
    args = parse_optim_del_args()
    
    print(args)
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'output/')
    
    
    data_preparer = models.Data_preparer()
    
#     model_class = getattr(sys.modules[__name__], args.model)
    
    dataset_name = args.dataset
    
#     full_output_dir = os.path.join(args.output_dir, dataset_name)
#         
#     if not os.path.exists(full_output_dir):
#         os.makedirs(full_output_dir)
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    args.num_class = num_class
    
#     labeled_training_dataset, unlabeled_training_dataset, validation_dataset, dataset_test  = obtain_mnist_examples(args)
    
#     labeled_training_dataset, unlabeled_training_dataset, validation_dataset, dataset_test  = obtain_mnist_examples2(args)
    
#     obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
# #     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
#     
#     training_dataset, val_dataset, test_dataset, full_output_dir,_ = obtain_data_function(args)
    
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, small_dataset, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)
    
    size = None
#     
#     size = 2000
#     
#     full_training_noisy_dataset.data = full_training_noisy_dataset.data[0:size]
#     
#     full_training_noisy_dataset.labels = full_training_noisy_dataset.labels[0:size]
#     
#     full_training_noisy_dataset.lenth = size   
#     
#     full_training_origin_labels = full_training_origin_labels[0:size]
    
    if args.start:
        
        w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False)
    
        torch.save(model, full_out_dir + '/model')
        
        torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        
        torch.save(w_list, full_out_dir + '/w_list')
        
        torch.save(grad_list, full_out_dir + '/grad_list')
        
        all_entry_grad_list = obtain_gradients_each_class(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, args.GPU, args.device)
        
        torch.save(all_entry_grad_list, full_out_dir + '/all_entry_grad_list')
    
    else:
        
        
        model = torch.load(full_out_dir + '/model')

        model = model.to(args.device)

        optimizer = model.get_optimizer(args.tlr, args.wd)
        
        random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
        
        w_list = torch.load(full_out_dir + '/w_list')
        
        grad_list = torch.load(full_out_dir + '/grad_list')
        
        all_entry_grad_list = torch.load(full_out_dir + '/all_entry_grad_list')
        
        updated_labels = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, size = size)
        
#         updated_labels = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.labels, full_training_origin_labels, size = size)
        
#         train_model_dataset(model, optimizer, None, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True)
        
        
        
        exp_updated_origin_grad_list = update_gradient_origin(all_entry_grad_list, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.GPU, args.device, w_list, model, optimizer)
        
        updated_origin_grad_list = update_gradient_incremental(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device, exp_updated_origin_grad_list)
        
#         obtain_updated_gradients(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, all_entry_grad_list, updated_labels, args.GPU, args.device)
    
    
        period = 5#args.period
            
        init_epochs = 5#args.init
        
        m = 3#args.m
        
        cached_size = 10000#args.cached_size
        
        
#         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
        
        grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list, updated_origin_grad_list)
        
#             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
        
        t1 = time.time()
        
        '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
        
        criterion = model.soft_loss_function_reduce
        
        set_model_parameters(model, w_list[0], args.device)
        
        exp_updated_w_list, exp_updated_grad_list,_ = train_model_dataset(model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True)
        
        set_model_parameters(model, w_list[0], args.device)
        
        updated_model = model_update_deltagrad(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor,updated_grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list)

    
    
    
    
    
#     remaining_AFs = torch.load(full_out_dir + '/remaining_AFs')

               
#                 batch_y = batch_y.type(torch.double)
        