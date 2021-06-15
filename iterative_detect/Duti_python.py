'''
Created on Feb 4, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob
from pip._internal.cli.cmdoptions import pre
# from iterative_detect.utils_iters import incremental_suffix
# from iterative_detect.utils_iters import prepare_approx_hessian_vec_prod

# from iterative_detect.utils_iters import origin_compute_sample_wise_gradients



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/real_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Reweight_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from utils_iters import *

try:
    from utils.utils import *
    from real_examples.utils_real import *
    from models.util_func import *
    from Reweight_examples.utils_reweight import *
except ImportError:
    from utils import *
    from utils_real import *
    from util_func import *
    from utils_reweight import *

import models
import math


sum_upper_bound = 5

final_sum_upper_bound = 0.2

def duti_cls(X_train, Y_train, X_valid, Y_valid, regularization_coeff, budget):
    n = X_train.shape[0]
    m = X_valid.shape[0]

def get_train_loss(criterion2, meta_train_outputs, all_y_tensor, eps):
    meta_train_loss = 0
                
    for p in range(args.num_class):
        
        meta_train_loss += torch.sum(criterion2(meta_train_outputs, all_y_tensor[p].view(1,-1).repeat(meta_train_outputs.shape[0], 1).view(-1)).view(-1)*eps[:,p])

    meta_train_loss = meta_train_loss/meta_train_outputs.shape[0]

    return meta_train_loss

def get_scatter_labels(labels, eps):
    unique_labels = torch.argmax(labels, dim = 1).view(-1)
#     onehot()
#     scattered_res = torch.scatter(eps, dim = 1, index = unique_labels.view(-1)).view(-1)
    scattered_res = torch.gather(eps, dim=1, index = unique_labels.view(-1,1))
    return scattered_res

#x_train->K, x_tilde -> K_tilde
def reweight_alg_duti(train_dataset, meta_dataset, test_dataset, training_DL, val_DL, test_DL, args, model, full_existing_labeled_ids, gamma=3, r_weight = None):
    
    bz = 1000
    
    training_data_count = 0
    
    if training_DL is None:
        train_dataset.lenth = train_dataset.data.shape[0]
        train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
        
        if test_dataset is not None:
            test_dataset.lenth = test_dataset.data.shape[0]
            test_loader =  DataLoader(test_dataset, batch_size=bz, shuffle=False)
        else:
            test_loader = None
            
        meta_dataset.lenth = meta_dataset.data.shape[0]
        meta_loader = DataLoader(meta_dataset, batch_size=bz, shuffle=True)
        valid_loader = DataLoader(meta_dataset, batch_size=bz, shuffle=True)
        
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
        
    
#     y_array =torch.rand([training_data_count], dtype = training_data_type, requires_grad=True)
    
    y_array = train_loader.dataset.labels.clone()
    
    y_array.requires_grad = True
    
    labeled_bool_tensor = torch.ones(train_loader.dataset.labels.shape[0]).bool()
    
    labeled_bool_tensor[full_existing_labeled_ids] = False
    
    mu_array = torch.rand([training_data_count], dtype = training_data_type, requires_grad=True)
    
    all_y_labels = []
    
    for p in range(args.num_class):
#         all_y_labels.append(onehot(torch.tensor([p]), args.num_class).view(-1))
        all_y_labels.append(torch.tensor([p]))
    
    
    all_y_tensor = torch.stack(all_y_labels, dim = 0)
    
    if args.GPU:
        all_y_tensor = all_y_tensor.to(args.device)
     
    beta_array = torch.rand(y_array.shape, dtype = training_data_type, requires_grad=True)
    
    gamma_array = torch.rand(y_array.shape, dtype = training_data_type, requires_grad=True)
     
#     gamma = torch.rand(1, dtype = training_data_type, requires_grad=True, device = args.device)
    
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
                                
            if r_weight is not None:
                curr_r_weight = r_weight[ids]
            
            y_array.requires_grad = True
            mu_array.requires_grad = True
#             
            beta_array.requires_grad = True
#             
            gamma_array.requires_grad = True
            
            
            opt.zero_grad()
            with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
                # 1. Update meta model on training data
                meta_train_outputs = meta_model(inputs)
                
                eps = y_array[ids]
                
                if args.GPU:
                    eps = eps.to(args.device)
                
                criterion2.reduction = 'none'
                
                meta_train_loss = get_train_loss(criterion2, meta_train_outputs, all_y_tensor, eps)
                
#                 meta_train_loss = 0
#                 
#                 for p in range(args.num_class):
#                     
#                     meta_train_loss += torch.sum(criterion2(meta_train_outputs, all_y_tensor[p].view(1,-1).repeat(meta_train_outputs.shape[0], 1)).view(-1)*eps[:,p])
                    
                    
#                 eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=args['device'])

                
                
                
                
#                 t1 = time.time()
#                 meta_train_loss = torch.sum(eps.view(-1) * meta_train_loss.view(-1))/torch.sum(eps)
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
                
                if gamma is None:
                    meta_val_loss += (meta_train_loss)
                else:
#                     meta_val_loss += (meta_train_loss + gamma/eps.shape[0]*torch.sum(1 - eps[:, labels]))
#                     meta_val_loss += (meta_train_loss + gamma/eps.shape[0]*torch.sum(1 - get_scatter_labels(labels, eps)))
                    
                    meta_val_loss += (meta_train_loss + gamma/eps.shape[0]*torch.sum(torch.norm(labels - eps, dim = 1, p=1)))
                
                sub_mu_array = mu_array[ids]
                
                sub_beta_array = beta_array[ids]
                
                sub_gamma_array = gamma_array[ids]
                
                if args.GPU:
                    sub_mu_array = sub_mu_array.to(args.device)
                    sub_beta_array = sub_beta_array.to(args.device)
                    sub_gamma_array = sub_gamma_array.to(args.device)
#                 meta_val_loss = torch.sum(eps.view(-1) * meta_val_loss.view(-1) - eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
                
                total_meta_val_loss = meta_val_loss + torch.sum((1 - torch.sum(eps, dim = 1)).view(-1)*sub_mu_array.view(-1))/eps.shape[0]+ torch.sum(sub_beta_array.view(-1)*(eps.view(-1)-1))/eps.shape[0] + torch.sum(sub_gamma_array.view(-1)*(-eps.view(-1)))/eps.shape[0] 
                
                
                eps_grads = torch.autograd.grad(total_meta_val_loss, eps, retain_graph=True)[0].detach()
                
                if gamma is None:
                    gamma = torch.max(get_scatter_labels(labels, eps_grads))
                
#                 mu_grads = torch.autograd.grad(total_meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
#                 
#                 beta_grads = torch.autograd.grad(meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
#                 
#                 gamma_grads = torch.autograd.grad(meta_val_loss, gamma)[0].detach()
    
            # 3. Compute weights for current training batch
            y_array.requires_grad = False
            
#             if 25657 in ids:
#                 print('here')
            
            y_array[ids[labeled_bool_tensor[ids]]] = y_array[ids][labeled_bool_tensor[ids]]-lr*eps_grads[labeled_bool_tensor[ids]].cpu()
            
            #


#             
#             if max_w is None:
#                 max_w = torch.max(w_array).item()
#                 min_w = torch.min(w_array).item()
#             else:
#                 max_w = max(torch.max(w_array).item(), max_w)
#                 min_w = min(torch.min(w_array).item(), min_w)
            
            
#             if torch.abs(training_data_count - removed_count - torch.sum(w_array)) < sum_upper_bound and torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
#                 lr = lr /2
#                 sum_upper_bound = sum_upper_bound/5
#                 single_upper_bound = single_upper_bound/5 
#             
#             if torch.max(w_array) - 1 < single_upper_bound and torch.min(w_array) > -single_upper_bound:
#                 upper_lower_bound_satisfied = True
#             
#             else:
#                 upper_lower_bound_satisfied = False
            
            
#             total_meta_val_loss = meta_val_loss + torch.sum(- eps.view(-1)*sub_mu_array.view(-1) + sub_beta_array.view(-1)*(eps.view(-1)-1) + gamma*((training_data_count - removed_count)/training_data_count - eps))/torch.sum(eps)
#             total_meta_val_loss = meta_val_loss + torch.sum(torch.sum(1 - eps, dim = 1).view(-1)*sub_mu_array.view(-1))/eps.shape[0]
            total_meta_val_loss = meta_val_loss + torch.sum((1 - torch.sum(eps, dim = 1)).view(-1)*sub_mu_array.view(-1))/eps.shape[0]+ torch.sum(sub_beta_array.view(-1)*(eps.view(-1)-1))/eps.shape[0] + torch.sum(sub_gamma_array.view(-1)*(-eps.view(-1)))/eps.shape[0]
            
            mu_grads = torch.autograd.grad(total_meta_val_loss, sub_mu_array, retain_graph=True)[0].detach()
                
            beta_grads = torch.autograd.grad(total_meta_val_loss, sub_beta_array, retain_graph=True)[0].detach()
#             
            gamma_grads = torch.autograd.grad(total_meta_val_loss, sub_gamma_array)[0].detach()
            
            
            mu_array.requires_grad = False
            
            beta_array.requires_grad = False
            
            gamma_array.requires_grad = False
            
            mu_array[ids[labeled_bool_tensor[ids]]] = mu_array[ids][labeled_bool_tensor[ids]] + lr*mu_grads[labeled_bool_tensor[ids]].cpu()
             
            beta_array[ids[labeled_bool_tensor[ids]]] = torch.clamp(beta_array[ids][labeled_bool_tensor[ids]] + lr*beta_grads[labeled_bool_tensor[ids]].cpu(), min = 0)
            
            gamma_array[ids[labeled_bool_tensor[ids]]] = torch.clamp(gamma_array[ids][labeled_bool_tensor[ids]] + lr*gamma_grads[labeled_bool_tensor[ids]].cpu(), min = 0)
#
#             gamma = torch.clamp(gamma + lr*gamma_grads, min = -1, max = 1)
            
            

            
#             mu_array[ids] = torch.clamp(mu_array[ids] + lr*(-w_array[ids]/torch.sum(w_array[ids])), min = 0)
#              
#             beta_array[ids] = torch.clamp(beta_array[ids] + lr*((w_array[ids]-1)/torch.sum(w_array[ids])), min = 0)
#              
#             gamma = gamma + lr*(torch.sum((training_data_count - removed_count)/training_data_count-w_array[ids])/torch.sum(w_array[ids]))
            
#             w_array[ids] = torch.clamp(w_array[ids]-lr*eps_grads, min=0, max=1)
            
#             w_array[ids] = w_array[ids]-lr*eps_grads
            
            y_array[ids] = torch.clamp(y_array[ids], min=1e-7, max = 1)
            
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
            
            curr_w = y_array[ids]
            
            if args.GPU:
                curr_w = curr_w.to(args.device)
            
            minibatch_loss = get_train_loss(criterion2, outputs, all_y_tensor, curr_w)
            
            
#             minibatch_loss = criterion2(outputs, labels)
#             
#             
#             
#             minibatch_loss = torch.sum(curr_w * minibatch_loss.view(-1))/torch.sum(curr_w)
            
            train_loss += minibatch_loss.detach().cpu()*inputs.shape[0]
            minibatch_loss.backward()
            opt.step()
    
            # keep track of epoch loss/accuracy
            
#             pred_labels = (F.sigmoid(outputs) > 0.5).int()
            
#             pred_labels = (outputs > 0.5).int()
#             
#             train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
        global sum_upper_bound
        if torch.sum(torch.abs(torch.sum(y_array, dim = 1) - 1)) < sum_upper_bound:
            lr = lr/5
            sum_upper_bound = sum_upper_bound/2
                
#         if torch.sum(torch.abs(torch.sum(y_array, dim = 1) - 1)) < final_sum_upper_bound:
        if lr < 0.1:
            break
    
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
            train_loss = train_loss/training_data_count#, train_acc/len(train_dataset)      
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
    
#         sorted_w_array, sorted_ids = torch.sort(w_array.view(-1), descending=True)
#         
#         print('sorted array::')
#         
#         print(sorted_w_array[0:removed_count])
#         
#         print(sorted_w_array[training_data_count - removed_count:])
#         
#         print('sorted array ids::')
#         
#         print(sorted_ids[training_data_count - removed_count:])
#         
#         
#         
#         torch.save(sorted_ids[training_data_count - removed_count:], full_output_dir + '/removed_ids_' + str(pid) + '_' + str(ep))
        
#         print(sorted_ids[0:train_dataset.data.shape[0] - removed_count])
        print('learning rate::', lr)
        
        print('r_sum_gap::', torch.sum(torch.abs(torch.sum(y_array, dim = 1) - 1)))    
        
        print('mu::', torch.max(mu_array), torch.min(mu_array))
#         
#         print('beta::', torch.max(beta_array), torch.min(beta_array))
#         
#         print('w::', max_w, min_w)
#         
#         print('gamma::', gamma)    
#         
#         print('here')
#         
#         if upper_lower_bound_satisfied and torch.abs(torch.sum(w_array) - (training_data_count - removed_count)) < final_sum_upper_bound:
#             break

    return y_array, gamma

if __name__ == '__main__':
    
    
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
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    # full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, small_dataset, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, train_annotated_label_tensor, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids, train_origin_dataset, valid_origin_dataset, test_origin_dataset = obtain_data_function(args, noisy=True, load_origin= False, is_tars = args.tars)
    
    random_ids_multi_super_iterations = None#torch.load(full_out_dir + '/random_ids_multi_super_iterations')

    
    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations)


    model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
    valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation', args.GPU, args.device, f1=False)
     
    valid_model_dataset(model, dataset_test, None, args.bz, 'init test', args.GPU, args.device, f1=False)

    valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
     
    valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)






    origin_noisy_id_tensor = torch.load(full_out_dir + '/noisy_sample_ids')
    
    origin_clean_id_tensor = torch.load(full_out_dir + '/clean_sample_ids')
    
    if type(origin_noisy_id_tensor) is list or type(origin_noisy_id_tensor) is set:
        origin_noisy_id_tensor = torch.tensor(list(origin_noisy_id_tensor))
        torch.save(origin_noisy_id_tensor, full_out_dir + '/noisy_sample_ids')
    
    if type(origin_clean_id_tensor) is list or type(origin_clean_id_tensor) is set:
        origin_clean_id_tensor = torch.tensor(list(origin_clean_id_tensor))
        torch.save(origin_clean_id_tensor, full_out_dir + '/clean_sample_ids')
    
    r_weight = torch.ones(full_training_noisy_dataset.data.shape[0],dtype = full_training_noisy_dataset.data.dtype)
            
    # if os.path.exists(full_out_dir + '/clean_sample_ids'):
        # origin_labeled_tensor = torch.load(full_out_dir + '/clean_sample_ids')
    r_weight *= args.regular_rate
    
    r_weight[origin_clean_id_tensor] = 1
    
    
    
    
    y_array, gamma = reweight_alg_duti(full_training_noisy_dataset, validation_dataset, dataset_test, None, None, None, args, model,origin_clean_id_tensor, r_weight = r_weight)
    
    
    selected_clean_sample_ids = torch.sort(torch.norm((y_array - full_training_noisy_dataset.labels)[origin_noisy_id_tensor],dim =1), descending = True)[1][0:args.removed_count]
    
    final_selected_clean_samples = origin_noisy_id_tensor[selected_clean_sample_ids]
    
    cleaned_labels = full_training_noisy_dataset.labels.clone()
    
    cleaned_labels_origin = full_training_origin_labels[final_selected_clean_samples]
    
    
    
    cleaned_labels[final_selected_clean_samples[cleaned_labels_origin!=-1]] = onehot(cleaned_labels_origin[cleaned_labels_origin != -1], args.num_class).type(torch.double)
    
    cleaned_training_dataset = models.MyDataset(full_training_noisy_dataset.data, cleaned_labels)
    
    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(cleaned_training_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations)
    
    
    model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
    valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation', args.GPU, args.device, f1=False)
     
    valid_model_dataset(model, dataset_test, None, args.bz, 'init test', args.GPU, args.device, f1=False)

    valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
     
    valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)
    
    
    
    
    print('here')
    
    
    