'''
Created on Jan 9, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob



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
    from models.utils_real import *
    from models.util_func import *
    from Reweight_examples.utils_reweight import *
except ImportError:
    from utils import *
    from utils_real import *
    from util_func import *
    from utils_reweight import *

import models
import math

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
    
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
    
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, small_dataset, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)
    
    
    iteration_count = 5
    
    remove_different_version_dataset(full_out_dir)
    
    
    model_para_list = []
    
    if args.start:
    
    
        w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
    
        full_existing_labeled_id_tensor = None
        
        for k in range(iteration_count):
        
            print('start labeling samples::', k)
            
            most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)

            torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids')
        
            torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight')
            
            if k == 0:
                updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = True)
            else:
                updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False)
                
            full_training_noisy_dataset.labels = updated_labels
            
            
            w_list, grad_list, _, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations)
        
            curr_model_para =get_model_para_list(model)
        
            model_para_list.append(curr_model_para)
        
            torch.save(model, full_out_dir + '/model_' + str(k))
            
            torch.save(w_list, full_out_dir + '/w_list_' + str(k))
            
            torch.save(grad_list, full_out_dir + '/grad_list_' + str(k))
            
            torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + str(k))
            
    #         torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
    #         
    #         torch.save(w_list, full_out_dir + '/w_list')
    #         
    #         torch.save(grad_list, full_out_dir + '/grad_list')
            
    #         all_entry_grad_list = obtain_gradients_each_class(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, args.GPU, args.device)
    #         
    #         torch.save(all_entry_grad_list, full_out_dir + '/all_entry_grad_list')
            
    #         torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor')
            
            
    #         torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_v0')
            
    #         torch.save(iter_count, full_out_dir + '/labeling_iter')
            
    
        torch.save(full_existing_labeled_id_tensor, full_out_dir + '/' + args.model + 'full_existing_labeled_id_tensor') 
    
    
    else:
        m = 3
#         remaining_sample_ids = torch.load(full_out_dir + '/remaining_sample_ids')
#         m = 3
#         
# #         for k in range(iteration_count):
#         k = 1
#         w_list = torch.load(full_out_dir + '/w_list_' + str(k))
#          
#         grad_list = torch.load(full_out_dir + '/grad_list_' + str(k))
#          
#         w1_list = torch.load(full_out_dir + '/w_list_' + str(k+1))
#          
#         grad1_list = torch.load(full_out_dir + '/grad_list_' + str(k+1))
#          
#          
#         S_k_list = []
#          
#         Y_k_list = []
#         
#         X = full_training_noisy_dataset.data[remaining_sample_ids[0]:remaining_sample_ids[0]+1]
#            
#         Y = full_training_noisy_dataset.labels[remaining_sample_ids[0]:remaining_sample_ids[0]+1]
#         
# #         X = full_training_noisy_dataset.data[remaining_sample_ids]
# #           
# #         Y = full_training_noisy_dataset.labels[remaining_sample_ids]
#         
#         model = torch.load(full_out_dir + '/model_' + str(k))
#         
#         optimizer = model.get_optimizer(args.tlr, args.wd)
#         
#         loss_func = model.soft_loss_function_reduce
#         
#         for r in range(m):
#              
#             curr_s_k = get_all_vectorized_parameters1(w1_list[-(m-r)], args.device) - get_all_vectorized_parameters1(w_list[-(m-r)], args.device)
#              
#             S_k_list.append(curr_s_k.view(1,-1))
#             
#             optimizer.zero_grad()
#             
#             set_model_parameters(model, w1_list[-(m-r)], args.device)
#             
#             curr_loss = loss_func(model(X), Y)
#             
#             curr_loss.backward()
#             
#             grad1 = get_vectorized_grads(model, args.device)
#             
#             optimizer.zero_grad()
#             
#             set_model_parameters(model, w_list[-(m-r)], args.device)
#             
#             curr_loss = loss_func(model(X), Y)
#             
#             curr_loss.backward()
#             
#             grad2 = get_vectorized_grads(model, args.device)
#             
#              
# #             Y_k_list.append(grad1_list[-(m-r)].view(1,-1) - grad_list[-(m-r)].view(1,-1) + args.wd*curr_s_k)
#             
#             Y_k_list.append(grad1.view(1,-1) - grad2.view(1,-1) + args.wd*curr_s_k)
#          
#         w2_list = torch.load(full_out_dir + '/w_list_' + str(k+2))
#          
#         grad2_list = torch.load(full_out_dir + '/grad_list_' + str(k+2))
#          
#         vec_para = get_all_vectorized_parameters1(w2_list[-1], args.device) - get_all_vectorized_parameters1(w_list[-1], args.device)
#          
#         hessian_para_prod_0, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, 0,0, m, vec_para.view(-1,1), 0,args.GPU, args.device)
#         
#         optimizer.zero_grad()     
#         
#         set_model_parameters(model, w2_list[-1], args.device)
#             
#         curr_loss = loss_func(model(X), Y)
#         
#         curr_loss.backward()
#         
#         grad1 = get_vectorized_grads(model, args.device)
#         
#         optimizer.zero_grad()
#         
#         set_model_parameters(model, w_list[-1], args.device)
#         
#         curr_loss = loss_func(model(X), Y)
#         
#         curr_loss.backward()
#         
#         grad2 = get_vectorized_grads(model, args.device)
#         
#              
#              
# #         expected_grad_diff = grad2_list[-1] - grad_list[-1] + args.wd*vec_para
#         expected_grad_diff = grad1 - grad2 + args.wd*vec_para
#          
#          
#         print(torch.dot(hessian_para_prod_0.view(-1),expected_grad_diff.view(-1))/(torch.norm(expected_grad_diff.view(-1))*torch.norm(hessian_para_prod_0.view(-1))))
#         print(torch.norm(hessian_para_prod_0.view(-1) - expected_grad_diff.view(-1)))
        
        
        
        full_existing_labeled_id_tensor = torch.load(full_out_dir + '/' + args.model + 'full_existing_labeled_id_tensor')
        
        remaining_sample_ids = torch.tensor(list(set(range(full_training_noisy_dataset.lenth)).difference(set(full_existing_labeled_id_tensor.tolist()))))
        
        all_grad_list = []
        
        all_para_list = []
        
        torch.save(remaining_sample_ids, full_out_dir + '/remaining_sample_ids')
        
        for k in range(iteration_count):
            
            print('start computing gradients::', k)
            
            full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + str(k))
            
            w_list = torch.load(full_out_dir + '/w_list_' + str(k))
            
            grad_list = torch.load(full_out_dir + '/grad_list_' + str(k))
            
            model = torch.load(full_out_dir + '/model_' + str(k))
            
            optimizer = model.get_optimizer(args.tlr, args.wd)
#             set_model_parameters(model, get_model_para_list(curr_model), args.device)
        
            curr_grads = get_vectorized_grads_sample_wise(model, w_list, grad_list, m, optimizer, full_training_noisy_dataset.data[remaining_sample_ids], full_training_noisy_dataset.labels[remaining_sample_ids], args.device)
        
#             all_grad_list.append(curr_grads)
            
#             all_para_list.append(get_model_para_list(model))
            
            torch.save(curr_grads, full_out_dir + '/all_sample_wise_grad_list_' + str(k))
        
            del curr_grads
        
#             torch.save(all_para_list, full_out_dir + '/all_sample_wise_para_list')
        
        
        
        
        

