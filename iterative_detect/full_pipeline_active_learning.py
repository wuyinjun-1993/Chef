'''
Created on Jan 13, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob
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
    from model.utils_real import *
    from models.util_func import *
    from Reweight_examples.utils_reweight import *
except ImportError:
    from utils import *
    from utils_real import *
    from util_func import *
    from utils_reweight import *

import models
import math

import active_learning_query

label_change_threshold = 1e-5

incremental_training_threshold = 0.05

last_iter = 3

small_constant = 1e-4

method = 'least_confidence'





def pre_compute_influence_order_history(w_list, grad_list, model, training_dataset, loss_func, optimizer, random_ids_all_epochs, batch_size, regularization_rate, num_class):
    
    
    set_model_parameters(model, w_list[-1])
    
    model_output = F.softmax(model(training_dataset.data), dim = 1)
            
            
    '''n*k'''
    loss_gap_coefficient = model_output*(1-model_output)*(torch.norm(training_dataset.data.view(training_dataset.data.shape[0], -1), dim = 1).view(-1,1)*np.sqrt(num_class))
    
    mu, c2, grad_bound_list, mu_bound_list = compute_lower_bound_hessian_w(grad_list, w_list, model, optimizer, loss_func, random_ids_all_epochs, training_dataset, batch_size, regularization_rate)
    
    w_diff_remove_1_2, w_diff_remove_1_3 = compute_w_bound(mu_bound_list, mu, args.tlr, grad_bound_list, training_dataset.data.shape[0])
    
    w_diff_remove_1 = 2/mu*c2/training_dataset.data.shape[0]
    
#     curr_w_diff_remove_1 = 2/mu_2*c2_2/training_dataset.data.shape[0]
    
#     loss_gap = 4*(c2*prev_w_diff_remove_1 + delta_loss1)
    return w_diff_remove_1, loss_gap_coefficient, w_diff_remove_1_2
    
    
    

def determine_threshold(curr_w_list, curr_grad_list, prev_w_list, prev_grad_list, model, training_dataset, updated_training_dataset, validation_dataset, loss_func, optimizer, random_ids_all_epochs, batch_size, regularization_rate):
    curr_last_w = get_all_vectorized_parameters1(curr_w_list[-1])
    prev_last_w = get_all_vectorized_parameters1(prev_w_list[-1])
    last_w_diff = torch.norm(curr_last_w - prev_last_w)
    set_model_parameters(model, curr_w_list[-1])
    
    set_model_parameters(model, prev_w_list[-1])
    
    hard_loss_func = model.get_loss_function()
    
    
    loss1 = hard_loss_func(model(validation_dataset.data), validation_dataset.labels)
    
    set_model_parameters(model, curr_w_list[-1])
    
    loss2 = hard_loss_func(model(validation_dataset.data), validation_dataset.labels)
    
    
    delta_loss1 = torch.abs(loss1 - loss2)
    
    mu, c2 = compute_lower_bound_hessian_w(prev_grad_list, prev_w_list, model, optimizer, loss_func, random_ids_all_epochs, training_dataset, batch_size, regularization_rate)
    
#     mu_2, c2_2 = compute_lower_bound_hessian_w(curr_grad_list, curr_w_list, model, optimizer, loss_func, random_ids_all_epochs, updated_training_dataset, batch_size, regularization_rate)
    
    prev_w_diff_remove_1 = 2/mu*c2/training_dataset.data.shape[0]
    
#     curr_w_diff_remove_1 = 2/mu_2*c2_2/training_dataset.data.shape[0]
    
    loss_gap = 4*(c2*prev_w_diff_remove_1 + delta_loss1)
    
    return loss_gap
    
    
    

def check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
    return torch.sum(ids_with_changed_ids)*1.0/full_training_noisy_dataset.data.shape[0] < incremental_training_threshold


def compute_loss_valid_dataset_grad(validation_dataset, model, loss_func, optimizer, w_diff_remove_1, model_param):
    set_model_parameters(model, model_param)
    
    optimizer.zero_grad()
    
    curr_loss = loss_func(model(validation_dataset.data),validation_dataset.labels)
    
    curr_loss.backward()
    
    vec_grad = get_vectorized_grads(model)
    
    loss_gap = torch.norm(vec_grad.view(-1))*w_diff_remove_1
    
    return loss_gap
    
    

def select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, is_incremental, iter_count, full_existing_labeled_id_tensor, curr_w_list, curr_grad_list, derive_probab_labels=False, train_annotated_label_tensor = None, suffix = 'al', r_weight = None):
    
    uncert_sampling = active_learning_query.UncertaintySampling(True)
    
    active_leaning_method_func = getattr(uncert_sampling, method)
    
#     active_leaning_method_func0 = getattr(uncert_sampling, method + '0')
    
    
#     if (not is_incremental) or (not check_iteration_count_file_existence(full_out_dir) or not compare_iteration_count(iter_count, full_out_dir)):
    if True:#(not is_incremental) or (iter_count <= last_iter): 
    
#         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
        
        
        final_remaining_ids = None
        
        if full_existing_labeled_id_tensor is not None:
            remaining_ids = torch.tensor(list(set(range(full_training_noisy_dataset.data.shape[0])).difference(set(full_existing_labeled_id_tensor.tolist()))))
        else:
            remaining_ids = torch.tensor(list(range(full_training_noisy_dataset.data.shape[0])))
            
        remaining_dataset_full = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
        
        t1 = time.time()
        
        prob_dist_list_tensor, influences1, ordered_list, sorted_train_ids = uncert_sampling.get_samples_batch(model, full_training_noisy_dataset.data[remaining_ids], active_leaning_method_func, number=args.removed_count, batch_size = args.bz, is_GPU = args.GPU, device = args.device)
        
#         prob_dist_list_tensor_exp, influences1_exp, ordered_list_exp, sorted_train_ids_exp = uncert_sampling.get_samples(model, full_training_noisy_dataset.data[remaining_ids], active_leaning_method_func0, number=args.removed_count)
        
        final_prod_dist_list_tensor = torch.zeros([full_training_noisy_dataset.data.shape[0], prob_dist_list_tensor.shape[1]], dtype = prob_dist_list_tensor.dtype)
        
        final_prod_dist_list_tensor[remaining_ids] = prob_dist_list_tensor.cpu()
        
        full_influences1 = torch.zeros(full_training_noisy_dataset.data.shape[0], dtype = prob_dist_list_tensor.dtype)
        
        full_influences1[remaining_ids] = influences1.cpu()
        
        origin_sorted_train_ids = remaining_ids[sorted_train_ids]
        
        t2 = time.time()
        
        print('time full::', t2 - t1)
        
        # if iter_count >= 1:
            # prev_sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_' + method + '_removed_ids_v' + str(iter_count-1))
            #
            # prev_ordered_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_v' + str(iter_count-1))
            #
            # prev_influences1 = torch.load(full_out_dir + '/' + args.model + '_' + method + '_v' + str(iter_count-1))
            #
            # prev_prob_dist_list_tensor = torch.load(full_out_dir + '/' + args.model + '_' + method + 'probs_v' + str(iter_count-1))
            
#         influences1, full_grad_tensors, ordered_list, sorted_train_ids, s_test_vec = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset)
    
    
    
        if not is_incremental:
        
            torch.save(origin_sorted_train_ids, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count))
            
            torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count))
            
            torch.save(full_influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count))
            
            torch.save(final_prod_dist_list_tensor, full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count))
        else:
            torch.save(origin_sorted_train_ids, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            
            torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            
            torch.save(full_influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            
            torch.save(final_prod_dist_list_tensor, full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            
        print('calculate influence time 1::', t2 - t1)
    # else:
    #
        # active_leaning_method_bound_func = getattr(uncert_sampling, method + '_bound')
        #
# #         set_model_parameters(model, curr_w_list[-1])
        #
        # w_diff_remove_1 = torch.load(full_out_dir + '/w_diff_remove_1_' + str(iter_count - 2))
        #
        # prob_dist_list_tensor_1 = torch.load(full_out_dir + '/' + args.model + '_' + method + 'probs_v' + str(iter_count - 2) + incremental_suffix)
        #
        #
        # loss_gap_coefficient = torch.load(full_out_dir + '/loss_gap_coefficient_' + str(iter_count - 2))
        # '''n*k'''
        #
        # prev_influences1_last_iter = torch.load(full_out_dir + '/' + args.model + '_'+ method +'_v' + str(iter_count - 2) + incremental_suffix)
        #
        # full_removed_id_tensor_0 = torch.load(full_out_dir + '/' + args.model + '_' + method + '_removed_ids_v'  + str(iter_count - 2) + incremental_suffix, map_location='cpu')
        #
        # full_existing_labeled_id_tensor_0 = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_v' + str(iter_count - 3) + incremental_suffix)
        #
# #             prev_remaining_ids = torch.tensor(list(set(range(full_training_noisy_dataset.data.shape[0])).difference(set(full_removed_id_tensor_0.tolist()))))
        #
        #
        #
        # loss_gap = loss_gap_coefficient*w_diff_remove_1*args.removed_count*3
        #
        # influence_lower_bound, influence_upper_bound, prev_influences1 = active_leaning_method_bound_func(prob_dist_list_tensor_1, loss_gap)
        #
        # t1 = time.time()
        #
        # influence_lower_bound[full_existing_labeled_id_tensor_0] = 0
        #
        # influence_upper_bound[full_existing_labeled_id_tensor_0] = 0
        #
        # prev_influences1[full_existing_labeled_id_tensor_0] = 0
        #
# #             upper_prob_bound = (prob_dist_list_tensor_1 + loss_gap)[:,0]
# #             
# #             lower_prob_bound = (prob_dist_list_tensor_1 - loss_gap)[:,0]
        #
# #             model_output = model(full_training_noisy_dataset.data)
# #             
# #             
# #             '''n*k'''
# #             loss_gap = model_output*(1-model_output)*(torch.norm(full_training_noisy_dataset.data.view(full_training_noisy_dataset.data.shape[0], -1), dim = 1)**args.num_class)*w_diff_remove_1
        #
        #
        #
# #             c2 = torch.load(full_out_dir + '/c2_'  + str(last_iter))
# #             
# #             mu = torch.load(full_out_dir + '/mu_' + str(last_iter))
# #             
# #             w_list_iter_0 = torch.load(full_out_dir + '/w_list_v' + str(last_iter))
# #             
# #             full_existing_labeled_id_tensor_0 = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_v' + str(last_iter))
# #             
# #             loss_func = model.get_loss_function()
# #             
# #             loss_gap1 = compute_loss_valid_dataset_grad(validation_dataset, model, loss_func, optimizer, w_diff_remove_1, w_list[-1])
# #             
# #             loss_gap2 = compute_loss_valid_dataset_grad(validation_dataset, model, loss_func, optimizer, w_diff_remove_1, w_list_iter_0[-1])
# #             
# #             loss_gap = torch.abs(loss_gap1) + torch.abs(loss_gap2)
        #
        #
        #
# #             delta_existing_labeled_id_tensor_0 = full_existing_labeled_id_tensor[~(torch.sum(full_existing_labeled_id_tensor.view(1,-1) == full_existing_labeled_id_tensor_0.view(-1,1), dim = 0).bool())]
    #
        # remaining_removed_id0 = full_removed_id_tensor_0[~(torch.sum(full_removed_id_tensor_0.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
        #
# #             remaining_removed_id1 = full_removed_id_tensor1[~(torch.sum(full_removed_id_tensor1.view(1,-1) == full_existing_labeled_id_tensor.view(-1,1), dim = 0)).bool()]
        #
# #             id_set1 = remaining_removed_id0[args.removed_count:][(influences1[remaining_removed_id0[args.removed_count:]].view(-1) + loss_gap[remaining_removed_id0[args.removed_count:]]) > (influences1[remaining_removed_id0[args.removed_count-1]].view(-1) - loss_gap[remaining_removed_id0[args.removed_count:]])]
        #
        # id_set1 = remaining_removed_id0[args.removed_count:][(influence_upper_bound[remaining_removed_id0[args.removed_count:]]) > (influence_lower_bound[remaining_removed_id0[args.removed_count-1]])]
        #
        # final_remaining_ids= torch.cat([remaining_removed_id0[0:args.removed_count].view(-1), id_set1], dim = 0)
        #
# #         t1 = time.time()
        # '''model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size'''
        # loss_func = model.soft_loss_function_reduce
        #
# #         if final_remaining_ids is not None:
        #
# #             remaining_dataset = models.MyDataset(full_training_noisy_dataset.data[final_remaining_ids], full_training_noisy_dataset.labels[final_remaining_ids])
# #             influences1_2, full_grad_tensors_2, ordered_list_2, sorted_train_ids_2, s_test_vec_2 = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, remaining_dataset, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset)
        #
        # t3 = time.time()
        # prob_dist_list_tensor_2, influences1_2, ordered_list_2, sorted_train_ids_2 = uncert_sampling.get_samples_batch(model, full_training_noisy_dataset.data[final_remaining_ids], active_leaning_method_func, number=args.removed_count, batch_size = args.bz, is_GPU = args.GPU, device = args.device)
        # t4 = time.time()
        #
        #
        # origin_sorted_train_ids_2 = final_remaining_ids[sorted_train_ids_2]
        #
        # final_prod_dist_list_tensor_2 = torch.zeros([full_training_noisy_dataset.data.shape[0], prob_dist_list_tensor_2.shape[1]], dtype = prob_dist_list_tensor_2.dtype)
        #
        # final_prod_dist_list_tensor_2[final_remaining_ids] = prob_dist_list_tensor_2
        #
        # full_influences1_2 = torch.zeros(full_training_noisy_dataset.data.shape[0], dtype = prob_dist_list_tensor_2.dtype)
        #
        # full_influences1_2[final_remaining_ids] = influences1_2
        #
        # t2 = time.time()
        #
        # print('time small::', t2 - t1)
        #
        # print('time small compute::', t4 - t3)
        #
# #         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
#
# #         prov_iter_count = load_iteration_count(full_out_dir)
# #         
# #         m = args.hist_size
# # #         prev_sample_wise_grad_list, curr_sample_wise_grad_list = load_sample_wise_grad_list(full_out_dir, prov_iter_count)
# # 
# #         all_ids_with_unchanged_labels_bool, all_ids_with_unchanged_labels = obtain_remaining_ids(full_out_dir, prov_iter_count, iter_count, is_incremental)
# #         
# #         all_ids_with_changed_labels_bool = ~all_ids_with_unchanged_labels_bool
# # 
# #         s_k_list, y_k_list, mini_Y_k_list, mini_Y_k_list2, full_grad_list, prev_w_list, prev_grad_list = obtain_s_y_list(args, all_ids_with_unchanged_labels, full_out_dir, is_incremental, m = m, k = prov_iter_count)
# # 
# #         loss_func = model.soft_loss_function_reduce
# #         
# #         y_k_list_tensor = torch.stack(y_k_list, dim = 0)
# # 
# #         combined_mat = torch.zeros([full_training_noisy_dataset.data.shape[0], y_k_list_tensor.shape[-1], 2*m], dtype = full_training_noisy_dataset.data.dtype)
# # 
# #         zero_mat_dim, full_S_k, full_Y_k, full_sigma_k, full_inv_mat, full_combined_mat = prepare_approx_hessian_vec_prod(s_k_list, y_k_list_tensor, m, combined_mat, False, args.device)
# #         
# # #         curr_w_list = torch.load(full_out_dir + '/w_list_' + str(iter_count))
# # #     
# # #         curr_grad_list = torch.load(full_out_dir + '/grad_list_' + str(iter_count))
# #         
# #         
# # #         set_model_parameters(model, curr_w_list[-1], args.device)
# #         
# #         t1 = time.time()
# #         
# #         influences2, full_grad_tensors, ordered_list, sorted_train_ids = incremental_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, s_k_list, y_k_list_tensor, full_grad_list, curr_w_list, prev_w_list, all_ids_with_unchanged_labels, args.GPU, args.device, args.wd,m, args.bz, zero_mat_dim, full_S_k, full_Y_k, full_sigma_k, full_inv_mat, full_combined_mat)
# #         
# #         t2 = time.time()
# #         
# #         print('calculate influence time 1::', t2 - t1)
        #
# #         set_model_parameters(model, curr_w_list[-1], args.device)
        #
# #         influences1, exp_full_grad_tensors, exp_ordered_list, exp_sorted_train_ids = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz)
        # origin_sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count))
        #
        # ordered_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count))
        #
        # full_influences1 = torch.load(full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) )
        #
        # final_prod_dist_list_tensor = torch.load(full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count))
        #
        # print('selected id difference::', origin_sorted_train_ids[0:args.removed_count] - origin_sorted_train_ids_2[0:args.removed_count])
        #
        # torch.save(origin_sorted_train_ids_2, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(ordered_list_2, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(full_influences1_2, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(final_prod_dist_list_tensor_2, full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count) + incremental_suffix)
# #         torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count) + incremental_suffix)
# #     
# #         torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count) + incremental_suffix)
# #         
# #         torch.save(influences2, full_out_dir + '/' + args.model + '_influence_v' + str(iter_count) + incremental_suffix)
        #
        # print('here')
        
        
    updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False, iter_count = iter_count, is_incremental = is_incremental, derive_probab_labels = derive_probab_labels, method = method, train_annotated_label_tensor = train_annotated_label_tensor, suffix = suffix)
    
    ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)
    
    print('samples with updated labels::', torch.nonzero(ids_with_changed_ids), torch.sum(ids_with_changed_ids))
    
    
    
    return updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids
    

# def select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, is_incremental, iter_count, full_existing_labeled_id_tensor, curr_w_list, curr_grad_list, derive_probab_labels=False, method = 'influence', train_annotated_label_tensor = None, derived_lr = 0.0002, regular_rate = 0.1, r_weight = None, suffix = 'sl'):
# #     if (not is_incremental) or (not check_iteration_count_file_existence(full_out_dir) or not compare_iteration_count(iter_count, full_out_dir)):
#
    # optimizer = model.get_optimizer(args.tlr, args.wd)
    #
    # if_dnn = False
    #
    # if iter_count <= last_iter + 1 or (not is_incremental): 
# #         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
        # final_remaining_ids = None
        #
        #
        # if full_existing_labeled_id_tensor is not None:
            # remaining_ids = torch.tensor(list(set(range(full_training_noisy_dataset.data.shape[0])).difference(set(full_existing_labeled_id_tensor.tolist()))))
        # else:
            # remaining_ids = torch.tensor(list(range(full_training_noisy_dataset.data.shape[0])))
            #
        # loss_func = model.soft_loss_function_reduce
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
        # t1 = time.time()
        #
        # Y_difference = torch.zeros([args.num_class, full_training_noisy_dataset.data.shape[0], args.num_class], dtype = torch.double)
        #
        # for p in range(args.num_class):
            # Y_class = onehot(torch.tensor([p]), args.num_class)
    # #         print(p, Y_class)
            # Y_difference[p] = Y_class.view(1,-1) - full_training_noisy_dataset.labels
            #
        # remaining_dataset_full = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
        #
        # # influences1, full_grad_tensors, ordered_list, origin_sorted_train_ids0, s_test_vec = origin_compute_sample_class_wise_gradients(args.num_class, model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, Y_difference = Y_difference[:,remaining_ids], derived_lr = derived_lr, regular_rate = regular_rate, r_weight = r_weight)
        #
        # if if_dnn:
            # influences1, full_grad_tensors, ordered_list, origin_sorted_train_ids0, s_test_vec = origin_compute_sample_class_wise_gradients2(args.num_class, model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, Y_difference = Y_difference[:,remaining_ids], derived_lr = derived_lr, regular_rate = regular_rate, r_weight = r_weight)
            #
        # else:
            # influences1, full_grad_tensors, ordered_list, origin_sorted_train_ids0, s_test_vec = origin_compute_sample_class_wise_gradients(args.num_class, model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, Y_difference = Y_difference[:,remaining_ids], derived_lr = derived_lr, regular_rate = regular_rate, r_weight = r_weight)
            #
        # sorted_train_ids = origin_sorted_train_ids0%remaining_ids.shape[0]
        #
        # suggested_updated_labels = torch.floor(origin_sorted_train_ids0/remaining_ids.shape[0])
        #
        # print('test::', torch.nonzero(full_training_origin_labels == 0).shape, torch.nonzero(full_training_origin_labels == 1).shape, suggested_updated_labels[0:500])
        #
        # final_influence_values = torch.zeros([args.num_class, full_training_noisy_dataset.data.shape[0]], dtype = torch.double)
        #
        # origin_sorted_train_ids = remaining_ids[sorted_train_ids] 
        #
        # if not if_dnn:
            # all_full_grad_tensors = torch.zeros([full_training_noisy_dataset.data.shape[0], full_grad_tensors.shape[1], full_grad_tensors.shape[2]], dtype = full_grad_tensors.dtype)
            # all_full_grad_tensors[remaining_ids] = full_grad_tensors
        # else:
            # all_full_grad_tensors = full_training_noisy_dataset.data.clone()
            #
        # final_influence_values[:,remaining_ids] = influences1
        #
        #
        #
        # print('influence shape::', final_influence_values.shape)
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
            #
        # t2 = time.time()
        #
        # print(calculate_influence_time_prefix, t2 - t1)
        #
        # print(remaining_samples_for_computing_influence_prefix, remaining_ids.shape[0])
        #
        # if not is_incremental:
        #
            # torch.save(origin_sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # torch.save(remaining_ids, full_out_dir + '/' + args.model + '_influence_remaining_ids_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # torch.save(final_influence_values, full_out_dir + '/' + args.model + '_influence_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # torch.save(s_test_vec, full_out_dir + '/' + args.model + '_influence_s_test_vec_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # torch.save(suggested_updated_labels, full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_' + suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
            #
            # if iter_count == last_iter+1:
                # torch.save(all_full_grad_tensors, full_out_dir + '/' + args.model + '_influence_sample_wise_grad_' + suffix + str(iter_count)  + args.resolve_conflict_str)
                #
        # else:
# #             origin_sorted_train_ids0 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count))
            #
            # torch.save(origin_sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix  + args.resolve_conflict_str)
            #
            # torch.save(remaining_ids, full_out_dir + '/' + args.model + '_influence_remaining_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
            #
            # torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix  + args.resolve_conflict_str)
            #
            # torch.save(final_influence_values, full_out_dir + '/' + args.model + '_influence_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
            #
            # torch.save(s_test_vec, full_out_dir + '/' + args.model + '_influence_s_test_vec_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
            #
            # torch.save(suggested_updated_labels, full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
            #
            # if iter_count == last_iter+1:
                # torch.save(all_full_grad_tensors, full_out_dir + '/' + args.model + '_influence_sample_wise_grad_' + suffix + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
    # else:
    #
# #         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
#
#
# #         if iter_count > last_iter + 1:
            #
# #         full_existing_labeled_id_tensor_0, full_training_noisy_dataset_0, model_0, random_ids_multi_super_iterations_0, w_list_0, grad_list_0 = load_retraining_history_info(iter_count-1, full_out_dir, args)
            #
# #         w_diff_remove_1 = torch.load(full_out_dir + '/w_diff_remove_1_sl_' + str(last_iter+1)+ args.resolve_conflict_str)
        #
# #         prev_full_grad_tensors0 = torch.load(full_out_dir + '/' + args.model + '_influence_sample_wise_grad_sl' + str(last_iter+1) + incremental_suffix+ args.resolve_conflict_str)
#
        # all_epochs = torch.load(full_out_dir + '/all_epochs_' + suffix + '_' + str(last_iter + 1)+ args.resolve_conflict_str)
        #
        # last_stopped_epoch = torch.load(full_out_dir + '/stopped_epoch_' + suffix + '_v' + str(iter_count - 1)+ incremental_suffix+ args.resolve_conflict_str)
        #
        # print('all_epochs::', all_epochs)
        #
        # selected_epoch_id = torch.argmin(torch.abs(last_stopped_epoch - all_epochs)).view(-1)
        #
        # selected_epoch = all_epochs[selected_epoch_id] 
        #
        #
        #
        # max_mu_list_tensor = torch.load(full_out_dir + '/max_mu_list_tensor_' + suffix + '_' + str(last_iter + 1) + '_epoch_' + str(selected_epoch.item())+ args.resolve_conflict_str)
        #
        # c2 = torch.load(full_out_dir + '/c2_' + suffix + '_'  + str(last_iter + 1)+ args.resolve_conflict_str)
        #
        # mu = torch.load(full_out_dir + '/mu_' + suffix + '_' + str(last_iter + 1)+ args.resolve_conflict_str)
        #
# #         w_list_iter_0 = torch.load(full_out_dir + '/w_list_v' + str(iter_count - 1) + incremental_suffix)
        #
        # origin_model_param_per_epoch_list = torch.load(full_out_dir + '/model_param_list_' + suffix + '_' + str(last_iter + 1)+ args.resolve_conflict_str)
        #
        # origin_model = origin_model_param_per_epoch_list[selected_epoch]#torch.load(full_out_dir + '/model_sl_v' + str(last_iter) + incremental_suffix)
        #
# #         origin_model0 = torch.load(full_out_dir + '/model_sl_v' + str(last_iter) + incremental_suffix)
# #         
# #         origin_stopped_epoch = torch.load(full_out_dir + '/stopped_epoch_sl_v' + str(last_iter) + incremental_suffix)
# #         
# #         last_model = torch.load(full_out_dir + '/model_sl_v' + str(iter_count-1))
        #
# #         print('origin model difference::', torch.norm(get_all_vectorized_parameters1(list(origin_model0.parameters())) - get_all_vectorized_parameters1(origin_model_param_per_epoch_list[origin_stopped_epoch])))
# # 
# #         print('origin model difference::', torch.norm(get_all_vectorized_parameters1(list(last_model.parameters())) - get_all_vectorized_parameters1(list(model.parameters()))))
# # 
        # print('selected_epoch::',  selected_epoch, last_stopped_epoch)
        #
        # prev_full_grad_tensors = torch.load(full_out_dir + '/pre_compute_sample_wise_grad_epoch_' + str(selected_epoch.item()) + args.resolve_conflict_str)
        #
        # origin_model_param = origin_model#get_model_para_list(origin_model)
        #
# #         full_existing_labeled_id_tensor_0 = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_v' + str(last_iter) + incremental_suffix)
        #
        #
        # origin_influences1 = torch.load(full_out_dir + '/' + args.model + '_influence_' + suffix + '_v' + str(last_iter+1) + incremental_suffix+ args.resolve_conflict_str)
        #
        # full_removed_id_tensor_0 = torch.load(full_out_dir + '/' + args.model + '_' + method + '_remaining_ids_' + suffix + '_v'  + str(last_iter+1) + incremental_suffix+ args.resolve_conflict_str, map_location='cpu')
        #
        #
        # model_param = get_model_para_list(model)
        #
        # delta_model_param = get_all_vectorized_parameters1(model_param) - get_all_vectorized_parameters1(origin_model_param)
        #
        # print('model parameter difference::', torch.norm(delta_model_param))
        #
        # prev_s_test_vec = torch.load(full_out_dir + '/' + args.model + '_influence_s_test_vec_' + suffix + '_v' + str(last_iter + 1) + incremental_suffix+ args.resolve_conflict_str)
        #
        #
# #         loss_func = model.get_loss_function()
        # loss_func = model.soft_loss_function_reduce
        #
# #         loss_gap2 = compute_loss_valid_dataset_grad(validation_dataset, model, loss_func, optimizer, w_diff_remove_1, get_model_para_list(origin_model))
        #
        # prev_full_grad_tensors_multi_class = prev_full_grad_tensors.view(1, prev_full_grad_tensors.shape[0], prev_full_grad_tensors.shape[1], prev_full_grad_tensors.shape[2]).repeat(args.num_class, 1, 1, 1)
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
        # t1 = time.time()
        #
        # Y_difference = torch.zeros([args.num_class, full_training_noisy_dataset.data.shape[0], args.num_class], dtype = torch.double)
        #
        # for p in range(args.num_class):
            # Y_class = onehot(torch.tensor([p]), args.num_class)
    # #         print(p, Y_class)
            # Y_difference[p] = Y_class.view(1,-1) - full_training_noisy_dataset.labels
            #
        # final_remaining_ids, s_test_vec_tensor = obtain_remaining_ids_incremental_class_wise2(args.num_class, model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, args.GPU, args.device, args.wd, args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, delta_model_param, prev_full_grad_tensors, prev_full_grad_tensors_multi_class, max_mu_list_tensor, full_removed_id_tensor_0, full_existing_labeled_id_tensor, args, origin_influences1, prev_s_test_vec, Y_difference)
        #
        # '''model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size'''
        #
        # remaining_dataset = models.MyDataset(full_training_noisy_dataset.data[final_remaining_ids], full_training_noisy_dataset.labels[final_remaining_ids])
        # influences1_2, full_grad_tensors_2, ordered_list_2, origin_sorted_train_ids_2_0, s_test_vec_2 = origin_compute_sample_class_wise_gradients(args.num_class, model, optimizer, loss_func, full_out_dir, remaining_dataset, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, s_test_vec_tensor = s_test_vec_tensor, Y_difference = Y_difference[:,final_remaining_ids], derived_lr = derived_lr, regular_rate = regular_rate)
        #
        # sorted_train_ids_2 = origin_sorted_train_ids_2_0%final_remaining_ids.shape[0]
        #
        # suggested_updated_labels = torch.floor(origin_sorted_train_ids_2_0/final_remaining_ids.shape[0])
        #
        #
        #
        # origin_sorted_train_ids_2 = final_remaining_ids[sorted_train_ids_2]
        #
        # final_influence_values_2 = torch.zeros([args.num_class, full_training_noisy_dataset.data.shape[0]], dtype = torch.double)
        #
        # final_influence_values_2[:, final_remaining_ids] = influences1_2
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
        # t2 = time.time()
        #
        # print(calculate_influence_time_prefix, t2 - t1)
        #
        #
        # if full_existing_labeled_id_tensor is not None:
            # remaining_ids = torch.tensor(list(set(range(full_training_noisy_dataset.data.shape[0])).difference(set(full_existing_labeled_id_tensor.tolist()))))
        # else:
            # remaining_ids = torch.tensor(list(range(full_training_noisy_dataset.data.shape[0])))
            #
            #
        # remaining_dataset_full = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
        # t5 = time.time()
        #
# #         influences1, full_grad_tensors, ordered_list, sorted_train_ids, s_test_vec = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset)
        #
        # influences1, full_grad_tensors, ordered_list, origin_sorted_train_ids0, s_test_vec = origin_compute_sample_class_wise_gradients(args.num_class, model, optimizer, loss_func, full_out_dir, remaining_dataset_full, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz, full_training_noisy_dataset.data.shape[0], full_training_noisy_dataset, Y_difference = Y_difference[:,remaining_ids], derived_lr=derived_lr, regular_rate = regular_rate)
        #
        #
        # sorted_train_ids = origin_sorted_train_ids0%remaining_ids.shape[0]
        #
        # suggested_updated_labels0 = origin_sorted_train_ids0/remaining_ids.shape[0]
        #
        # final_influence_values = torch.zeros([args.num_class, full_training_noisy_dataset.data.shape[0]], dtype = torch.double)
        #
        # origin_sorted_train_ids = remaining_ids[sorted_train_ids] 
        #
        # final_influence_values[:, remaining_ids] = influences1
        #
        # if args.GPU:
            # torch.cuda.synchronize(device = args.device)
        # t6 = time.time()
        #
        # print('sorted train ids difference::', origin_sorted_train_ids[0:args.removed_count] - origin_sorted_train_ids_2[0:args.removed_count])
        #
# #         print('w difference::', w_diff_remove_1)
        #
# #         print('loss gap::', loss_gap)
# #         
        # print(remaining_samples_for_computing_influence_prefix, final_remaining_ids.shape[0])
        #
        # print('full time::', t6 - t5)
        #
        #
        # torch.save(origin_sorted_train_ids_2, full_out_dir + '/' + args.model + '_influence_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # torch.save(ordered_list_2, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # torch.save(final_influence_values_2, full_out_dir + '/' + args.model + '_influence_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # torch.save(s_test_vec_2, full_out_dir + '/' + args.model + '_influence_s_test_vec_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # torch.save(suggested_updated_labels, full_out_dir + '/' + args.model + '_influence_suggested_updated_labels_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # torch.save(remaining_ids, full_out_dir + '/' + args.model + '_influence_remaining_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
        #
        # print('here')
        #
        #
    # updated_labels, full_existing_labeled_id_tensor, final_labeled_id_tensor = obtain_updated_labels_class_wise(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False, iter_count = iter_count, is_incremental = is_incremental, derive_probab_labels = derive_probab_labels,method_flag = '_' + suffix,resolve_conflict=args.resolve_conflict, train_annotated_label_tensor = train_annotated_label_tensor)
    #
    # ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)
    #
    # print('samples with updated labels::', torch.nonzero(ids_with_changed_ids), torch.sum(ids_with_changed_ids))
    #
    #
    #
    # return updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids

'''model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, suffix'''

'''full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count + 1, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, suffix, r_weight = r_weight'''
def model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, start = False, regularization_prob_samples = 0.1, r_weight = None):
    
    
    # if args.incremental and iter_count - 1 == last_iter:
    #     print('collect influence function history info')
    #
    #     # model = model.to('cpu')
    #
    #     model_param = get_model_para_list(model)
    #
    #     model_param_per_epoch_list = get_model_params_per_epochs(random_ids_multi_super_iterations, w_list, model_param, full_training_noisy_dataset, args.bz)
    #
    #     stopped_epoch = torch.load(full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count-1))
    #
    #     updated_dataset = models.MyDataset(full_training_noisy_dataset.data, updated_labels)
    #     w_diff_remove_1, c2, mu, all_epochs = pre_compute_influence_order_history(args,w_list, grad_list, model, full_training_noisy_dataset, r_weight, model.soft_loss_function, optimizer, random_ids_multi_super_iterations, args.bz, args.wd, model_param, args.GPU, args.device, args.num_class, model_param_per_epoch_list, stopped_epoch, args.hist_period, full_out_dir + '/max_mu_list_tensor_' + args.suffix + '_' + str(iter_count),full_out_dir + '/max_sample_mu_list_tensor_' + args.suffix + '_' + str(iter_count), full_out_dir)
    #     torch.save(w_diff_remove_1, full_out_dir + '/w_diff_remove_1_' + args.suffix + '_' + str(iter_count)+ args.resolve_conflict_str)
    #
    #     torch.save(c2, full_out_dir + '/c2_' + args.suffix + '_' + str(iter_count)+ args.resolve_conflict_str)
    #
    #     torch.save(mu, full_out_dir + '/mu_' + args.suffix + '_' + str(iter_count)+ args.resolve_conflict_str)
    #
    #     torch.save(torch.tensor(all_epochs), full_out_dir + '/all_epochs_' + args.suffix + '_' + str(iter_count)+ args.resolve_conflict_str)
    #
    #     torch.save(model_param_per_epoch_list, full_out_dir + '/model_param_list_' + args.suffix + '_' + str(iter_count)+ args.resolve_conflict_str)
    #     set_model_parameters(model, model_param, args.device)
    
    r_weight_old = r_weight.clone()
    
    r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)*regularization_prob_samples
    
    r_weight[full_existing_labeled_id_tensor.type(torch.long)] = 1
    
    model = model.to(args.device)
    
    print('regular rate::', regularization_prob_samples)
    
    exp_updated_origin_grad_list = None 
    
#         exp_updated_origin_grad_list = update_gradient_origin(all_entry_grad_list, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.GPU, args.device, w_list, model, optimizer)
    
    criterion = model.soft_loss_function_reduce
    
    
    origin_model_param = get_vectorized_params(model)
    
    if start:
        set_model_parameters(model, w_list[-1], args.device)
    else:
        set_model_parameters(model, w_list[0], args.device)
    
    print('sample count with changed labels::', torch.sum(ids_with_changed_ids))
    
    exp_updated_w_list, exp_updated_grad_list = None, None
    
    
    
#     if not is_incremental:
    # if iter_count <= last_iter + 1 or (not is_incremental): 
    if True:
        if args.GPU:
            torch.cuda.synchronize(device = args.device)
    
        t1 = time.time()
        '''model, optimizer, random_ids_multi_super_iterations, train_dataset, batch_size, epochs, is_GPU, device, loss_func = None, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping=True, test_performance = True'''
        
#         updated_training_dataset = models.MyDataset(full_training_noisy_dataset.data, updated_labels)
        
        random_ids_multi_super_iterations = random_ids_multi_super_iterations[0:args.epochs]
        
        if regularization_prob_samples == 0:
            training_data_feature = full_training_noisy_dataset.data[full_existing_labeled_id_tensor]
    
            training_label_tensor = updated_labels[full_existing_labeled_id_tensor]
            
            # training_data_feature = torch.cat([training_data_feature, validation_dataset.data], dim = 0)
            #
            # training_label_tensor = torch.cat([training_label_tensor, onehot(validation_dataset.labels, args.num_class)], dim = 0).detach()
            
            training_dataset = MyDataset(training_data_feature, training_label_tensor)

            if args.GPU_measure:
                updated_w_list, updated_grad_list,random_ids_multi_super_iterations, GPU_mem_usage_list = train_model_dataset(args, model, optimizer, None, training_dataset.data, training_label_tensor, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False, measure_GPU_utils = True)
            else:
                updated_w_list, updated_grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, training_dataset.data, training_label_tensor, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False)
        
        else:
            
            if args.GPU_measure:
                updated_w_list, updated_grad_list,_, GPU_mem_usage_list = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False, measure_GPU_utils = True, r_weight = r_weight)
            else:
                updated_w_list, updated_grad_list,_ = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False, r_weight = r_weight)
    
#         final_exp_updated_model_param = get_all_vectorized_parameters1(model.parameters())
    
        if args.GPU:
            torch.cuda.synchronize(device = args.device)
        t2 = time.time()

        

        print(training_time_prefix, t2 - t1)
        
        # if args.GPU_measure:
        #     print_gpu_utilization(GPU_mem_usage_list, None)
        
        stopped_epoch = len(random_ids_multi_super_iterations)
        
        if regularization_prob_samples == 0:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)
        
        else:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation F1', args.GPU, args.device, f1=True)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'test F1', args.GPU, args.device, f1=True)
        
        if not args.incremental:
            torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count))
        
            torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count))
            
            torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count))
            
            torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count))
        else:
            torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
        
            torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count)  + incremental_suffix)
        
            torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
        
            torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
        
        
#         influences0 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count))
#         
#         influences1 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count-1))
#         
#         full_removed_id_tensor0 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count), map_location='cpu')
#          
#         full_removed_id_tensor1 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count-1), map_location='cpu')
#         
#         full_removed_values_tensor0 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count), map_location='cpu')
#          
#         full_removed_values_tensor1 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count-1), map_location='cpu')
#         
#         loss_gap = determine_threshold(updated_w_list, updated_grad_list, w_list, grad_list, model, full_training_noisy_dataset, updated_training_dataset, validation_dataset, criterion, optimizer, random_ids_multi_super_iterations, args.bz, args.wd)
        
        
        print('here')
        
        
        
#         updated_origin_grad_list = update_gradient_incremental(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device, exp_updated_origin_grad_list)
#     else:
#
#
#         period = args.period
#
#         init_epochs = args.init
#
#         m = args.hist_size
#
#         cached_size = 100#args.cached_size
#
# #         prev_full_existing_labeled_id_tensor = torch.load(full_out_dir + '/prev_labeled_id_tensor')
#
#         if True:#check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
#     #         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
# #             w_list = torch.load(full_out_dir + '/w_list_sl_v' + str(1))
# #       
# #             grad_list = torch.load(full_out_dir + '/grad_list_sl_v' + str(1))
#
#             if args.GPU:
#                 torch.cuda.synchronize(device = args.device)
#             t2 = time.time()
#
#
#
#             grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
#
#         #             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
#
#
#
#             '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
#
#             exp_updated_w_list = None#torch.load(full_out_dir + '/w_list_sl_v' + str(iter_count))
#
#             exp_updated_grad_list = None#torch.load(full_out_dir + '/grad_list_sl_v' + str(iter_count))
#
# #             t4 = time.time()
#
#             set_model_parameters(model, w_list[0], args.device)
#
#             if args.GPU_measure:
#                 updated_model, updated_w_list, updated_grad_list, GPU_mem_usage_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False, GPU_measure = True, GPUID = args.GPUID)
#             else:
#                 # updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False)
#                 updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list = exp_updated_w_list, exp_updated_grad_list = exp_updated_grad_list, compare = False, r_weight_old = r_weight_old, r_weight_new = r_weight)
#
#             if args.GPU:
#                 torch.cuda.synchronize(device = args.device)
#             t3 = time.time()
#
# #             S_k_list, Y_k_list = None, None
# #             
# #             sigma_k_list = None
# #             
# #             mat_prime_list = None
# #             
# #             if check_existing_files(full_out_dir + '/S_k_list_tensor_' + str(iter_count-1)):
# #                 S_k_list = torch.load(full_out_dir + '/S_k_list_tensor_' + str(iter_count-1))
# #                 
# #             if check_existing_files(full_out_dir + '/Y_k_list_tensor_' + str(iter_count-1)):
# #                 Y_k_list = torch.load(full_out_dir + '/Y_k_list_tensor_' + str(iter_count-1))
# #             
# #             if check_existing_files(full_out_dir + '/sigma_k_list_' + str(iter_count-1)):
# #                 sigma_k_list = torch.load(full_out_dir + '/sigma_k_list_' + str(iter_count-1))
# #             
# #             if check_existing_files(full_out_dir + '/mat_prime_list_' + str(iter_count-1)):
# #                 mat_prime_list = torch.load(full_out_dir + '/mat_prime_list_' + str(iter_count-1))
# #             
# #             set_model_parameters(model, w_list[0], args.device)
# #             
# #             updated_model_2, updated_w_list_2, updated_grad_list_2, updated_S_k_list_2, updated_Y_k_list_2,mat_prime_list, sigma_k_list, grad_origin_label_list = model_update_deltagrad3(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False, S_k_list = S_k_list, Y_k_list = Y_k_list, mat_prime_list = mat_prime_list, sigma_k_list = sigma_k_list)
# #             
# #             origin_w_list = torch.load(full_out_dir + '/w_list_v0')
# #             
# #             origin_grad_list = torch.load(full_out_dir + '/grad_list_v0')
# #             
# #             print(len(grad_origin_label_list))
# #             
# #             updated_S_k_list, updated_Y_k_list, updated_sigma_k_list, updated_mat_prime_list = construct_s_k_y_k_list(init_epochs, period, origin_w_list, origin_grad_list, updated_w_list, grad_origin_label_list, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, m, args.wd, args.GPU, args.device)
# #             
# # #             print(torch.norm(torch.cat(updated_S_k_list_2, dim = 0) - torch.cat(updated_S_k_list, dim = 0)))
# # #             
# # #             print(torch.norm(torch.cat(updated_Y_k_list_2, dim = 0) - torch.cat(updated_Y_k_list, dim = 0)))
# #             
# #             S_k_list_tensor = torch.cat(updated_S_k_list, dim = 0)
# #             
# #             Y_k_list_tensor = torch.cat(updated_Y_k_list, dim = 0)
# #             
# #             torch.save(S_k_list_tensor, full_out_dir + '/S_k_list_tensor_' + str(iter_count))
# #             
# #             torch.save(Y_k_list_tensor, full_out_dir + '/Y_k_list_tensor_' + str(iter_count))
# #             
# #             torch.save(updated_sigma_k_list, full_out_dir + '/sigma_k_list_' + str(iter_count))
# #             
# #             torch.save(updated_mat_prime_list, full_out_dir + '/mat_prime_list_' + str(iter_count))
#
#
#         else:
#
#             if args.GPU:
#                 torch.cuda.synchronize(device = args.device)
#
#             set_model_parameters(model, w_list[0], args.device)    
#
#             t2 = time.time()
#
#             if args.GPU_measure:
#                 updated_w_list, updated_grad_list,_,GPU_mem_usage_list = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = True, measure_GPU_utils = True, r_weight = r_weight)
#             else:
#                 updated_w_list, updated_grad_list,_ = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = True, r_weight = r_weight)
#
#
#             if args.GPU:
#                 torch.cuda.synchronize(device = args.device)
#             t3 = time.time()
#
#             torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
#
#             torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
#
#             torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
# #             torch.save(full_existing_labeled_id_tensor, full_out_dir + '/prev_labeled_id_tensor')
#
#         print(training_time_prefix, t3 - t2)
#
#         # if args.GPU_measure:
#         #     print_gpu_utilization(GPU_mem_usage_list, None)
#
#         stopped_epoch = len(random_ids_multi_super_iterations)
#
#         model, stopped_epoch = select_params_early_stop(args.epochs,random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
#
#         valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
#
#         valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
#
#
#         valid_model_dataset(model, validation_dataset, None, args.bz, 'validation F1', args.GPU, args.device, f1=True)
#
#         valid_model_dataset(model, dataset_test, None, args.bz, 'test F1', args.GPU, args.device, f1=True)
#
#         torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#
#         torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#
#         torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#
#         torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#     if iter_count == last_iter:


    updated_model_param = get_vectorized_params(model)
    
    print('model param updates::', torch.norm(updated_model_param - origin_model_param))

        
    return updated_w_list, updated_grad_list, model
#     if check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset).item() and (not check_iteration_count_file_existence(full_out_dir)):
#             
#             m = args.hist_size
#             
#             prev_grads = get_vectorized_grads_sample_wise(model, w_list, grad_list, m, optimizer, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, args.device)
#             
#             torch.save(prev_grads, full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count - 1))
#     
#             curr_grads = get_vectorized_grads_sample_wise(model, updated_w_list, updated_grad_list, m, optimizer, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, args.device)
#             
#             torch.save(curr_grads, full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count))
#             
#             store_iteration_count(iter_count - 1, full_out_dir)
#         
#     else:
#         if not check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset).item():
#             remove_iteration_count_file(full_out_dir)
#     if exp_updated_w_list is not None and len(exp_updated_w_list) > 0:
#         print('model para diff::',torch.norm(get_all_vectorized_parameters1(updated_w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))
#         print('model para change::',torch.norm(get_all_vectorized_parameters1(w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))
  

# def model_training_main(model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, suffix):
#     exp_updated_origin_grad_list = None 
#
# #         exp_updated_origin_grad_list = update_gradient_origin(all_entry_grad_list, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.GPU, args.device, w_list, model, optimizer)
#
#     criterion = model.soft_loss_function_reduce
#
#     set_model_parameters(model, w_list[0], args.device)
#
#     print('sample count with changed labels::', torch.sum(ids_with_changed_ids))
#
#     exp_updated_w_list, exp_updated_grad_list = None, None
#
#     # if not is_incremental:
# #     if True:
#
#
#     r_weight_old = r_weight.clone()
#
#     r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)*regularization_prob_samples
#
#     r_weight[full_existing_labeled_id_tensor.type(torch.long)] = 1
#
#     # training_data_feature = full_training_noisy_dataset.data[full_existing_labeled_id_tensor]
#     #
#     # training_label_tensor = updated_labels[full_existing_labeled_id_tensor]
#
#     # training_data_feature = torch.cat([training_data_feature, validation_dataset.data], dim = 0)
#     #
#     # training_label_tensor = torch.cat([training_label_tensor, onehot(validation_dataset.labels, args.num_class)], dim = 0).detach()
#
#     # training_dataset = MyDataset(training_data_feature, training_label_tensor)
#
#     # print(training_data_feature, training_label_tensor, updated_labels, full_existing_labeled_id_tensor)
#
#     t1 = time.time()
#
#
#     '''args, model, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping=True, test_performance = True, measure_GPU_utils = False'''
#     updated_w_list, updated_grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, training_dataset.data, training_label_tensor, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = True, r_weight = r_weight)
#
#     final_exp_updated_model_param = get_all_vectorized_parameters1(model.parameters())
#
#     t2 = time.time()
#
#     print('time1::', t2 - t1)
#
#     model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
#
#     valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
#
#     valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
#
#     valid_model_dataset(model, validation_dataset, None, args.bz, 'validation F1', args.GPU, args.device, f1=True)
#
#     valid_model_dataset(model, dataset_test, None, args.bz, 'test F1', args.GPU, args.device, f1=True)
# #         valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=True)
# #         
# #         valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=True)
#
#     # if not args.incremental:
#
#     torch.save(updated_w_list, full_out_dir + '/w_list_' + suffix + '_v' + str(iter_count))
#
#     torch.save(updated_grad_list, full_out_dir + '/grad_list_' + suffix + '_v' + str(iter_count))
#
#     torch.save(model, full_out_dir + '/model_' + suffix + '_v' + str(iter_count))
#
#     torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + suffix + '_v' + str(iter_count))
#     # else:
#         # torch.save(updated_w_list, full_out_dir + '/w_list_' + suffix + 'v' + str(iter_count) + incremental_suffix)
#         #
#         # torch.save(model, full_out_dir + '/model_v' + str(iter_count) + incremental_suffix)
#         #
#         # torch.save(updated_grad_list, full_out_dir + '/grad_list_v' + str(iter_count) + incremental_suffix)
#         #
#         # torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_al_v' + str(iter_count) + incremental_suffix)
#
#     updated_training_dataset = models.MyDataset(full_training_noisy_dataset.data, updated_labels)
#
#
# #         influences0 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count))
# #         
# #         influences1 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count-1))
# #         
# #         full_removed_id_tensor0 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count), map_location='cpu')
# #          
# #         full_removed_id_tensor1 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count-1), map_location='cpu')
# #         
# #         full_removed_values_tensor0 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count), map_location='cpu')
# #          
# #         full_removed_values_tensor1 = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count-1), map_location='cpu')
# #         
# #         loss_gap = determine_threshold(updated_w_list, updated_grad_list, w_list, grad_list, model, full_training_noisy_dataset, updated_training_dataset, validation_dataset, criterion, optimizer, random_ids_multi_super_iterations, args.bz, args.wd)
# #         
# #         
# #         print('here')
#
#
#
# #         updated_origin_grad_list = update_gradient_incremental(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device, exp_updated_origin_grad_list)
#     # else:
#     #
#     #
#         # period = args.period
#         #
#         # init_epochs = args.init
#         #
#         # m = args.hist_size
#         #
#         # cached_size = 10000#args.cached_size
#         #
# # #         prev_full_existing_labeled_id_tensor = torch.load(full_out_dir + '/prev_labeled_id_tensor')
#         #
#         # if check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
#     # #         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
# # #             w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count-1))
# # #     
# # #             grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count-1))
#             #
#             #
#             # t2 = time.time()
#             #
#             #
#             #
#             # grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
#             #
#         # #             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
#         #
#         #
#         #
#             # '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
#             #
# # #             exp_updated_w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count))
# # #         
# # #             exp_updated_grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count))
#             #
#             # t4 = time.time()
#             #
#             # set_model_parameters(model, w_list[0], args.device)
#             #
#             # updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False)
#             #
#             # t3 = time.time()
#             #
#         # else:
#         #
#             # t2 = time.time()
#             #
#             # updated_w_list, updated_grad_list,_ = train_model_dataset(model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False)
#             #
#             # t3 = time.time()
# # #             torch.save(full_existing_labeled_id_tensor, full_out_dir + '/prev_labeled_id_tensor')
#         #
#         # print('time2::', t3 - t2)
#         #
#         # model = select_params_early_stop(random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
#         #
#         # valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
#         #
#         # valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
#         #
#         #
# # #         valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=True)
# # #         
# # #         valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=True)
#         #
#         # torch.save(updated_w_list, full_out_dir + '/w_list_v' + str(iter_count) + incremental_suffix)
#         #
#         # torch.save(model, full_out_dir + '/model_v' + str(iter_count) + incremental_suffix)
#         #
#         # torch.save(updated_grad_list, full_out_dir + '/grad_list_v' + str(iter_count) + incremental_suffix)
#
# #     if iter_count == last_iter:
#     print('collect influence function history info')
#     '''w_list, grad_list, model, training_dataset, loss_func, optimizer, random_ids_all_epochs, batch_size, regularization_rate, num_class'''
#
#     # if args.incremental:
#     #
#         # model_param = get_model_para_list(model)
#         #
#         # updated_dataset = models.MyDataset(full_training_noisy_dataset.data, updated_labels)
#         #
#         # w_diff_remove_1, loss_gap_coefficient, w_diff_remove_1_2 = pre_compute_influence_order_history(updated_w_list, updated_grad_list, model, updated_dataset, model.soft_loss_function_reduce, optimizer, random_ids_multi_super_iterations, args.bz, args.wd, args.num_class)
#         # torch.save(w_diff_remove_1_2, full_out_dir + '/w_diff_remove_1_' + str(iter_count))
#         #
#         # torch.save(loss_gap_coefficient, full_out_dir + '/loss_gap_coefficient_' + str(iter_count))
#         #
#         # set_model_parameters(model, model_param, args.device)
#
#     return updated_w_list, updated_grad_list, model
# #         torch.save(mu, full_out_dir + '/mu_' + str(iter_count))
#
# #     if check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset).item() and (not check_iteration_count_file_existence(full_out_dir)):
# #             
# #             m = args.hist_size
# #             
# #             prev_grads = get_vectorized_grads_sample_wise(model, w_list, grad_list, m, optimizer, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, args.device)
# #             
# #             torch.save(prev_grads, full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count - 1))
# #     
# #             curr_grads = get_vectorized_grads_sample_wise(model, updated_w_list, updated_grad_list, m, optimizer, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, args.device)
# #             
# #             torch.save(curr_grads, full_out_dir + '/all_sample_wise_grad_list_' + str(iter_count))
# #             
# #             store_iteration_count(iter_count - 1, full_out_dir)
# #         
# #     else:
# #         if not check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset).item():
# #             remove_iteration_count_file(full_out_dir)
# #     if exp_updated_w_list is not None and len(exp_updated_w_list) > 0:
# #         print('model para diff::',torch.norm(get_all_vectorized_parameters1(updated_w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))
# #         print('model para change::',torch.norm(get_all_vectorized_parameters1(w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))


    
def load_retraining_influence_info(iter_count, full_out_dir, args):
    sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count) + incremental_suffix)
        
    ordered_list = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count) + incremental_suffix)

    influences1 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count) + incremental_suffix)    

    exp_sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count))
        
    exp_ordered_list = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count))

    influences2 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count))
    
    return influences2,exp_ordered_list,exp_sorted_train_ids,influences1,ordered_list,sorted_train_ids 


#     print('sub time2::', t3 - t4)
def load_retraining_history_info(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count))
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count))
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count),map_location=torch.device('cpu'))

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count))
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count))
    
    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list    

def load_incremental_history_info(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix,map_location=torch.device('cpu'))

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)

    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list


def load_history_info(is_incremental, iter_count, full_out_dir, args):

    print('iter count::', iter_count)
    
    if iter_count == 0 or (not is_incremental):
    
#         else:
#             iter_count = 0
        
        return load_retraining_history_info(iter_count, full_out_dir, args)
        
        
        
        
    else:
        return load_incremental_history_info(iter_count, full_out_dir, args)
        

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
    if_dnn = True

    if args.model == 'Binary_Logistic_regression' or args.model == 'Logistic_regression':
        if_dnn = False
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, train_annotated_label_tensor, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids, train_origin_dataset, valid_origin_dataset, test_origin_dataset = obtain_data_function(args, noisy=True, load_origin = if_dnn, is_tars = args.tars)
    
    size = None
    
    continue_labeling = args.continue_labeling
    
    is_incremental = args.incremental
    
    load_incremental = args.incremental
    
    suffix = args.suffix#'al'
    
    # if args.no_probs:
    #
        # suffix += '_probs'
    
    # args.suffix = suffix
    
    
    
    if if_dnn:
        validation_dataset = valid_origin_dataset
        
        dataset_test = test_origin_dataset
        
        full_training_noisy_dataset = train_origin_dataset
    
    if args.start:
        
        remove_different_version_dataset(full_out_dir, args.model)
        
        # random_ids_multi_super_iterations = None
        
#         os.remove(full_out_dir + '/model_initial')
        
        if not args.restart and os.path.exists(full_out_dir + '/model_initial'):
            model = torch.load(full_out_dir + '/model_initial',map_location=torch.device('cpu'))
        
            w_list = torch.load(full_out_dir + '/w_list_initial')
            
            grad_list = torch.load(full_out_dir + '/grad_list_initial')
        
#             if os.path.exists(full_out_dir + '/random_ids_multi_super_iterations'):
            # random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
        
        else:
            
            # if not args.restart and os.path.exists(full_out_dir + '/random_ids_multi_super_iterations'):
                # random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
            
            clean_sample_ids = None

            if True:#args.no_probs:
                clean_sample_ids = torch.load(full_out_dir + '/clean_sample_ids')
            
            training_data_feature = full_training_noisy_dataset.data.clone()
            
            training_data_labels = full_training_noisy_dataset.labels.clone()
            
            # if len(clean_sample_ids) > 0:
            #     if clean_sample_ids is not None:
            #         training_data_feature = training_data_feature[clean_sample_ids]
            #         training_data_labels = training_data_labels[clean_sample_ids]
            #
            #     training_data_feature = torch.cat([training_data_feature, validation_dataset.data], dim = 0)
            #
            #     training_data_labels = torch.cat([training_data_labels, onehot(validation_dataset.labels, args.num_class)], dim = 0)
            #
            # else:
            #     training_data_feature = validation_dataset.data.clone()
            #
            #     training_data_labels = onehot(validation_dataset.labels, args.num_class)
            #
            # training_dataset = MyDataset(training_data_feature, training_data_labels)
            #
            # w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(training_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
            #
            #
            # model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
            r_weight = torch.ones(full_training_noisy_dataset.data.shape[0],dtype = full_training_noisy_dataset.data.dtype)
            
            if os.path.exists(full_out_dir + '/clean_sample_ids'):
                origin_labeled_tensor = torch.load(full_out_dir + '/clean_sample_ids')
                
                if type(origin_labeled_tensor) is list:
                    origin_labeled_tensor = torch.tensor(origin_labeled_tensor)
                    torch.save(origin_labeled_tensor, full_out_dir + '/clean_sample_ids')
                
                print('clean sample ids::', origin_labeled_tensor)
                
                if len(origin_labeled_tensor) > 0:
                    r_weight *= args.regular_rate
                    
                    r_weight[origin_labeled_tensor.type(torch.long)] = 1
                
            
            
            print('r weight unique::', torch.unique(r_weight))
            if args.regular_rate == 0:
                # training_data_feature = full_training_noisy_dataset.data[origin_labeled_tensor]
                #
                # training_data_labels = full_training_noisy_dataset.labels[origin_labeled_tensor]
                
                if origin_labeled_tensor.shape[0] > 0:
                    training_data_feature = full_training_noisy_dataset.data[origin_labeled_tensor]
                    
                    training_data_labels = full_training_noisy_dataset.labels[origin_labeled_tensor]
                
                else:
                    training_data_feature = full_training_noisy_dataset.data
                    
                    training_data_labels = full_training_noisy_dataset.labels
                
                training_dataset = MyDataset(training_data_feature, training_data_labels)
                
                w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(training_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
            
                model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)
                  
            else:
                if if_dnn:
                    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(train_origin_dataset, valid_origin_dataset, test_origin_dataset, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations, r_weight = r_weight)
                    
                    # stopped_epoch = 0
                    model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, train_origin_dataset, valid_origin_dataset, test_origin_dataset, args.bz, args.GPU, args.device, no_prov = args.no_prov)
                    
                else:
                    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations, r_weight = r_weight)
            
            
                    model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)

            
            torch.save(model, full_out_dir + '/model_initial')
            
            torch.save(w_list, full_out_dir + '/w_list_initial')
            
            torch.save(grad_list, full_out_dir + '/grad_list_initial')
        
        
        '''first iteration of selecting samples by the influence function and labeled by experts::'''
        
#         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, 0, model.soft_loss_function, args.model, args.num_class)
#         optimizer = model.get_optimizer(args.tlr, args.wd)
#         updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids = select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, False, 0, None, w_list, grad_list, derive_probab_labels = False)


#         torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids_v0')
#         
#         torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v0')    

        '''label the selected samples and update the probablistic labels'''
        
        
        
#         updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=None, size = None, start = True, iter_count = 0, derive_probab_labels=False)
        
#         ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)

#         print('samples with updated labels::', torch.nonzero(ids_with_changed_ids), torch.sum(ids_with_changed_ids))
        
        
        args.incremental = False
        
#         exp_w_list, exp_grad_list, model = model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, False, 0, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, start = True)
        
#         model = select_params_early_stop(random_ids_multi_super_iterations, exp_w_list, model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
        # model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
        
#         full_training_noisy_dataset.labels = updated_labels
#         w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)

#         torch.save(model, full_out_dir + '/model_v0')

        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation', args.GPU, args.device, f1=False)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test', args.GPU, args.device, f1=False)

        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)

#         w_list = torch.load(full_out_dir + '/w_list_v0')
#         
#         grad_list = torch.load(full_out_dir + '/grad_list_v0')
#         
#         
#         model = torch.load(full_out_dir + '/model_v0')


#         w_diff_remove_1, c2, mu = pre_compute_influence_order_history(exp_w_list, exp_grad_list, model, full_training_noisy_dataset, model.soft_loss_function_reduce, optimizer, random_ids_multi_super_iterations, args.bz, args.wd)

        # simulate_human_annotations(3, full_training_origin_labels, full_out_dir)

        
        # torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        
        torch.save(w_list, full_out_dir + '/w_list_' + suffix + '_v0')
         
        torch.save(grad_list, full_out_dir + '/grad_list_' + suffix + '_v0')
        
        torch.save(model, full_out_dir + '/model_' + suffix + '_v0')
            
        # torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + suffix + '_v0')
        
#         torch.save(w_diff_remove_1, full_out_dir + '/w_diff_remove_1_0')
#         
#         torch.save(c2, full_out_dir + '/c2_0')
#         
#         torch.save(mu, full_out_dir + '/mu_0')
        
#         all_entry_grad_list = obtain_gradients_each_class(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, args.GPU, args.device)
#         
#         torch.save(all_entry_grad_list, full_out_dir + '/all_entry_grad_list')

        if os.path.exists(full_out_dir + '/clean_sample_ids'):
            origin_labeled_tensor = torch.load(full_out_dir + '/clean_sample_ids')
        
            if type(origin_labeled_tensor) is list:
                origin_labeled_tensor = torch.tensor(origin_labeled_tensor)
                torch.save(origin_labeled_tensor, full_out_dir + '/clean_sample_ids')
        
            
                torch.save(origin_labeled_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v0')
            else:
                torch.save(origin_labeled_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v0')
        else:
            torch.save(torch.Tensor(), full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v0')
        
        torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + suffix + '_v0')
        
        iter_count = 0
        
        torch.save(iter_count, full_out_dir + '/labeling_iter_' + suffix)

    else:
        
#         if continue_labeling:
        
        
        
        iter_count = torch.load(full_out_dir + '/labeling_iter_' + suffix)
        # iter_count = 0


        full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list= load_history_info(load_incremental, iter_count, full_out_dir, args)

        model = model.to(args.device)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation', args.GPU, args.device, f1=False)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test', args.GPU, args.device, f1=False)

        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)

        optimizer = model.get_optimizer(args.tlr, args.wd)
        
        if len(full_existing_labeled_id_tensor) > 0:
            r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)*args.regular_rate
    
            r_weight[full_existing_labeled_id_tensor.type(torch.long)] = 1
        else:
            r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)
        
        
        updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids = select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, is_incremental, iter_count + 1, full_existing_labeled_id_tensor,w_list, grad_list, derive_probab_labels = False, train_annotated_label_tensor = train_annotated_label_tensor, suffix = suffix, r_weight = r_weight)
        
#         if continue_labeling:

#         labeled_dataset = models.MyDataset(full_training_noisy_dataset.data[full_existing_labeled_id_tensor], updated_labels[full_existing_labeled_id_tensor])

        '''full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, start = False, regularization_prob_samples = 0.1, r_weight = None'''

        model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count + 1, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, suffix, r_weight = r_weight, regularization_prob_samples = args.regular_rate)

        
        full_training_noisy_dataset.labels = updated_labels
        
        if not is_incremental:
            torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count+1))
            
#             torch.save(model, full_out_dir + '/model_v' + str(iter_count+1))
            
            torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count+1))
        
        else:
            torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count+1) + incremental_suffix)
            
            
            torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count+1) + incremental_suffix)
            
        if continue_labeling:
        
            torch.save(iter_count + 1, full_out_dir + '/labeling_iter_' + suffix)
            
#         all_entry_grad_list = torch.load(full_out_dir + '/all_entry_grad_list')
        
#         all_entry_grad_list_new = transform_entry_grad(all_entry_grad_list, full_training_noisy_dataset.data.shape[0], args.bz, args.epochs)
#     
#         print(all_entry_grad_list_new.shape)
#     
#         all_entry_grad_list = all_entry_grad_list_new
    
        
    
    