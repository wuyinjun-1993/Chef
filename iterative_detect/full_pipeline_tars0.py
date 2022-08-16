'''
Created on May 6, 2021

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

# import active_learning_query



def load_retraining_history_info(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + args.resolve_conflict_str, map_location=torch.device('cpu'))

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + args.resolve_conflict_str)
    
    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list    

def load_retraining_history_info0(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count))
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count))
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count))

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count))
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count))
    
    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list    


def load_incremental_history_info(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + args.resolve_conflict_str)

    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list

def load_incremental_history_info0(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix)

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
        if iter_count ==0:
            args.resolve_conflict_str = ''
        if len(args.resolve_conflict_str) == 0:
            return load_retraining_history_info0(iter_count, full_out_dir, args)
        else:
            return load_retraining_history_info(iter_count, full_out_dir, args)
        
        
        
        
    else:
        
        if len(args.resolve_conflict_str) == 0:
            return load_incremental_history_info0(iter_count, full_out_dir, args)
        else:
            return load_incremental_history_info(iter_count, full_out_dir, args)


def compute_model_improvement(full_out_dir, args, training_dataset, remaining_sample_ids, model, random_ids_multi_super_iterations, w_list, grad_list, valid_dataset, train_prob_case2, criterion, full_existing_labeled_id_tensor):
    
    period = args.period
        
    init_epochs = args.init
    
    m = args.hist_size
    
    cached_size = 100
    
    
    
    ids_with_changed_ids = torch.zeros(training_dataset.data.shape[0]).bool()
    
    ids_with_unchanged_ids = torch.ones(training_dataset.data.shape[0]).bool()
    
    start_valid_loss = valid_model_dataset(model, valid_dataset, None, args.bz, 'valid F1', args.GPU, args.device, f1=True)
    
    valid_loss_change = []
    
    r_weight = torch.ones(training_dataset.data.shape[0],dtype = training_dataset.data.dtype)
    
    if len(full_existing_labeled_id_tensor) > 0:
        r_weight[remaining_sample_ids] = args.regular_rate
    
    
    for k in range(remaining_sample_ids.shape[0]):
        # curr_x = training_dataset.data[remaining_sample_ids[k]]
        # curr_y = training_dataset.labels[remaining_sample_ids[k]]
    
        r_weight_copy = r_weight.clone()
    
        set_model_parameters(model, w_list[0], args.device)
    
        grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
        
        updated_labels = training_dataset.labels.clone()
        
        updated_labels[remaining_sample_ids[k]] = 1 - updated_labels[remaining_sample_ids[k]]
        
        r_weight_copy[remaining_sample_ids[k]] = 1
        # ids_with_changed_ids = torch.tensor([remaining_sample_ids[k]])
        
        ids_with_changed_ids[remaining_sample_ids[k]] = True
        
        ids_with_unchanged_ids[remaining_sample_ids[k]] = False
        
        # ids_with_unchanged_ids = torch.nonzero(full_id_tensor).view(-1)
        
        updated_w_list, updated_grad_list,_ = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, training_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False, exp_w_list = w_list, exp_grad_list = grad_list, r_weight = r_weight_copy)
        
        model_para_list0 = get_model_para_list(model)
        
        set_model_parameters(model, w_list[0], args.device)
        
        updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, training_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list = updated_w_list, exp_updated_grad_list = updated_grad_list, compare = True, r_weight_old = r_weight, r_weight_new = r_weight_copy)
        
        curr_valid_loss = valid_model_dataset(updated_model, valid_dataset, None, args.bz, 'valid F1', args.GPU, args.device, f1=True)
    
        valid_loss_change.append(curr_valid_loss - start_valid_loss)
        
    valid_loss_change_tensor = torch.tensor(valid_loss_change) 
    
    valid_loss_change_tensor = valid_loss_change_tensor.view(-1) * train_prob_case2.view(-1)
    
    sorted_loss_values, sorted_ids = torch.sort(valid_loss_change_tensor, descending = True)
    
    return sorted_ids

def select_samples_influence_function_main(w_list, grad_list, random_ids_multi_super_iterations, args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, is_incremental, iter_count, full_existing_labeled_id_tensor, curr_w_list, curr_grad_list, derive_probab_labels=False, train_annotated_label_tensor = None, suffix = 'al', method = 'tars'):
    
    # uncert_sampling = active_learning_query.UncertaintySampling(True)
    
    # active_leaning_method_func = getattr(uncert_sampling, method)
    
#     active_leaning_method_func0 = getattr(uncert_sampling, method + '0')
    
    
#     if (not is_incremental) or (not check_iteration_count_file_existence(full_out_dir) or not compare_iteration_count(iter_count, full_out_dir)):
    if True:#(not is_incremental) or (iter_count <= last_iter): 
        
#         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
        
        criterion = model.soft_loss_function_reduce
        final_remaining_ids = None
        
        if full_existing_labeled_id_tensor is not None:
            remaining_ids = torch.tensor(list(set(range(full_training_noisy_dataset.data.shape[0])).difference(set(full_existing_labeled_id_tensor.tolist()))))
        else:
            remaining_ids = torch.tensor(list(range(full_training_noisy_dataset.data.shape[0])))
        
        remaining_dataset_full = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
        
        t1 = time.time()
        
        
        train_prob_case2 = torch.load(os.path.join(full_out_dir, 'train_prob_case2'))
        
        consolidate_label_list_tensor = torch.load(os.path.join(full_out_dir, 'consolidate_label_list')) 
        
        sorted_ids = compute_model_improvement(full_out_dir, args, full_training_noisy_dataset, remaining_ids, model, random_ids_multi_super_iterations, w_list, grad_list, validation_dataset, train_prob_case2, criterion, full_existing_labeled_id_tensor)
        
        # updated_labels = full_training_noisy_dataset.labels.clone()
        #
        # updated_labels[sorted_ids] = consolidate_label_list_tensor[sorted_ids] 
        
        # prob_dist_list_tensor, influences1, ordered_list, sorted_train_ids = uncert_sampling.get_samples_batch(model, full_training_noisy_dataset.data[remaining_ids], active_leaning_method_func, number=args.removed_count, batch_size = args.bz, is_GPU = args.GPU, device = args.device)
        
#         prob_dist_list_tensor_exp, influences1_exp, ordered_list_exp, sorted_train_ids_exp = uncert_sampling.get_samples(model, full_training_noisy_dataset.data[remaining_ids], active_leaning_method_func0, number=args.removed_count)
        
        # final_prod_dist_list_tensor = torch.zeros([full_training_noisy_dataset.data.shape[0], prob_dist_list_tensor.shape[1]], dtype = prob_dist_list_tensor.dtype)
        #
        # final_prod_dist_list_tensor[remaining_ids] = prob_dist_list_tensor
        
        # full_influences1 = torch.zeros(full_training_noisy_dataset.data.shape[0], dtype = prob_dist_list_tensor.dtype)
        #
        # full_influences1[remaining_ids] = influences1
        
        origin_sorted_train_ids = remaining_ids[sorted_ids]
        
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
        
            # torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count))
            
            # torch.save(full_influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count))
            #
            # torch.save(final_prod_dist_list_tensor, full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count))
        else:
            torch.save(origin_sorted_train_ids, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        
            # torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix)
    
            # torch.save(full_influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            #
            # torch.save(final_prod_dist_list_tensor, full_out_dir + '/' + args.model + '_' + method + 'probs_' + suffix + '_v' + str(iter_count) + incremental_suffix)
            
        print('calculate influence time 1::', t2 - t1)

    updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels_tars(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False, iter_count = iter_count, is_incremental = is_incremental, derive_probab_labels = derive_probab_labels, method = method, train_annotated_label_tensor = train_annotated_label_tensor, suffix = suffix)

    ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)

    print('samples with updated labels::', torch.nonzero(ids_with_changed_ids), torch.sum(ids_with_changed_ids))



    return updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids


def model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, start = False, regularization_prob_samples = 0.1, r_weight = None):
    
    
    
    model = model.to(args.device)
    
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
#     if True:
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
            # print_gpu_utilization(GPU_mem_usage_list, None)
        
        stopped_epoch = len(random_ids_multi_super_iterations)
        
        if regularization_prob_samples == 0:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
        else:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation F1', args.GPU, args.device, f1=True)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'test F1', args.GPU, args.device, f1=True)
        
        if not args.incremental:
            torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count)+ args.resolve_conflict_str)
        
            torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count)+ args.resolve_conflict_str)
            
            torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count)+ args.resolve_conflict_str)
            
            torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count)+ args.resolve_conflict_str)
        else:
            torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        
            torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count)  + incremental_suffix+ args.resolve_conflict_str)
        
            torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        
            torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        
        
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
    # else:
    #
    #
        # period = args.period
        #
        # init_epochs = args.init
        #
        # m = args.hist_size
        #
        # cached_size = 100#args.cached_size
        #
# #         prev_full_existing_labeled_id_tensor = torch.load(full_out_dir + '/prev_labeled_id_tensor')
        #
        # if True:#check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
    # #         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
# #             w_list = torch.load(full_out_dir + '/w_list_sl_v' + str(1))
# #       
# #             grad_list = torch.load(full_out_dir + '/grad_list_sl_v' + str(1))
            #
            # if args.GPU:
                # torch.cuda.synchronize(device = args.device)
            # t2 = time.time()
            #
            #
            #
            # grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
            #
        # #             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
        #
        #
        #
            # '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
            #
            # exp_updated_w_list = None#torch.load(full_out_dir + '/w_list_sl_v' + str(iter_count))
            #
            # exp_updated_grad_list = None#torch.load(full_out_dir + '/grad_list_sl_v' + str(iter_count))
            #
# #             t4 = time.time()
            #
            # set_model_parameters(model, w_list[0], args.device)
            #
            # if args.GPU_measure:
                # updated_model, updated_w_list, updated_grad_list, GPU_mem_usage_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False, GPU_measure = True, GPUID = args.GPUID)
            # else:
                # updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False)
                #
                #
            # if args.GPU:
                # torch.cuda.synchronize(device = args.device)
            # t3 = time.time()
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
        # else:
        #
            # if args.GPU:
                # torch.cuda.synchronize(device = args.device)
                #
            # set_model_parameters(model, w_list[0], args.device)    
            #
            # t2 = time.time()
            #
            # if args.GPU_measure:
                # updated_w_list, updated_grad_list,_,GPU_mem_usage_list = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = True, measure_GPU_utils = True, r_weight = r_weight)
            # else:
                # updated_w_list, updated_grad_list,_ = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = True, r_weight = r_weight)
                #
                #
            # if args.GPU:
                # torch.cuda.synchronize(device = args.device)
            # t3 = time.time()
            #
            # torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
            #
            # torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
            #
            # torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix + '_expected')
# #             torch.save(full_existing_labeled_id_tensor, full_out_dir + '/prev_labeled_id_tensor')
        #
        # print(training_time_prefix, t3 - t2)
        #
        # if args.GPU_measure:
            # print_gpu_utilization(GPU_mem_usage_list, None)
            #
        # stopped_epoch = len(random_ids_multi_super_iterations)
        #
        # model, stopped_epoch = select_params_early_stop(args.epochs,random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        #
        # valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
        #
        # valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
        #
        #
        # valid_model_dataset(model, validation_dataset, None, args.bz, 'validation F1', args.GPU, args.device, f1=True)
        #
        # valid_model_dataset(model, dataset_test, None, args.bz, 'test F1', args.GPU, args.device, f1=True)
        #
        # torch.save(updated_w_list, full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        #
        # torch.save(updated_grad_list, full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        #
        # torch.save(model, full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
        #
        # torch.save(stopped_epoch, full_out_dir + '/stopped_epoch_' + args.suffix + '_v' + str(iter_count) + incremental_suffix+ args.resolve_conflict_str)
#     if iter_count == last_iter:


    updated_model_param = get_vectorized_params(model)
    
    print('model param updates::', torch.norm(updated_model_param - origin_model_param))

        
    return updated_w_list, updated_grad_list, model




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
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, train_annotated_label_tensor, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)
    
    size = None
    
    continue_labeling = args.continue_labeling
    
    is_incremental = args.incremental
    
    load_incremental = args.incremental
    
    suffix = args.suffix#'al'

    regularization_prob_samples = args.regular_rate
    
    if args.start:
        
        remove_different_version_dataset(full_out_dir, args.model)
        
        if not args.restart and os.path.exists(full_out_dir + '/model_initial'):
            model = torch.load(full_out_dir + '/model_initial')
        
            w_list = torch.load(full_out_dir + '/w_list_initial')
            
            grad_list = torch.load(full_out_dir + '/grad_list_initial')
        
        else:
            
            clean_sample_ids = None
            random_ids_multi_super_iterations = None
            # if True:#args.no_probs:
                # clean_sample_ids = torch.load(full_out_dir + '/clean_sample_ids')
                #
            # training_data_feature = full_training_noisy_dataset.data.clone()
            #
            # training_data_labels = full_training_noisy_dataset.labels.clone()
            #
            # if len(clean_sample_ids) > 0:
                # if clean_sample_ids is not None:
                    # training_data_feature = training_data_feature[clean_sample_ids]
                    # training_data_labels = training_data_labels[clean_sample_ids]
                    #
                # # training_data_feature = torch.cat([training_data_feature, validation_dataset.data], dim = 0)
                # #
                # # training_data_labels = torch.cat([training_data_labels, onehot(validation_dataset.labels, args.num_class)], dim = 0)
                #
            # else:
                # training_data_feature = validation_dataset.data.clone()
                #
                # training_data_labels = onehot(validation_dataset.labels, args.num_class)
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
                    r_weight *= regularization_prob_samples
                    
                    r_weight[origin_labeled_tensor.type(torch.long)] = 1
                
            
            print('r weight unique::', torch.unique(r_weight))
            if regularization_prob_samples == 0:
                training_data_feature = full_training_noisy_dataset.data[origin_labeled_tensor]
                
                training_data_labels = full_training_noisy_dataset.labels[origin_labeled_tensor]
                
                training_dataset = MyDataset(training_data_feature, training_data_labels)
                
                w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(training_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
            
                model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
                  
            else:
                w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = random_ids_multi_super_iterations, r_weight = r_weight)
            
            
                model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)

            
            torch.save(model, full_out_dir + '/model_initial')
            
            torch.save(w_list, full_out_dir + '/w_list_initial')
            
            torch.save(grad_list, full_out_dir + '/grad_list_initial')
        
        
        '''first iteration of selecting samples by the influence function and labeled by experts::'''
        

        '''label the selected samples and update the probablistic labels'''
        
        
        args.incremental = False
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation', args.GPU, args.device, f1=False)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test', args.GPU, args.device, f1=False)

        valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
         
        valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)

        # simulate_human_annotations(3, full_training_origin_labels, full_out_dir)
        
        torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        
        torch.save(w_list, full_out_dir + '/w_list_' + suffix + '_v0')
         
        torch.save(grad_list, full_out_dir + '/grad_list_' + suffix + '_v0')
        
        torch.save(model, full_out_dir + '/model_' + suffix + '_v0')
            
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


        optimizer = model.get_optimizer(args.tlr, args.wd)
        
        updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids = select_samples_influence_function_main(w_list, grad_list, random_ids_multi_super_iterations, args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, full_out_dir, is_incremental, iter_count + 1, full_existing_labeled_id_tensor,w_list, grad_list, derive_probab_labels = False, train_annotated_label_tensor = train_annotated_label_tensor, suffix = suffix)
        
#         if continue_labeling:

#         labeled_dataset = models.MyDataset(full_training_noisy_dataset.data[full_existing_labeled_id_tensor], updated_labels[full_existing_labeled_id_tensor])

        model_training_main(model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count + 1, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, suffix)

        
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
    
    
    