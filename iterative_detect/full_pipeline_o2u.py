'''
Created on Jan 13, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob
# from iterative_detect.utils_iters import model_update_deltagrad2,\
    # incremental_suffix

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

label_change_threshold = 1e-5

incremental_training_threshold = 0.05

def check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
    return torch.sum(ids_with_changed_ids)*1.0/full_training_noisy_dataset.data.shape[0] < incremental_training_threshold


def select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model,optimizer,full_out_dir, is_incremental, iter_count, full_existing_labeled_id_tensor, method = 'influence', suffix = 'o2u', train_annotated_label_tensor = None, r_weight = None):
#     if (not is_incremental) or (not check_iteration_count_file_existence(full_out_dir) or not compare_iteration_count(iter_count, full_out_dir)):
    # if (not is_incremental) or iter_count < 2:
        
#         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
    loss_func = model.soft_loss_function_reduce

    o2u_random_ids_multi_epochs = None
    if check_existing_files(full_out_dir + '/o2u_random_ids_multi_epochs'):
        o2u_random_ids_multi_epochs = torch.load(full_out_dir + '/o2u_random_ids_multi_epochs')
    
    t1 = time.time()
    '''model, optimizer, loss_func, full_out_dir, dataset_train, valid_dataset, test_dataset, curr_w_list, curr_grad_list, is_GPU, device, regularization_coeff, batch_size'''
#         influences1, full_grad_tensors, ordered_list0, sorted_train_ids0 = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz)

    influences1, ordered_list, sorted_train_ids, o2u_random_ids_multi_epochs, o2u_w_list, o2u_grad_list = O2U_second_stage(model, optimizer, loss_func, args.o2u_epochs, full_training_noisy_dataset, validation_dataset, dataset_test, None, args.GPU, args.device, args.removed_count, random_ids_multi_epochs = o2u_random_ids_multi_epochs, r_weight =r_weight)

    t2 = time.time()

    if not is_incremental:
    
        torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count))
    
        torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count))
        
        torch.save(influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count))
        
        torch.save(o2u_w_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_w_list_' + suffix + '_v' + str(iter_count))
        
        torch.save(o2u_grad_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_grad_list_' + suffix + '_v' + str(iter_count))
        
        torch.save(o2u_random_ids_multi_epochs, full_out_dir + '/o2u_random_ids_multi_epochs')
    else:
        torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix)
    
        torch.save(ordered_list, full_out_dir + '/' + args.model + '_' + method + '_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix)

        torch.save(influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        
        torch.save(o2u_w_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_w_list_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        
        torch.save(o2u_grad_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_grad_list_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        
        torch.save(o2u_random_ids_multi_epochs, full_out_dir + '/o2u_random_ids_multi_epochs')
        
    print('calculate influence time 1::', t2 - t1)
    # else:
        # o2u_random_ids_multi_epochs = torch.load(full_out_dir + '/o2u_random_ids_multi_epochs')
        #
        # prev_full_training_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + suffix + '_v' + str(iter_count-2))
        #
        # o2u_w_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_o2u_w_list_' + suffix + '_v' + str(iter_count-1) + incremental_suffix)
        #
        # o2u_grad_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_o2u_grad_list_' + suffix + '_v' + str(iter_count-1) + incremental_suffix)
        #
        # ids_with_changed_ids = torch.load(full_out_dir + '/ids_with_changed_ids_' + suffix + '_v' + str(iter_count - 1))
        #
        # ids_with_unchanged_ids = torch.load(full_out_dir + '/ids_with_unchanged_ids_' + suffix + '_v' + str(iter_count - 1))
        #
        #
        #
        # criterion = model.soft_loss_function
        #
        # criterion_no_reduce = model.soft_loss_function_reduce
        #
        # cached_size = 10000
        #
        # grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, o2u_w_list, o2u_grad_list)
        #
        # set_model_parameters(model, o2u_w_list[0], args.device)
        #
        #
        #
        # influences1, ordered_list, sorted_train_ids, o2u_random_ids_multi_epochs, o2u_w_list, o2u_grad_list = model_update_deltagrad2_o2u(args.o2u_epochs, args.period, 1, args.init, prev_full_training_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, full_training_noisy_dataset.labels, ids_with_changed_ids, ids_with_unchanged_ids, args.hist_size, args.lr, o2u_random_ids_multi_epochs, criterion, criterion_no_reduce, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list = None, exp_updated_grad_list = None, compare = False)
        #
        # exp_o2u_w_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_o2u_w_list_' + suffix + '_v' + str(iter_count))
        #
        # exp_o2u_grad_list = torch.load(full_out_dir + '/' + args.model + '_' + method + '_o2u_grad_list_' + suffix + '_v' + str(iter_count))
        #
        #
        # print('here')
        #
# #         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
#
# #         prov_iter_count = load_iteration_count(full_out_dir)
# #         
# #         m = args.hist_size
# # #         prev_sample_wise_grad_list, curr_sample_wise_grad_list = load_sample_wise_grad_list(full_out_dir, prov_iter_count)
# # 
# #         all_ids_with_unchanged_labels_bool, all_ids_with_unchanged_labels = obtain_remaining_ids(full_out_dir, prov_iter_count, iter_count)
# #         
# #         all_ids_with_changed_labels_bool = ~all_ids_with_unchanged_labels_bool
# # 
# #         s_k_list, y_k_list, mini_Y_k_list, mini_Y_k_list2, full_grad_list, prev_w_list, prev_grad_list = obtain_s_y_list(args, all_ids_with_unchanged_labels, full_out_dir, m = m, k = prov_iter_count)
# # 
# # 
# #         loss_func = model.soft_loss_function_reduce
# #         
# #         y_k_list_tensor = torch.stack(y_k_list, dim = 0)
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
# #         influences2, full_grad_tensors, ordered_list, sorted_train_ids = incremental_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, s_k_list, y_k_list_tensor, full_grad_list, curr_w_list, prev_w_list, all_ids_with_unchanged_labels, args.GPU, args.device, args.wd,m, args.bz)
# #         
# #         t2 = time.time()
# #         
# #         print('calculate influence time 1::', t2 - t1)
        #
# #         set_model_parameters(model, curr_w_list[-1], args.device)
        #
# #         influences1, exp_full_grad_tensors, exp_ordered_list, exp_sorted_train_ids = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, curr_w_list, curr_grad_list, args.GPU, args.device, args.wd,args.bz)
        #
        # torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(influences1, full_out_dir + '/' + args.model + '_' + method + '_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(o2u_w_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_w_list_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # torch.save(o2u_grad_list, full_out_dir + '/' + args.model + '_' + method + '_o2u_grad_list_' + suffix + '_v' + str(iter_count) + incremental_suffix)
        #
        # print('here')
    

    updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False, iter_count = iter_count, is_incremental = is_incremental, method = method, suffix = suffix, train_annotated_label_tensor = train_annotated_label_tensor)

    ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)

    print('samples with updated labels::', torch.nonzero(ids_with_changed_ids))



    return updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids


def model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, regularization_prob_samples = 0.1, r_weight = None, suffix = 'o2u'):
    exp_updated_origin_grad_list = None 
    
#         exp_updated_origin_grad_list = update_gradient_origin(all_entry_grad_list, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.GPU, args.device, w_list, model, optimizer)
    
    criterion = model.soft_loss_function_reduce
    
    set_model_parameters(model, w_list[0], args.device)
    
    print('sample count with changed labels::', torch.sum(ids_with_changed_ids))
    
    exp_updated_w_list, exp_updated_grad_list = None, None
    
    if True:
    
        t1 = time.time()
        
        
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
            updated_w_list, updated_grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False, r_weight = r_weight)
    
        final_exp_updated_model_param = get_all_vectorized_parameters1(model.parameters())
    
        t2 = time.time()

        print('time1::', t2 - t1)
        
        
        # print(training_time_prefix, t2 - t1)
        
        # if args.GPU_measure:
            # print_gpu_utilization(GPU_mem_usage_list, None)
        
        stopped_epoch = len(random_ids_multi_super_iterations)
        
        if regularization_prob_samples == 0:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, training_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)
        
        else:
            model, stopped_epoch = select_params_early_stop(args.epochs, random_ids_multi_super_iterations, updated_w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device, no_prov = args.no_prov)

        
        
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=True)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=True)
        
        
        torch.save(updated_w_list, full_out_dir + '/w_list_'+ suffix + '_v' + str(iter_count))
        
        torch.save(updated_grad_list, full_out_dir + '/grad_list_'+ suffix + '_v' + str(iter_count))
        
    
#         updated_origin_grad_list = update_gradient_incremental(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device, exp_updated_origin_grad_list)
    else:
        

        period = args.period
            
        init_epochs = args.init
        
        m = args.hist_size
        
        cached_size = 10000#args.cached_size
        
#         prev_full_existing_labeled_id_tensor = torch.load(full_out_dir + '/prev_labeled_id_tensor')
        
        if check_small_updates_on_labels(ids_with_changed_ids, full_training_noisy_dataset):
    #         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
#             w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count-1))
#     
#             grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count-1))
            
            
            t2 = time.time()
            
            

            grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
            
        #             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
            
            
            
            '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
            
#             exp_updated_w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count))
#         
#             exp_updated_grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count))
            
            t4 = time.time()
            
            set_model_parameters(model, w_list[0], args.device)
            
            updated_model, updated_w_list, updated_grad_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False)
    
            t3 = time.time()
        
        else:
            
            t2 = time.time()
            
            updated_w_list, updated_grad_list,_ = train_model_dataset(model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False)
            
            t3 = time.time()
#             torch.save(full_existing_labeled_id_tensor, full_out_dir + '/prev_labeled_id_tensor')
        
        print('time2::', t3 - t2)
        
        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=False)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=False)


        valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=True)
        
        valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=True)
        
        torch.save(updated_w_list, full_out_dir + '/w_list_'+ suffix + '_v' + str(iter_count) + incremental_suffix)
        
        torch.save(updated_grad_list, full_out_dir + '/grad_list_'+ suffix + '_v' + str(iter_count) + incremental_suffix)
    
    
    
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
    
    
def load_retraining_influence_info(iter_count, full_out_dir, args):
    sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count) + incremental_suffix)
        
    ordered_list = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count) + incremental_suffix)

    influences1 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count) + incremental_suffix)    

    exp_sorted_train_ids = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_v' + str(iter_count))
        
    exp_ordered_list = torch.load(full_out_dir + '/' + args.model + '_influence_removed_ids_weight_v' + str(iter_count))

    influences2 = torch.load(full_out_dir + '/' + args.model + '_influence_v' + str(iter_count))
    
    return influences2,exp_ordered_list,exp_sorted_train_ids,influences1,ordered_list,sorted_train_ids 


def load_retraining_history_info(iter_count, full_out_dir, args):
    full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_' + args.suffix + '_v' + str(iter_count))
    
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + args.suffix + '_v' + str(iter_count))
    
    model = torch.load(full_out_dir + '/model_' + args.suffix + '_v' + str(iter_count), map_location=torch.device('cpu'))

    model = model.to(args.device)

    random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    
    w_list = torch.load(full_out_dir + '/w_list_' + args.suffix + '_v' + str(iter_count))
    
    grad_list = torch.load(full_out_dir + '/grad_list_' + args.suffix + '_v' + str(iter_count))
    
    return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list    


#     print('sub time2::', t3 - t4)
# def load_retraining_history_info(iter_count, full_out_dir, args):
    # full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_v' + str(iter_count))
    #
    # full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(iter_count))
    #
    # model = torch.load(full_out_dir + '/model_v' + str(iter_count))
    #
    # model = model.to(args.device)
    #
    # random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    #
    # w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count))
    #
    # grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count))
    #
    # return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list    

# def load_incremental_history_info(iter_count, full_out_dir, args):
    # full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor_v' + str(iter_count) + incremental_suffix)
    #
    # full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(iter_count) + incremental_suffix)
    #
    # model = torch.load(full_out_dir + '/model_v' + str(iter_count) + incremental_suffix)
    #
    # model = model.to(args.device)
    #
    # random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
    #
    # w_list = torch.load(full_out_dir + '/w_list_v' + str(iter_count) + incremental_suffix)
    #
    # grad_list = torch.load(full_out_dir + '/grad_list_v' + str(iter_count) + incremental_suffix)
    #
    # return full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list


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
        # if iter_count ==0:
            # args.resolve_conflict_str = ''
        # if len(args.resolve_conflict_str) == 0:
            # return load_retraining_history_info0(iter_count, full_out_dir, args)
        # else:
        return load_retraining_history_info(iter_count, full_out_dir, args)
        
        
        
        
    else:
        
        # if len(args.resolve_conflict_str) == 0:
            # return load_incremental_history_info0(iter_count, full_out_dir, args)
        # else:
        return load_incremental_history_info(iter_count, full_out_dir, args)

# def load_history_info(is_incremental, iter_count, full_out_dir, args):
#
    # print('iter count::', iter_count)
    #
    # if iter_count == 0 or (not is_incremental):
    #
# #         else:
# #             iter_count = 0
        #
        # return load_retraining_history_info(iter_count, full_out_dir, args)
        #
        #
        #
        #
    # else:
        # return load_incremental_history_info(iter_count, full_out_dir, args)
        

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
    
    suffix = args.suffix #o2u
    
    method = 'O2U'
    
    regularization_prob_samples = args.regular_rate
    
    
    
    if if_dnn:
        validation_dataset = valid_origin_dataset
        
        dataset_test = test_origin_dataset
        
        full_training_noisy_dataset = train_origin_dataset
    
    
    
    
    if args.start:
        remove_different_version_dataset(full_out_dir, args.model)
        
        random_ids_multi_super_iterations = None
        
#         os.remove(full_out_dir + '/model_initial')
        
        if not args.restart and os.path.exists(full_out_dir + '/model_initial'):
            model = torch.load(full_out_dir + '/model_initial',map_location=torch.device('cpu'))
        
            w_list = torch.load(full_out_dir + '/w_list_initial')
            
            grad_list = torch.load(full_out_dir + '/grad_list_initial')
        
#             if os.path.exists(full_out_dir + '/random_ids_multi_super_iterations'):
            random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
        
        else:
            
            if not args.restart and os.path.exists(full_out_dir + '/random_ids_multi_super_iterations'):
                random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
            
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
        
        # model, stopped_epoch = select_params_early_stop(args.epochs,random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, full_training_noisy_dataset, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
        
        
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

        
        torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        
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
            torch.save(torch.Tensor(), full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v0')
        
        torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + suffix + '_v0')
        
        iter_count = 0
        
        torch.save(iter_count, full_out_dir + '/labeling_iter_' + suffix)
        # remove_different_version_dataset(full_out_dir, args.model)
        #
        # w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
        #
        # '''first iteration of selecting samples by the influence function and labeled by experts::'''
        # t1 = time.time()
# #         most_influence_point, ordered_list0, sorted_train_ids0 = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, 0, model.soft_loss_function, args.model, args.num_class)
        #
        # t2 = time.time()
        #
        # criterion = model.soft_loss_function
# #         influence1, ordered_list, sorted_train_ids = O2U_second_stage(model, optimizer, criterion, args.o2u_epochs, full_training_noisy_dataset, validation_dataset, dataset_test, None, args.GPU, args.device, args.removed_count)
#
        # updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids = select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, optimizer, full_out_dir, is_incremental, 0, None, method)
        #
        #
        # t3 = time.time()
        #
        # print(t2 - t1)
        #
        # print(t3 - t2)
        #
# #         torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_'+method+'_removed_ids_v0')
# #          
# #         torch.save(ordered_list, full_out_dir + '/' + args.model + '_'+method+'_removed_ids_weight_v0')    
#
# #         '''label the selected samples and update the probablistic labels'''
# #         
# #         
# #         
# #         updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=None, size = None, start = True, iter_count = 0, method = method)
        #
        # full_training_noisy_dataset.labels = updated_labels
        #
        #
        # w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
        #
        #
        # valid_model_dataset(model, validation_dataset, None, args.bz, 'validation', args.GPU, args.device, f1=True)
        #
        # valid_model_dataset(model, dataset_test, None, args.bz, 'test', args.GPU, args.device, f1=True)
        #
        #
        # torch.save(model, full_out_dir + '/model_v0')
        #
        # torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        #
        # torch.save(w_list, full_out_dir + '/w_list_v0')
        #
        # torch.save(grad_list, full_out_dir + '/grad_list_v0')
        #
# #         all_entry_grad_list = obtain_gradients_each_class(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, args.GPU, args.device)
# #         
# #         torch.save(all_entry_grad_list, full_out_dir + '/all_entry_grad_list')
        #
        # torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor_v0')
        #
        # torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_v0')
        #
        # iter_count = 0
        #
        # torch.save(iter_count, full_out_dir + '/labeling_iter')
        
    else:
        
#         if continue_labeling:
        
        iter_count = torch.load(full_out_dir + '/labeling_iter_' + suffix)
        # iter_count = 0

        full_existing_labeled_id_tensor, full_training_noisy_dataset, model, random_ids_multi_super_iterations, w_list, grad_list= load_history_info(load_incremental, iter_count, full_out_dir, args)


        optimizer = model.get_optimizer(args.tlr, args.wd)
        
        
        if len(full_existing_labeled_id_tensor) > 0:
            r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)*regularization_prob_samples
    
            r_weight[full_existing_labeled_id_tensor.type(torch.long)] = 1
        else:
            r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)
        
        updated_labels, full_existing_labeled_id_tensor, ids_with_changed_ids, ids_with_unchanged_ids = select_samples_influence_function_main(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, optimizer, full_out_dir, is_incremental, iter_count + 1, full_existing_labeled_id_tensor, suffix = suffix, train_annotated_label_tensor = train_annotated_label_tensor, r_weight = r_weight)
        
        r_weight = torch.ones(full_training_noisy_dataset.data.shape[0], dtype = full_training_noisy_dataset.data.dtype)*regularization_prob_samples
    
        r_weight[full_existing_labeled_id_tensor.type(torch.long)] = 1
#         if continue_labeling:

        # model_training_main(model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count + 1, ids_with_changed_ids, ids_with_unchanged_ids, suffix = suffix)
        # '''model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count, ids_with_changed_ids, ids_with_unchanged_ids, regularization_prob_samples = 0.1, suffix = 'o2u''''
        model_training_main(full_out_dir, model, optimizer, random_ids_multi_super_iterations, w_list, grad_list, full_training_noisy_dataset, updated_labels, args, validation_dataset, dataset_test, is_incremental, iter_count + 1, ids_with_changed_ids, ids_with_unchanged_ids, full_existing_labeled_id_tensor, regularization_prob_samples = regularization_prob_samples, r_weight = r_weight, suffix = suffix)

        
        full_training_noisy_dataset.labels = updated_labels
        
        if not is_incremental:
            torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + suffix + '_v' + str(iter_count+1))
            
            torch.save(model, full_out_dir + '/model_' + suffix + '_v' + str(iter_count+1))
            
            torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v' + str(iter_count+1))
            
            torch.save(ids_with_changed_ids, full_out_dir + '/ids_with_changed_ids_' + suffix + '_v' + str(iter_count + 1))
            
            torch.save(ids_with_unchanged_ids, full_out_dir + '/ids_with_unchanged_ids_' + suffix + '_v' + str(iter_count + 1))
        
        else:
            torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_' + suffix + '_v' + str(iter_count+1) + incremental_suffix)
            
            torch.save(model, full_out_dir + '/model_' + suffix + '_v' + str(iter_count+1) + incremental_suffix)
            
            torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor_' + suffix + '_v' + str(iter_count+1) + incremental_suffix)
            
            torch.save(ids_with_changed_ids, full_out_dir + '/ids_with_changed_ids_' + suffix + '_v' + str(iter_count + 1) + incremental_suffix)
            
            torch.save(ids_with_unchanged_ids, full_out_dir + '/ids_with_unchanged_ids_' + suffix + '_v' + str(iter_count + 1) + incremental_suffix)
            
        if continue_labeling:
        
            torch.save(iter_count + 1, full_out_dir + '/labeling_iter_' + suffix)
            
#         all_entry_grad_list = torch.load(full_out_dir + '/all_entry_grad_list')
        
#         all_entry_grad_list_new = transform_entry_grad(all_entry_grad_list, full_training_noisy_dataset.data.shape[0], args.bz, args.epochs)
#     
#         print(all_entry_grad_list_new.shape)
#     
#         all_entry_grad_list = all_entry_grad_list_new
    
        
    
    