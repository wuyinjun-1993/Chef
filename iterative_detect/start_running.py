'''
Created on Jan 7, 2021

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



def transform_entry_grad(all_entry_grad_list, size, batch_size, epoch):
    count = math.ceil(size*1.0/batch_size)
    
    final_entry_grad_list_tensor = []
    
    for i in range(epoch):
        curr_entry_grad_list = []
        for j in range(count):
            curr_entry_grad_list.append(all_entry_grad_list[i*count + j])
            
        curr_entry_grad_list_tensor = torch.cat(curr_entry_grad_list, dim = 0)
        
        final_entry_grad_list_tensor.append(curr_entry_grad_list_tensor)
    
    
    return torch.stack(final_entry_grad_list_tensor, dim = 0)
    


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
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, small_dataset, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)
    
    size = None
    
    if args.start:
        
        updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=None, size = None, start = True)
        
        full_training_noisy_dataset.labels = updated_labels
        
        w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
    
        torch.save(model, full_out_dir + '/model')
        
        torch.save(random_ids_multi_super_iterations, full_out_dir + '/random_ids_multi_super_iterations')
        
        torch.save(w_list, full_out_dir + '/w_list')
        
        torch.save(grad_list, full_out_dir + '/grad_list')
        
#         all_entry_grad_list = obtain_gradients_each_class(w_list, grad_list, model, random_ids_multi_super_iterations, full_training_noisy_dataset, args.bz, args.num_class, optimizer, args.GPU, args.device)
#         
#         torch.save(all_entry_grad_list, full_out_dir + '/all_entry_grad_list')
        
        torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor')
        
        remove_different_version_dataset(full_out_dir)
        
        torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_v0')
        
        iter_count = 0
        
        torch.save(iter_count, full_out_dir + '/labeling_iter')
        
        most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
        
        torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids')
    
        torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight')    
    
    else:
        iter_count = torch.load(full_out_dir + '/labeling_iter')
        
        print('iter count::', iter_count)
        
        full_existing_labeled_id_tensor = torch.load(full_out_dir + '/full_existing_labeled_id_tensor')
    
        full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_v' + str(iter_count))
        
        model = torch.load(full_out_dir + '/model')

        model = model.to(args.device)

        optimizer = model.get_optimizer(args.tlr, args.wd)
        
        random_ids_multi_super_iterations = torch.load(full_out_dir + '/random_ids_multi_super_iterations')
        
        w_list = torch.load(full_out_dir + '/w_list')
        
        grad_list = torch.load(full_out_dir + '/grad_list')
        
    
        updated_labels, full_existing_labeled_id_tensor = obtain_updated_labels(full_out_dir, args, full_training_noisy_dataset.data, full_training_noisy_dataset.labels, full_training_origin_labels, existing_labeled_id_tensor=full_existing_labeled_id_tensor, size = None, start = False)
    
        ids_with_changed_ids, ids_with_unchanged_ids = sample_ids_with_changed_labels(updated_labels, full_training_noisy_dataset.labels)
    
#         torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor')

#         all_entry_grad_list = torch.load(full_out_dir + '/all_entry_grad_list')
        
#         all_entry_grad_list_new = transform_entry_grad(all_entry_grad_list, full_training_noisy_dataset.data.shape[0], args.bz, args.epochs)
#     
#         print(all_entry_grad_list_new.shape)
#     
#         all_entry_grad_list = all_entry_grad_list_new
    
        exp_updated_origin_grad_list = None 
    
#         exp_updated_origin_grad_list = update_gradient_origin(all_entry_grad_list, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.GPU, args.device, w_list, model, optimizer)
    
        criterion = model.soft_loss_function_reduce
        
        set_model_parameters(model, w_list[0], args.device)
        
        t1 = time.time()
        
        exp_updated_w_list, exp_updated_grad_list,_ = train_model_dataset(model, optimizer, random_ids_multi_super_iterations, full_training_noisy_dataset.data, updated_labels, args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = validation_dataset, test_dataset = dataset_test, f1 = False, capture_prov = True, is_early_stopping=False, test_performance = False)
    
        final_exp_updated_model_param = get_all_vectorized_parameters1(model.parameters())
    
        t2 = time.time()
    
    
#         updated_origin_grad_list = update_gradient_incremental(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device, exp_updated_origin_grad_list)
    
    
        period = 5#args.period
            
        init_epochs = 10#args.init
        
        m = 2#args.m
        
        cached_size = 10000#args.cached_size
        
        
#         updated_grad_list = update_gradient(all_entry_grad_list, random_ids_multi_super_iterations, updated_labels, args.bz, args.GPU, args.device)
        
        grad_list_all_epochs_tensor, updated_grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, updated_grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(full_out_dir, cached_size, args.GPU, args.device, w_list, grad_list)
        
#             model_update_provenance_test3(period, 1, init_epochs, dataset_train, model, grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, max_epoch, 2, learning_rate_all_epochs, random_ids_multi_epochs, sorted_ids_multi_epochs, batch_size, dim, added_random_ids_multi_super_iteration, X_to_add, Y_to_add, criterion, optimizer, lr_scheduler, regularization_coeff, is_GPU, device)
        
        
        
        '''max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor, para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, all_entry_grad_list, m, learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device'''
        
        t4 = time.time()
        
        set_model_parameters(model, w_list[0], args.device)
        
        updated_model, updated_w_list = model_update_deltagrad2(args.epochs, period, 1, init_epochs, full_training_noisy_dataset, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, updated_labels, ids_with_changed_ids, ids_with_unchanged_ids, m, args.tlr, random_ids_multi_super_iterations, args.bz, criterion, optimizer, args.wd, args.GPU, args.device, exp_updated_w_list, exp_updated_grad_list, compare = False)

        t3 = time.time()
        
        if exp_updated_w_list is not None and len(exp_updated_w_list) > 0:
            print('model para diff::',torch.norm(get_all_vectorized_parameters1(updated_w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))
            print('model para change::',torch.norm(get_all_vectorized_parameters1(w_list[-1]) - get_all_vectorized_parameters1(exp_updated_w_list[-1])))
        print('time1::', t2 - t1)
        
        print('time2::', t3 - t2)

        print('sub time2::', t3 - t4)
        
        print('sample count with changed labels::', torch.sum(ids_with_changed_ids))

#         most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
#         
#         torch.save(sorted_train_ids, full_out_dir + '/' + args.model + '_influence_removed_ids')
#     
#         torch.save(ordered_list, full_out_dir + '/' + args.model + '_influence_removed_ids_weight')
#     
#         full_training_noisy_dataset.labels = updated_labels
#     
#         torch.save(full_existing_labeled_id_tensor, full_out_dir + '/full_existing_labeled_id_tensor')
#         iter_count += 1
#     
#         torch.save(iter_count, full_out_dir + '/labeling_iter')
#     
#         torch.save(full_training_noisy_dataset, full_out_dir + '/full_training_noisy_dataset_v' + str(iter_count))
    