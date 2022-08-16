'''
Created on Jan 10, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob
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

#     full_existing_labeled_id_tensor = torch.load(full_out_dir + '/' + args.model + 'full_existing_labeled_id_tensor')
#     
#     full_training_noisy_dataset.labels[full_existing_labeled_id_tensor] = onehot(full_training_origin_labels[full_existing_labeled_id_tensor], args.num_class).type(full_training_noisy_dataset.labels.dtype)

    
#     all_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list')
#         
#     all_para_list = torch.load(full_out_dir + '/all_sample_wise_para_list')
    
    remaining_sample_ids = torch.load(full_out_dir + '/remaining_sample_ids')#[0:100]

    iter_id = 3
    m = 2
    s_k_list, y_k_list, mini_Y_k_list, mini_Y_k_list2, full_grad_list = obtain_s_y_list(args, torch.tensor(list(range(remaining_sample_ids.shape[0]))), full_out_dir, m = m)
    
    
    y_k_list_tensor = torch.stack(y_k_list, dim = 0)
    
    print(y_k_list_tensor.shape)
    
    curr_w_list = torch.load(full_out_dir + '/w_list_' + str(iter_id))
    
    curr_grad_list = torch.load(full_out_dir + '/grad_list_' + str(iter_id))
    
    prev_w_list = torch.load(full_out_dir + '/w_list_' + str(0))
    
    prev_grad_list = torch.load(full_out_dir + '/grad_list_' + str(0))
    
    
    model = torch.load(full_out_dir + '/model_' + str(iter_id), map_location=torch.device('cpu'))
    
    set_model_parameters(model, curr_w_list[-1], args.device)
    
#     t1 = time.time()
#           
#     influences1, ordered_list1, sorted_train_ids1 = evaluate_influence_function_repetitive2(args, full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, model, args.removed_count, model.soft_loss_function, args.model, args.num_class)
#       
#     t2 = time.time()
    
    
#     curr_full_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(iter_id))
#     
#     prev_full_grad_list = torch.load(full_out_dir + '/all_sample_wise_grad_list_' + str(0))
    
#     curr_para = get_all_vectorized_parameters1(torch.load(full_out_dir + '/w_list_' + str(i))[-1])
    full_training_noisy_dataset = torch.load(full_out_dir + '/full_training_noisy_dataset_' + str(iter_id))
    
    set_model_parameters(model, curr_w_list[-1], args.device)
    
    optimizer = model.get_optimizer(args.tlr, args.wd)
    
    loss_func = model.soft_loss_function_reduce
    
    t1 = time.time()
    
    influences2, full_grad_tensors, ordered_list, sorted_train_ids = incremental_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, s_k_list, y_k_list_tensor, full_grad_list, curr_w_list, prev_w_list, remaining_sample_ids, args.GPU, args.device, args.wd,m, args.bz)
    
    t2 = time.time()
    
    influences1, full_grad_tensors1, ordered_list1, sorted_train_ids1 = origin_compute_sample_wise_gradients(model, optimizer, loss_func, full_out_dir, full_training_noisy_dataset, validation_dataset, dataset_test, s_k_list, y_k_list_tensor, full_grad_list, curr_w_list, prev_w_list, remaining_sample_ids, args.GPU, args.device, args.wd,m, args.bz)
    
    t3 = time.time()
    
    print('here:')

    checked_count = 5

    print('incremental time::', t2 - t1)
    
    print('origin compute time::', t3 - t2)
    
    print('influence difference::', torch.max(torch.abs(influences1[0:checked_count] - influences2[0:checked_count])))
    
    print('intersected selected samples::', len(set(sorted_train_ids[0:checked_count].view(-1).tolist()).intersection(set(sorted_train_ids1[0:checked_count].view(-1).tolist())))*1.0/checked_count)






