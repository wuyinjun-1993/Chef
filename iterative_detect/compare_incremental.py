'''
Created on Jan 14, 2021

'''
import os, sys

import torch
import torch.functional as F
from collections import deque  
import os, glob
from iterative_detect.utils_iters import incremental_suffix
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
from full_pipeline import *


label_change_threshold = 1e-5


def compare_model_para_list(w_list1, w_list2):
    for k in range(len(w_list1)):
        w1 = get_all_vectorized_parameters1(w_list1[k])
        w2 = get_all_vectorized_parameters1(w_list2[k])
        
        print('diff::', k, torch.norm(w2 - w1))

def compare_model_grad_list(grad_list1, grad_list2):
    for k in range(len(grad_list1)):
        grad1 = grad_list1[k]
        grad2 = grad_list2[k]
        print('diff::', k, torch.norm(grad1 - grad2))


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
    
    picked_num = 10
     
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, small_dataset, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids = obtain_data_function(args, noisy=True)

    iter_count = 1

    full_existing_labeled_id_tensor1, _, model1, _, w_list1, grad_list1 = load_retraining_history_info(iter_count, full_out_dir, args)
    
    full_existing_labeled_id_tensor2, _, model2, _, w_list2, grad_list2 = load_incremental_history_info(iter_count, full_out_dir, args)
    
    full_existing_labeled_id_tensor0, _, model0, _, w_list0, grad_list0 = load_retraining_history_info(iter_count-1, full_out_dir, args)
    
    influences2,exp_ordered_list,exp_sorted_train_ids,influences1,ordered_list,sorted_train_ids = load_retraining_influence_info(iter_count, full_out_dir, args)
    
    
    print('model grad diff::')
    
    compare_model_para_list(grad_list1, grad_list2)
    
    print('model para diff::')
    
    compare_model_para_list(w_list1, w_list2)
    
    
    
    print('model para changes::')
    
    compare_model_para_list(w_list1, w_list0)

    print('current influence values::', influences1[0:picked_num])
    
    print('expected influence values::', influences2[0:picked_num])
    
    print('intersected picked samples::', picked_num, len(set(sorted_train_ids[0:picked_num].tolist()).intersect(set(exp_sorted_train_ids[0:picked_num].tolist()))))
    
    
    
