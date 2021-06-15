'''
Created on Dec 20, 2020

'''

import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import copy
import torch.nn.functional as F
import copy
from torch import autograd
import higher
import itertools
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import warnings
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/real_examples')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/pytorch_influence_functions')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/goggles')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/iterative_detect')

from sklearn.metrics import roc_auc_score
from sklearn import metrics

from torch.utils.data import Dataset, DataLoader

import models
# from goggles import GogglesDataset
# from goggles import construct_image_affinity_matrices
# import goggles
#
# import active_learning_query

try:
    from models.util_func import *
    from utils.utils import *
    
    from Reweight_examples.utils_reweight import *
    from iterative_detect.utils_iters import *
    from real_examples.utils_real import *
    from iterative_detect.utils_iters import *
except ImportError:
    from util_func import *
    
    from utils import *
    
    from utils_reweight import *
    from utils_iters import *
    from utils_real import *
    from utils_iters import *

def generate_random_noisy_labels4(dataset_train, args, ratio, soft_label, full_output_dir = None, validation_dataset=None, dataset_test=None):
    
    
    
    
    perturb_bound = 0.4
     
    clean_dataset_train_ids = torch.nonzero(torch.logical_or(dataset_train.labels == 1,dataset_train.labels == 0)).view(-1)
    
    clean_dataset_train_data = dataset_train.data[clean_dataset_train_ids]
    
    clean_dataset_train_labels = dataset_train.labels[clean_dataset_train_ids]
    
    clean_dataset = models.MyDataset(clean_dataset_train_data, clean_dataset_train_labels)
    
    clean_dataset2 = models.MyDataset(clean_dataset_train_data, onehot(clean_dataset_train_labels, args.num_class))
    
    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(clean_dataset2, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
#     model = do_training_general(args, clean_dataset, 'Logistic_regression', args.num_class, None, False, clean_dataset, clean_dataset)
    
    # most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, clean_dataset, clean_dataset.labels, validation_dataset, dataset_test, model, 0, model.soft_loss_function, args.model, args.num_class)
    
    r_weight = torch.ones(clean_dataset.data.shape[0], dtype = torch.double)
    
    influences1, full_grad_tensors, ordered_list, sorted_train_ids, s_test_vec = origin_compute_sample_wise_gradients(args, model, optimizer, model.soft_loss_function_reduce, full_output_dir, clean_dataset2, validation_dataset, dataset_test, None, grad_list, args.GPU, args.device, args.wd,args.bz, clean_dataset2.data.shape[0], clean_dataset2, regular_rate = 1, r_weight = r_weight)
    
    
    uncert_sampling = active_learning_query.UncertaintySampling(True)
    
    method = 'least_confidence'
    
    active_leaning_method_func = getattr(uncert_sampling, method)
    
    # prob_dist_list_tensor, influences1, ordered_list, sorted_train_ids = uncert_sampling.get_samples_batch(model, clean_dataset_train_data, active_leaning_method_func, number=args.removed_count, batch_size = args.bz, is_GPU = args.GPU, device = args.device)
    print(clean_dataset.data.shape, clean_dataset.labels.shape)
    
    noisy_label_count = int(clean_dataset.data.shape[0]*ratio)
    
#     random_sample_ids = torch.randperm(clean_dataset.data.shape[0]).view(-1)
    
    
#     sorted_train_ids = torch.randperm(clean_dataset.data.shape[0])
      
    # noisy_sample_ids = sorted_train_ids[clean_dataset.data.shape[0] - noisy_label_count:]
    #
    # remaining_sample_ids = sorted_train_ids[0:clean_dataset.data.shape[0] - noisy_label_count]
    
    noisy_sample_ids = sorted_train_ids[0:noisy_label_count]
       
    remaining_sample_ids = sorted_train_ids[noisy_label_count:]
    
    
    
    
    updated_labels = clean_dataset.labels[noisy_sample_ids].type(torch.double) 
    
#     updated_labels = torch.rand(noisy_sample_ids.shape[0], dtype = torch.double)
    
#     updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] - torch.rand(torch.sum(clean_dataset.labels[noisy_sample_ids] > 0.5), dtype = torch.double)*perturb_bound
#         
#     updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] + torch.rand(torch.sum(clean_dataset.labels[noisy_sample_ids] <= 0.5), dtype = torch.double)*perturb_bound
    
#     updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] - perturb_bound
#        
#     updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] + perturb_bound
       
    updated_labels = torch.clamp(updated_labels, min = 0, max = 1)
    
    updated_label_tensor = torch.zeros([clean_dataset.labels.shape[0], 2],dtype = torch.double)
    
    updated_label_tensor[noisy_sample_ids,0] = updated_labels
    
    updated_label_tensor[noisy_sample_ids,1] = 1-updated_labels
    
    updated_label_tensor[remaining_sample_ids] = onehot(clean_dataset.labels[remaining_sample_ids], args.num_class).type(torch.double)
    
    full_noisy_dataset = models.MyDataset(clean_dataset.data.clone(), updated_label_tensor)
    print(noisy_sample_ids)
    torch.save(noisy_sample_ids, full_output_dir + '/noisy_sample_ids')
    torch.save(remaining_sample_ids, full_output_dir + '/clean_sample_ids')
    return full_noisy_dataset, clean_dataset_train_labels.clone(), None, None, torch.tensor(list(range(updated_labels.shape[0])))

# 
def generate_random_noisy_labels3(dataset_train, args, ratio, soft_label, full_output_dir = None, validation_dataset=None, dataset_test=None):
    
    
    
    
    perturb_bound = 1
     
    clean_dataset_train_ids = torch.nonzero(torch.logical_or(dataset_train.labels == 1,dataset_train.labels == 0)).view(-1)
    
    clean_dataset_train_data = dataset_train.data[clean_dataset_train_ids]
    
    clean_dataset_train_labels = dataset_train.labels[clean_dataset_train_ids]
    
    print(clean_dataset_train_data.shape, clean_dataset_train_labels.shape, torch.unique(clean_dataset_train_labels))
    
    clean_dataset = models.MyDataset(clean_dataset_train_data, clean_dataset_train_labels)
    
    
    
    clean_dataset2 = models.MyDataset(clean_dataset_train_data, onehot(clean_dataset_train_labels, args.num_class))
    
    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(clean_dataset2, validation_dataset, dataset_test, args, binary=False, is_early_stopping = False)
    
    valid_model_dataset(model, validation_dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
         
    valid_model_dataset(model, dataset_test, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)
    
    model, stopped_epoch = select_params_early_stop(args.epochs,random_ids_multi_super_iterations, w_list, get_model_para_list(model), model, clean_dataset2, validation_dataset, dataset_test, args.bz, args.GPU, args.device)
    
#     model = do_training_general(args, clean_dataset, 'Logistic_regression', args.num_class, None, False, clean_dataset, clean_dataset)
    
#     most_influence_point, ordered_list, sorted_train_ids = evaluate_influence_function_repetitive2(args, clean_dataset, clean_dataset.labels, validation_dataset, dataset_test, model, 0, model.soft_loss_function, args.model, args.num_class)
    uncert_sampling = active_learning_query.UncertaintySampling(True)
    
    method = 'least_confidence'
    
    # active_leaning_method_func = getattr(uncert_sampling, method)
    #
    # prob_dist_list_tensor, influences1, ordered_list, sorted_train_ids = uncert_sampling.get_samples_batch(model, clean_dataset_train_data, active_leaning_method_func, number=args.removed_count, batch_size = args.bz, is_GPU = args.GPU, device = args.device)
    # print(clean_dataset.data.shape, clean_dataset.labels.shape)
    
    noisy_label_count = int(clean_dataset.data.shape[0]*ratio)
    
#     random_sample_ids = torch.randperm(clean_dataset.data.shape[0]).view(-1)
    
    
    sorted_train_ids = torch.randperm(clean_dataset.data.shape[0])
      
    noisy_sample_ids = sorted_train_ids[clean_dataset.data.shape[0] - noisy_label_count:]
       
    remaining_sample_ids = sorted_train_ids[0:clean_dataset.data.shape[0] - noisy_label_count]
    
#     noisy_sample_ids = sorted_train_ids[0:noisy_label_count]
#       
#     remaining_sample_ids = sorted_train_ids[noisy_label_count:]
    
    
    
    
    updated_labels = clean_dataset.labels[noisy_sample_ids].type(torch.double) 
    
    updated_labels = torch.rand([noisy_sample_ids.shape[0], args.num_class], dtype = torch.double)
    
    updated_labels = updated_labels/torch.sum(updated_labels, 1).view(-1,1)
    updated_labels2 = torch.softmax(updated_labels, dim = 1)
    
#     updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] - torch.rand(torch.sum(clean_dataset.labels[noisy_sample_ids] > 0.5), dtype = torch.double)*perturb_bound
#          
#     updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] + torch.rand(torch.sum(clean_dataset.labels[noisy_sample_ids] <= 0.5), dtype = torch.double)*perturb_bound
    
#     updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] > 0.5] - perturb_bound
#        
#     updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] = updated_labels[clean_dataset.labels[noisy_sample_ids] <= 0.5] + perturb_bound
       
#     updated_labels = torch.clamp(updated_labels, min = 0, max = 1)
    
    updated_label_tensor = torch.zeros([clean_dataset.labels.shape[0], args.num_class],dtype = torch.double)
    
    updated_label_tensor[noisy_sample_ids] = updated_labels
#     
#     updated_label_tensor[noisy_sample_ids,1] = 1-updated_labels
    
    updated_label_tensor[remaining_sample_ids] = onehot(clean_dataset.labels[remaining_sample_ids], args.num_class).type(torch.double)
    
    full_noisy_dataset = models.MyDataset(clean_dataset.data.clone(), updated_label_tensor)
    print(noisy_sample_ids)
    torch.save(noisy_sample_ids, full_output_dir + '/noisy_sample_ids')
    torch.save(remaining_sample_ids, full_output_dir + '/clean_sample_ids')
    return full_noisy_dataset, clean_dataset_train_labels.clone(), None, None, torch.tensor(list(range(updated_labels.shape[0])))
#     updated_labels = torch.
    
        



def generate_random_noisy_labels2(dataset_train, args, ratio, soft_label, full_output_dir = None):
    
#     if args.small_test:
#         dataset_train.data = dataset_train.data[0:100]
#               
#         dataset_train.labels = dataset_train.labels[0:100]
    
    small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, selected_small_sample_ids, selected_noisy_sample_ids = partition_hard_labeled_data_noisy_data3(dataset_train, ratio, args)
    
    print('small dataset count::', small_dataset_train.labels.shape[0], torch.unique(small_dataset_train.labels))
    
#     if args.small_test:
#     
#         selected_small_sample_ids = torch.cat([selected_small_sample_ids, torch.tensor([0])], dim = 0)
#            
#         small_dataset_train.data = torch.cat([small_dataset_train.data, dataset_train.data[0:1]], dim = 0)
#         small_dataset_train.labels = torch.cat([small_dataset_train.labels, dataset_train.labels[0:1]], dim = 0)
    
    AFs = []
    if os.path.exists(full_output_dir + '/AFs'):
        AF_tensor = torch.load(full_output_dir + '/AFs')
        for i in range(AF_tensor.shape[0]):
            AFs.append(AF_tensor[i])
        remaining_AFs = []
        for AF in AFs:
            print(AF)
            curr_remaining_AF_tensor = torch.from_numpy(AF[selected_noisy_sample_ids.numpy()][:, selected_noisy_sample_ids.numpy()])
            remaining_AFs.append(curr_remaining_AF_tensor)
        full_remaining_AFs = torch.stack(remaining_AFs, dim = 0)
        
        torch.save(full_remaining_AFs, full_output_dir + '/remaining_AFs')
        
    else:
        size = 1
            
        for k in range(len(dataset_train.data.shape)-1):
            size *= dataset_train.data.shape[k+1]
        
        model = models.Logistic_regression(size, args.num_class, bias = True)
        AFs = construct_image_affinity_matrices(dataset_train.data.type(torch.DoubleTensor), args.GPU, args.device,model = model,output_path = full_output_dir)
        
        print('AF shape::', len(AFs), AFs[0].shape)
        
        remaining_AFs = []
        
#         AF_tensor = torch.stack(AFs, dim = 0)
#         
#         torch.save(AF_tensor, full_output_dir + '/AFs')
    
        AF_tensor_list = []
        
        for AF in AFs:
            print(AF)
            AF_tensor_list.append(torch.from_numpy(AF))
            
            curr_remaining_AF_tensor = torch.from_numpy(AF[selected_noisy_sample_ids.numpy()][:, selected_noisy_sample_ids.numpy()])
            
            remaining_AFs.append(curr_remaining_AF_tensor)
        
        AF_tensor = torch.stack(AF_tensor_list, dim = 0)
        
        full_remaining_AFs = torch.stack(remaining_AFs, dim = 0)
        
        torch.save(AF_tensor, full_output_dir + '/AFs')
        
        torch.save(full_remaining_AFs, full_output_dir + '/remaining_AFs')
#     
#     print('construct AF done')
    
#     if len(small_dataset_train.data.shape) > 2:
#         
#         small_dataset_train_data = small_dataset_train.data.view(-1, size).tolist()
#     
#     else:
#         small_dataset_train_data = small_dataset_train.data.tolist()
    
    prob = goggles.infer_labels(AFs,selected_small_sample_ids.view(-1).tolist(),small_dataset_train.labels.view(-1).tolist())
    
    print('probabilistic label shape::', prob.shape)
    
    full_noisy_dataset = models.MyDataset(dataset_train.data[selected_noisy_sample_ids], torch.tensor(prob[selected_noisy_sample_ids], dtype = torch.double))
    
    full_origin_labels = dataset_train.labels[selected_noisy_sample_ids]
    
    return full_noisy_dataset, full_origin_labels, small_dataset_train, selected_small_sample_ids, selected_noisy_sample_ids
    
#     print(prob.shape)
#     
#     print(prob)
#     
# #     labels = 
#     
#     
#     AFs = construct_image_affinity_matrices(dataset_train.data.view(dataset_train.data.shape[0], -1),cache=True, model = model)
#     
#     
#     
# #     hg = program_synthesis.HeuristicGenerator(selected_noisy_samples.view(selected_noisy_samples.shape[0], -1).numpy(), small_dataset_train.data.view(small_dataset_train.data.shape[0], -1).numpy(), small_dataset_train.labels.numpy(), train_ground=selected_noisy_origin_labels.numpy(), b=0.5)
#     
#     idx = None
#     
#     validation_accuracy = []
#     training_accuracy = []
#     validation_coverage = []
#     training_coverage = []
#     
#     training_marginals = []
#     
#     for i in range(3,26):
#         if (i-2)%5 == 0:
#             print("Running iteration: ", str(i-2))
#             
#         #Repeat synthesize-prune-verify at each iterations
#         if i == 3:
#             hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
#         else:
#             hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
#         hg.run_verifier()
#         
#         #Save evaluation metrics
#         va,ta, vc, tc = hg.evaluate()
#         validation_accuracy.append(va)
#         training_accuracy.append(ta)
#         training_marginals.append(hg.vf.train_marginals)
#         validation_coverage.append(vc)
#         training_coverage.append(tc)
#         
#         #Plot Training Set Label Distribution
# #         if i <= 8:
# #             plt.subplot(2,3,i-2)
# #             plt.hist(training_marginals[-1], bins=10, range=(0.0,1.0)); 
# #             plt.title('Iteration ' + str(i-2));
# #             plt.xlim([0.0,1.0])
# #             plt.ylim([0,825])
#         
#         #Find low confidence datapoints in the labeled set
#         hg.find_feedback()
#         idx = hg.feedback_idx
#         
#         #Stop the iterative process when no low confidence labels
#         if idx == []:
#             break
    

def label_remaining_dataset(args, small_dataset_train, soft_label, selected_noisy_samples, selected_noisy_origin_labels):
    #args, train_dataset, model_name, num_class=1, r_weight = None, soft = True
    model = do_training_general(args, small_dataset_train, 'Logistic_regression', num_class = args.num_class ,soft= False, valid_dataset = small_dataset_train, test_dataset = small_dataset_train)
    
    
    evaluate_model_test_dataset(small_dataset_train, model, args, tag = 'training')
    
    selected_noisy_labels = labeling_noisy_samples(model, selected_noisy_samples, args, args.bz, soft = soft_label)
    
#     full_noisy_samples = torch.cat([small_dataset_train.data, selected_noisy_samples], dim = 0)
#     
#     full_noisy_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_labels.view(-1, small_dataset_train.labels.shape[1])], dim = 0)
#     
#     full_origin_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])], dim = 0)
    
#     meta_count = int(selected_noisy_samples.shape[0]*0.1)
#     
#     rand_ids = torch.randperm(selected_noisy_samples.shape[0])
#     
#     meta_sample_ids = rand_ids[0:meta_count]
#     
#     remaining_ids = rand_ids[meta_count:]
#     
#     
#     valid_dataset = models.MyDataset(selected_noisy_samples[meta_sample_ids], selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])[meta_sample_ids])
    
    final_selected_noisy_samples = selected_noisy_samples
    
    final_selected_noisy_labels = selected_noisy_labels.type(torch.DoubleTensor).view(-1, args.num_class)
    
    final_selected_noisy_origin_labels = selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1)
    
    
    full_noisy_samples = torch.cat([final_selected_noisy_samples], dim = 0)
    
    full_noisy_labels = torch.cat([final_selected_noisy_labels], dim = 0)
    
    full_origin_labels = torch.cat([final_selected_noisy_origin_labels], dim = 0)
    
    
    full_noisy_dataset = models.MyDataset(full_noisy_samples, full_noisy_labels)
    
    
    
    
    

    
    
    
#     full_training_noisy_dataset = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
#     
#     
#     
#     full_training_origin_labels = full_training_origin_labels[remaining_ids]
    
    
    
    return full_noisy_dataset, full_origin_labels, small_dataset_train


def label_remaining_dataset2(args, small_dataset_train, soft_label, selected_noisy_samples, selected_noisy_origin_labels, final_labeled_id_tensor, final_unlabeled_id_tensor, LPs = None, num_class = 2):
    #args, train_dataset, model_name, num_class=1, r_weight = None, soft = True
#     if LPs is None:
#         LPs, num_class = goggles.compute_LPs(LPs, final_labeled_id_tensor.view(-1).tolist(),small_dataset_train.labels.view(-1).tolist(),evaluate=True)
        
        
    prob = goggles.get_estimate_probs(LPs, num_class,final_labeled_id_tensor.view(-1).tolist(),small_dataset_train.labels.view(-1).tolist(),evaluate=True)
    
#     prob = goggles.infer_labels(remaining_AFs,final_labeled_id_tensor.view(-1).tolist(),small_dataset_train.labels.view(-1).tolist())
    
    full_noisy_dataset = models.MyDataset(selected_noisy_samples, torch.tensor(prob, dtype = torch.double)[final_unlabeled_id_tensor])
    
    full_origin_labels = selected_noisy_origin_labels
    
    
    return full_noisy_dataset, full_origin_labels, small_dataset_train


def generate_random_noisy_labels1(dataset_train, args, ratio, soft_label, full_output_dir  =None):
    #small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, selected_small_sample_ids, selected_noisy_sample_ids
    small_dataset_train, selected_noisy_samples, selected_noisy_origin_labels, _, _ = partition_hard_labeled_data_noisy_data3(dataset_train, ratio, args)
    
    #args, train_dataset, model_name, num_class=1, r_weight = None, soft = True
    model = do_training_general(args, small_dataset_train, 'Logistic_regression', num_class = args.num_class ,soft= False, valid_dataset = small_dataset_train, test_dataset = small_dataset_train)
    
    
    evaluate_model_test_dataset(small_dataset_train, model, args, tag = 'training')
    
    selected_noisy_labels = labeling_noisy_samples(model, selected_noisy_samples, args, args.bz, soft = soft_label)
    
#     full_noisy_samples = torch.cat([small_dataset_train.data, selected_noisy_samples], dim = 0)
#     
#     full_noisy_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_labels.view(-1, small_dataset_train.labels.shape[1])], dim = 0)
#     
#     full_origin_labels = torch.cat([small_dataset_train.labels.type(torch.DoubleTensor), selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])], dim = 0)
    
#     meta_count = int(selected_noisy_samples.shape[0]*0.1)
#     
#     rand_ids = torch.randperm(selected_noisy_samples.shape[0])
#     
#     meta_sample_ids = rand_ids[0:meta_count]
#     
#     remaining_ids = rand_ids[meta_count:]
#     
#     
#     valid_dataset = models.MyDataset(selected_noisy_samples[meta_sample_ids], selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1, small_dataset_train.labels.shape[1])[meta_sample_ids])
    
    final_selected_noisy_samples = selected_noisy_samples
    
    final_selected_noisy_labels = selected_noisy_labels.type(torch.DoubleTensor).view(-1, args.num_class)
    
    final_selected_noisy_origin_labels = selected_noisy_origin_labels.type(torch.DoubleTensor).view(-1)
    
    
    full_noisy_samples = torch.cat([final_selected_noisy_samples], dim = 0)
    
    full_noisy_labels = torch.cat([final_selected_noisy_labels], dim = 0)
    
    full_origin_labels = torch.cat([final_selected_noisy_origin_labels], dim = 0)
    
    
    full_noisy_dataset = models.MyDataset(full_noisy_samples, full_noisy_labels)
    
    
    
    
    

    
    
    
#     full_training_noisy_dataset = models.MyDataset(full_training_noisy_dataset.data[remaining_ids], full_training_noisy_dataset.labels[remaining_ids])
#     
#     
#     
#     full_training_origin_labels = full_training_origin_labels[remaining_ids]
    
    
    
    return full_noisy_dataset, full_origin_labels, small_dataset_train

def main(args):
    
#     train_DL, valid_DL, test_DL, full_output_dir, binary = obtain_mimic_examples_origin(args, origin=True)
    
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples_origin')
#     training_dataset, val_dataset, test_dataset, full_output_dir = obtain_chexpert_examples(args)
    train_DL, valid_DL, test_DL, full_output_dir, binary = obtain_data_function(args, origin=True)
    
    
    print('train data shape::', train_DL.dataset.data.shape)
    
    print(torch.unique(train_DL.dataset.labels))
    
    train_DL.dataset.lenth = train_DL.dataset.data.shape[0]
    
    print('valid data shape::', valid_DL.dataset.data.shape)
    
    print('test data shape::', test_DL.dataset.data.shape)
    
    print(torch.unique(valid_DL.dataset.labels))
    
    valid_DL.dataset.lenth = valid_DL.dataset.data.shape[0]
    
    print(torch.unique(test_DL.dataset.labels))
    
    test_DL.dataset.lenth = test_DL.dataset.data.shape[0]
    
    data_preparer = models.Data_preparer()
    
#     model_class = getattr(sys.modules[__name__], args.model)
    
    dataset_name = args.dataset
    
#     removed_count = args.removed_count
#     
#     args.removed_count = removed_count
    
#     args.device = 'cpu'
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    args.num_class = num_class
    
    full_training_noisy_dataset, full_training_origin_labels, small_dataset, selected_small_sample_ids, selected_noisy_sample_ids = generate_random_noisy_labels3(train_DL.dataset, args, ratio=args.noisy_ratio, soft_label=True, full_output_dir = full_output_dir, validation_dataset=valid_DL.dataset, dataset_test=test_DL.dataset)
    # full_training_noisy_dataset, full_training_origin_labels, small_dataset, selected_small_sample_ids, selected_noisy_sample_ids = generate_random_noisy_labels4(train_DL.dataset, args, ratio=0.9, soft_label=True, full_output_dir = full_output_dir, validation_dataset=valid_DL.dataset, dataset_test=test_DL.dataset)    
    w_list, grad_list, random_ids_multi_super_iterations, optimizer, model = initial_train_model(full_training_noisy_dataset, valid_DL.dataset, test_DL.dataset, args, binary=False, is_early_stopping = False)
#     small_dataset.labels = torch.argmax(small_dataset.labels, dim = 1)
    
    valid_model_dataset(model, valid_DL.dataset, None, args.bz, 'init validation F1', args.GPU, args.device, f1=True)
         
    valid_model_dataset(model, test_DL.dataset, None, args.bz, 'init test F1', args.GPU, args.device, f1=True)
    
    
#     validation_dataset.labels = torch.argmax(validation_dataset.labels, dim = 1)
    if small_dataset is not None:
        print('small dataset size::', small_dataset.data.shape[0])

    torch.save(full_training_noisy_dataset, full_output_dir + '/full_training_noisy_dataset')
    
    torch.save(full_training_origin_labels, full_output_dir + '/full_training_origin_labels')
    
    torch.save(valid_DL.dataset, full_output_dir + '/validation_dataset')
    
    if small_dataset is not None:
        torch.save(small_dataset, full_output_dir + '/small_dataset')
    
    torch.save(test_DL.dataset, full_output_dir + '/test_dataset')
    
    if selected_small_sample_ids is not None:
        torch.save(selected_small_sample_ids, full_output_dir + '/selected_small_sample_ids')
    
    if selected_noisy_sample_ids is not None:
        torch.save(selected_noisy_sample_ids, full_output_dir + '/selected_noisy_sample_ids')
    
    
if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")

    args = parse_optim_del_args()
    main(args)
    