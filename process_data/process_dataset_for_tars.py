'''
Created on May 29, 2021

'''

import csv
import json
from argparse import ArgumentParser
import os,sys
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import gensim.downloader as api
import re
import torch
from snorkel.labeling.model import LabelModel
import pandas as pd

import re

# import copy as copy
from copy import deepcopy

import os
import pickle
import numpy as np
import torch
import pandas as pd
# from utils import AVAILABLEDATASETS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from snorkel.labeling.model import LabelModel






sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/interactive_weak_supervision')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/iterative_detect')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/real_examples')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Reweight_examples')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/process_data')


from gensim.test.utils import datapath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/iterative_detect')


try:
    from models.utils_real import *
    from utils.utils import *
    from process_data.pre_process_fact import *
    from interactive_weak_supervision.iws_utils import *
    from interactive_weak_supervision.iws import InteractiveWeakSupervision
    from interactive_weak_supervision.torchmodels import TorchMLP
    from models.Data_preparer import *
    from models.util_func import *
    from iterative_detect.utils_iters import *
    from process_data.utils_process_data import *
    from Reweight_examples.utils_reweight import *
except ImportError:
    from utils_real import *
    from utils import *
    from pre_process_fact import *
    from iws_utils import *
    from iws import InteractiveWeakSupervision
    from torchmodels import TorchMLP
    from Data_preparer import *
    from util_func import *
    from utils_iters import *
    from utils_process_data import *
    from utils_reweight import *

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")

    args = parse_optim_del_args()
    
    obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
    
    full_training_noisy_dataset, full_training_origin_labels, validation_dataset, dataset_test, train_annotated_label_tensor, full_out_dir, selected_small_sample_ids,selected_noisy_sample_ids, train_origin_dataset, valid_origin_dataset, test_origin_dataset = obtain_data_function(args, noisy=True, load_origin= False)
    
    rounded_prob_labels = (torch.tensor(full_training_noisy_dataset.labels) > 0.5)[:,1].type(torch.long)
    
    print('out dir::', full_out_dir)

    origin_labeled_tensor = torch.load(full_out_dir + '/clean_sample_ids')
    
    origin_unlabeled_tensor = torch.load(full_out_dir + '/noisy_sample_ids')
    
    prob_label_tensor = onehot(rounded_prob_labels, 2)
    
    full_training_noisy_dataset.labels = prob_label_tensor
    
    print(full_training_noisy_dataset.data.shape, full_training_noisy_dataset.labels.shape)
    
    Y_prior = compute_prior(torch.tensor(list(full_training_origin_labels[origin_labeled_tensor])))
    
    if train_annotated_label_tensor is None:
        annotated_labels_tensor = load_human_annotations(0, full_out_dir)
    else:
        annotated_labels_tensor = train_annotated_label_tensor
    
    agg_labels = resolve_conflict_majority_vote(annotated_labels_tensor, origin_labeled_tensor)
    
    worker_confusion_matrix_list = compute_confusion_matrix(agg_labels, annotated_labels_tensor[origin_labeled_tensor])
    
    # dirty_train_df = train_df.iloc[dirty_ids]
    
    full_consolidate_matrix = compute_final_confusion_matrix(worker_confusion_matrix_list, 3, Y_prior)
    
    train_consolidate_label = compute_consolidate_multi_labels(worker_confusion_matrix_list, annotated_labels_tensor, Y_prior) 
    
    # full_prob_case2_list = torch.zeros(train_df.shape[0])

    # consolidate_label_list_tensor = compute_consolidate_labels_full(worker_confusion_matrix_list, torch.tensor(np.array(dirty_train_df[['T1_Q1','T2_Q1','T3_Q1']])), Y_prior)

    
    prob_case2_list = compute_prob_case_2(full_consolidate_matrix, Y_prior, train_consolidate_label)
    
    print('prob labels::', prob_case2_list, prob_case2_list.shape)
    
    print('consolidate labels::', train_consolidate_label, train_consolidate_label.shape)
    
    torch.save(prob_case2_list, os.path.join(full_out_dir, 'train_prob_case2'))
    torch.save(train_consolidate_label, os.path.join(full_out_dir, 'consolidate_label_list'))
    
    print('out dir::', full_out_dir)
    torch.save(full_training_noisy_dataset, os.path.join(full_out_dir,'full_training_noisy_dataset_tars'))
    
    
    