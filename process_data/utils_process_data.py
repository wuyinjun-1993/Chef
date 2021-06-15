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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/iterative_detect')

try:
    from iterative_detect.utils_iters import *
except:
    from utils_iters import *



def compute_prior(validation_labels):
    positive_prior = torch.sum(validation_labels == 1)/len(validation_labels)
    
    negative_prior = 1 - positive_prior
    
    return {1: positive_prior, 0:negative_prior}


def compute_consolidate_labels(confusion_mat_list, worker_label_list, prior_probs):
    
    possible_label_list = [0,1]
    
    prob_list= []
    
    for possible_label in possible_label_list:
    
        res = prior_probs[possible_label].clone()
        
        for k in range(len(worker_label_list)):
            
            res = res* confusion_mat_list[k][possible_label][worker_label_list[k]]
            
        prob_list.append(res)
    
    return np.argmax(np.array(prob_list)) 

def compute_consolidate_labels_full(confusion_mat_list, full_worker_label_list, prior_probs):
    
    
    consolidate_label_list = []
    
    for k in range(len(full_worker_label_list)):
        curr_label = compute_consolidate_labels(confusion_mat_list, full_worker_label_list[k], prior_probs)
        
        consolidate_label_list.append(curr_label)
        
    return torch.tensor(consolidate_label_list)
    
    # possible_label_list = [0,1]
    #
    # prob_list= []
    #
    #
    #
    # for possible_label in possible_label_list:
    #
        # res = prior_probs[possible_label].clone()
        #
        # for k in range(len(worker_label_list)):
        #
            # res = res* confusion_mat_list[k][possible_label][worker_label_list[k]]
            #
        # prob_list.append(res)
        #
    # return np.argmax(np.array(prob_list)) 




def get_confusionMat_element(confusion_mat_list, k, possible_label, curr_worker_label_list):
    
    curr_prob_res_list = []
    
    for r in range(len(curr_worker_label_list)):
        
        if curr_worker_label_list[r].item() == -1:
            curr_prob_res = 0     
        else:
            curr_prob_res = confusion_mat_list[k][possible_label][curr_worker_label_list[r].item()]
        
        curr_prob_res_list.append(curr_prob_res)
        
    return torch.tensor(curr_prob_res_list)
    
    

def compute_consolidate_multi_labels(confusion_mat_list, worker_label_list, prior_probs):
    
    possible_label_list = [0,1]
    
    prob_list= []
    
    for possible_label in possible_label_list:
    
        res = prior_probs[possible_label]
        
        for k in range(worker_label_list.shape[1]):
            
            curr_res = get_confusionMat_element(confusion_mat_list, k, possible_label, worker_label_list[:,k])
            
            res = res*curr_res
            # res *= confusion_mat_list[k][possible_label][worker_label_list[:,k]]
            
        prob_list.append(res)
    
    return torch.argmax(torch.stack(prob_list, dim = 1), dim = 1)


def compute_posterior(worker_label_list, confusion_mat_list, label):
    res = 1
    
    for k in range(len(worker_label_list)):
        res = res * confusion_mat_list[k][label][worker_label_list[k]]
        
    return res
        

def get_all_label_combination(worker_count):
    
    worker_label_combinations = []
    
    for i in range(worker_count):
        if i == 0:
            worker_label_combinations.append([0])
            worker_label_combinations.append([1])
            
        else:
            
            worker_label_combination_count = len(worker_label_combinations)
            
            worker_label_combinations.extend(deepcopy(worker_label_combinations))
            
            for k in range(worker_label_combination_count):
                worker_label_combinations[k].append(0)
            for k in range(worker_label_combination_count):
                worker_label_combinations[k+worker_label_combination_count].append(1)
                
    return worker_label_combinations

    
def compute_final_confusion_matrix(confusion_mat_list, worker_count, prior_probs):
    
    # consolidate_label = compute_consolidate_labels(confusion_mat_list, worker_label_list, prior_probs)
    
    all_worker_label_combination = get_all_label_combination(worker_count)
    
    final_confusion_matrix = {}
    
    for Y_prime in [0,1]:
        for Y in [0,1]:
            posterior = 1
            
            for k in range(len(all_worker_label_combination)):
                worker_label_combination = all_worker_label_combination[k]
                curr_consolidate_label = compute_consolidate_labels(confusion_mat_list, worker_label_combination, prior_probs)
                if Y_prime == curr_consolidate_label:
                    posterior*=compute_posterior(worker_label_combination, confusion_mat_list, Y)
                    
            # if len(final_confusion_matrix) <= 0:
            if Y_prime not in final_confusion_matrix:  
                final_confusion_matrix[Y_prime] = {}
            final_confusion_matrix[Y_prime][Y] = posterior
        
    return final_confusion_matrix

def compute_prob_case_2(final_confusion_matrix, prior_prob, consolidate_label_list):
    
    prob_case2_list = []
    
    for k in range(len(consolidate_label_list)):
        consolidate_label = consolidate_label_list[k].item()
        r1 = final_confusion_matrix[consolidate_label][1-consolidate_label]
        
        r2 = final_confusion_matrix[consolidate_label][consolidate_label]
        
        prob_case2 = (r1*prior_prob[1-consolidate_label])/(r1*prior_prob[1-consolidate_label] + r2*prior_prob[consolidate_label])
        
        prob_case2_list.append(prob_case2)
    
    return torch.tensor(prob_case2_list)



def compute_confusion_matrix(validation_gold_labels, validation_annotated_labels):
    
    # tp_list = []
    #
    # fp_list = []
    #
    # tn_list = []
    #
    # fn_list = []
    
    confusion_mat_list = []
    
    for k in range(validation_annotated_labels.shape[1]):
        tp = torch.sum((validation_gold_labels == 1) & (validation_annotated_labels[:,k] == 1))/torch.sum((validation_gold_labels == 1))
        
        fp = torch.sum((validation_gold_labels == 1) & (validation_annotated_labels[:,k] == 0))/torch.sum((validation_gold_labels == 1))
        
        tn = torch.sum((validation_gold_labels == 0) & (validation_annotated_labels[:,k] == 0))/torch.sum((validation_gold_labels == 0))
        
        fn = torch.sum((validation_gold_labels == 0) & (validation_annotated_labels[:,k] == 1))/torch.sum((validation_gold_labels == 0))
        
        tp = torch.clamp(tp, min = 0.0001, max=  0.9999)
        
        tn = torch.clamp(tn, min = 0.0001, max=  0.9999)
        
        fn = torch.clamp(fn, min = 0.0001, max=  0.9999)
        
        fp = torch.clamp(fp, min = 0.0001, max=  0.9999)
        
        confusion_mat_list.append({1:{1:tp.item(), 0:fp.item()}, 0:{0:tn.item(), 1:fn.item()}})
        
        # tp_list.append(tp)
        #
        # fp_list.append(fp)
        #
        # tn_list.append(tn)
        #
        # fn_list.append(fn)
        
    return confusion_mat_list#tp_list, fp_list, tn_list, fn_list
         
