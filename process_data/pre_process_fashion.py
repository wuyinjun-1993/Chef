'''
Created on Apr 24, 2021

'''
import dill as pickle

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/process_data')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')

from gensim.test.utils import datapath


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



fashion_csv_file_name = 'Annotation_PerImage_All.csv'

other_info_file_names = ['Metadata/info.csv']

info_file_column_mappings = {'Metadata/info.csv':['O_URL', 'DESCRIPTION']}


def replace_url_with_image_name(url):
    image_name = url.split('/')[-1]
    real_image_name = image_name.split('_')[0]
    return real_image_name


def replace_numeric_labels(label):
    if label == 'yes':
        return 1
    else:
        if label == 'no':
            return 0
        else:
            return -1



    


def load_labels_per_image(clean_sample_ratio, file_name, git_ignore_dir, train_sample_names = None, valid_sample_names = None, test_sample_names = None, compare_tars = True):
    df = pd.read_csv(file_name)
    df['img_name'] = df['PictureURL'].apply(replace_url_with_image_name)
    df['T1_Q1'] = df['T1_Q1'].apply(replace_numeric_labels)
    
    df['T2_Q1'] = df['T2_Q1'].apply(replace_numeric_labels)
    
    df['T3_Q1'] = df['T3_Q1'].apply(replace_numeric_labels)
    
    df['Majority Q1'] = df['Majority Q1'].apply(replace_numeric_labels) 
    
    # train_df, valid_df, test_df = partition_data_frame_train_valid_test(df)
    # train_df = df[df['img_name'].isin()]
    if train_sample_names is not None and valid_sample_names is not None and test_sample_names is not None:
        train_df = train_sample_names.merge(df, how = 'inner', on = 'img_name')
        
        valid_df = valid_sample_names.merge(df, how = 'inner', on = 'img_name')
        
        test_df = test_sample_names.merge(df, how = 'inner', on = 'img_name')
    else:
        train_df, valid_df, test_df = partition_data_frame_train_valid_test(df)
    # train_img_names = train_df['img_name']
    
    full_info_file_name = os.path.join(git_ignore_dir, other_info_file_names[0])
    
    valid_df = valid_df[valid_df['Majority Q1'] != -1]
    
    prob_labels, clean_ids, dirty_ids, clean_df = read_descriptions_per_image(train_df, full_info_file_name, other_info_file_names[0], valid_df, git_ignore_dir, clean_sample_ratio = clean_sample_ratio)
    
    # prob_labels, clean_ids, dirty_ids = torch.rand([train_df.shape[0]], dtype = torch.float), torch.tensor(list(range(0,100))), torch.tensor(list(range(100,train_df.shape[0])))#read_descriptions_per_image(train_df, full_info_file_name, other_info_file_names[0], valid_df, git_ignore_dir)
    #
    # prob_labels = prob_labels[dirty_ids]
    
    if not compare_tars:
        prob_label_tensor = torch.zeros([len(prob_labels), 2], dtype = torch.float)
        
        prob_label_tensor[:,1] = torch.tensor(list(prob_labels))
        
        prob_label_tensor[:,0] = 1 - torch.tensor(list(prob_labels))
    
    else:
        rounded_prob_labels = (torch.tensor(prob_labels) > 0.5).type(torch.long)
        
        prob_label_tensor = onehot(rounded_prob_labels, 2)
        
        Y_prior = compute_prior(torch.tensor(list(clean_df['Majority Q1'])))
        
        worker_confusion_matrix_list = compute_confusion_matrix(torch.tensor(np.array(clean_df['Majority Q1'])), torch.tensor(np.array(clean_df[['T1_Q1','T2_Q1','T3_Q1']])))
        
        dirty_train_df = train_df.iloc[dirty_ids]
        
        full_consolidate_matrix = compute_final_confusion_matrix(worker_confusion_matrix_list, 3, Y_prior)
        
        train_consolidate_label = compute_consolidate_multi_labels(worker_confusion_matrix_list, torch.tensor(np.array(train_df[['T1_Q1','T2_Q1','T3_Q1']])), Y_prior) 
        
        # full_prob_case2_list = torch.zeros(train_df.shape[0])
    
        # consolidate_label_list_tensor = compute_consolidate_labels_full(worker_confusion_matrix_list, torch.tensor(np.array(dirty_train_df[['T1_Q1','T2_Q1','T3_Q1']])), Y_prior)
    
        
        prob_case2_list = compute_prob_case_2(full_consolidate_matrix, Y_prior, train_consolidate_label)
        
        # full_prob_case2_list[dirty_ids] = prob_case2_list
        
    
    torch.save(clean_ids, os.path.join(git_ignore_dir, 'clean_sample_ids'))
    
    torch.save(dirty_ids, os.path.join(git_ignore_dir, 'noisy_sample_ids'))
    
    if compare_tars:
        torch.save(prob_case2_list, os.path.join(git_ignore_dir, 'train_prob_case2'))
        torch.save(train_consolidate_label, os.path.join(git_ignore_dir, 'consolidate_label_list'))
    training_labels = torch.zeros([train_df.shape[0], 2])
    
    training_labels[dirty_ids] = prob_label_tensor
    
    
    if len(clean_ids) > 0:
        training_labels[clean_ids] = onehot(torch.tensor(list(train_df.iloc[clean_ids]['Majority Q1'])), 2)
    
    print(training_labels)
    
    return list(train_df['img_name']), list(valid_df['img_name']), list(test_df['img_name']), training_labels, torch.from_numpy(np.array(train_df[['T1_Q1','T2_Q1','T3_Q1']])), torch.tensor(list(valid_df['Majority Q1'])), torch.tensor(list(test_df['Majority Q1']))
    # return df[['img_name', 'T1_Q1','T2_Q1','T3_Q1','Majority Q1']]


def extract_img_name(url):
    image_name = url.split('/')[-1]
    real_image_name = image_name.split('_')[0]
    return real_image_name 




def derive_prob_labels(corpus, ground_truth_label, corpus_test, git_ignore_dir):
    ground_truth_label_copy = np.array(ground_truth_label)
    
    # ground_truth_label_copy[ground_truth_label_copy == 0] = -1
    #
    # ground_truth_label_copy[ground_truth_label_copy == 2] = 0
    
    
    
    
    mindf = 0#10.0/len(corpus)
    vectorizer = CountVectorizer(strip_accents='ascii',stop_words='english',ngram_range=(1, 1), analyzer='word', max_df=0.90, min_df=mindf, max_features=None, vocabulary=None, binary=False)
    # print(corpus)
    Xunigram = vectorizer.fit_transform(corpus)
    
    Xunigramtest = vectorizer.transform(corpus_test)
    n,m = Xunigram.shape
    if m < 300:
        Xsvd = np.asarray(Xunigram.todense()).astype(np.float32)
        Xtestsvd = np.asarray(Xunigramtest.todense()).astype(np.float32)
    else:        
        svd = TruncatedSVD(n_components=300, n_iter=20, random_state=42)
        Xsvd = svd.fit_transform(Xunigram).astype(np.float32)
        Xtestsvd = svd.transform(Xunigramtest).astype(np.float32)

    LFs,lf_descriptions = generate_ngram_LFs(corpus,'unigram',mindf = 0)
    
    svd = TruncatedSVD(n_components=30, n_iter=20, random_state=42)
    LFfeatures = svd.fit_transform(LFs.T).astype(np.float32)
    
    
    positive_sentiment_words = ['beautiful', 'fashionable']
    negative_sentimen_words = ['obsolete','outdated']
    # neutral_sentimatent_words = ['the', 'will']
    start_idxs = []
    halfway= int(len(lf_descriptions)/2)# we generated positive LFs first, then negative
    # iterate over positive LFs
    for i,desc in enumerate(lf_descriptions[:halfway]):
        for word in positive_sentiment_words:
            if word in desc:
                start_idxs.append(i)
                if len(start_idxs)==2:
                    break
        if len(start_idxs)==2:
            break
    
    # iterate over negative LFs
    for i,desc in enumerate(lf_descriptions[halfway:2*halfway]):
        idx = halfway+i
        for word in negative_sentimen_words:
            if word in desc:
                start_idxs.append(idx)
                if len(start_idxs)==4:
                    break
        if len(start_idxs)==4:
            break
        
    # for i,desc in enumerate(lf_descriptions[2*halfway:]):
        # idx = halfway+i
        # for word in neutral_sentimatent_words:
            # if word in desc:
                # start_idxs.append(idx)
                # if len(start_idxs)==6:
                    # break
        # if len(start_idxs)==6:
            # break    
    
    for i in start_idxs:
        print(lf_descriptions[i])
        
        
    initial_labels = {i:1 for i in start_idxs}
    saveinfo = {'dname':'twitter','lftype':'unigram'}
    savedir=git_ignore_dir
    savetodisc = False
    username = 'user'
    numthreads = min(10, os.cpu_count())
    
    # if not os.path.exists(os.path.join(git_ignore_dir, 'iws_session.pkl')):
    IWSsession = InteractiveWeakSupervision(LFs,LFfeatures,lf_descriptions,initial_labels,acquisition='LSE', r=0.6,
                                            Ytrue=ground_truth_label_copy, auto=True, corpus=corpus, save_to_disc=savetodisc, savedir=savedir, 
                                            saveinfo=saveinfo, username=username, progressbar=True,
                                            ensemblejobs=numthreads,numshow=2)
    num_iter = 150
    # IWSsession.run_experiments(num_iter = 150)
    IWSsession.run_experiments(num_iter = num_iter)
    IWSsession.model.mpool.close()
    IWSsession.model.mpool.join()
        # with open(os.path.join(git_ignore_dir, 'iws_session.pkl'), 'wb') as output:
            # pickle.dump(IWSsession, output, pickle.HIGHEST_PROTOCOL)
            #
    # else:
        # with open(os.path.join(git_ignore_dir, 'iws_session.pkl'), 'rb') as input_info:
            # IWSsession = pickle.load(input_info)
    
    LFsets = get_final_set('LSE ac',IWSsession,npredict=200,r=None)
    
    downstream_results = train_end_classifier(Xsvd,Xtestsvd,IWSsession,LFsets,gap=20,verbose=True)
    
    return downstream_results[1][num_iter-1]
    
# def iterate_images_convert_to_inter_representations(image_main_folder):
    # for sub_folder in os.listdir(image_main_folder):
        # if os.path.isdir(sub_folder):
            # full_dir = os.path.join(image_main_folder, sub_folder)
            # for image_files in os.listdir(full_dir):

def group_text(text_set):
    
    text_list = list(text_set)
    
    text_list = [re.sub(r'[^A-Za-z0-9 ]+', '', x) for x in text_list if str(x) != 'nan']
    
    if len(text_list) <= 0:
        return ''
    
    text_list =' '.join(list(text_set))
    
    return text_list


def replace_nan_values(text):
    if str(text) == 'nan':
        return ''
    
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    
    return text



def read_descriptions_per_image(train_df, full_file_name, file_name, valid_df, git_ignore_dir, clean_sample_ratio = 0):
    df = pd.read_csv(full_file_name,index_col=False)
    url_cln_name = info_file_column_mappings[file_name][0]
    info_cln_name = info_file_column_mappings[file_name][1]
    
    df[url_cln_name] = df[url_cln_name].apply(replace_url_with_image_name)
    
    df = df.groupby(url_cln_name)[info_cln_name].apply(group_text)
    
    
    train_merged_df = train_df.merge(df, how = 'left', left_on = 'img_name', right_on = url_cln_name)
    
    valid_merged_df = valid_df.merge(df, how = 'left', left_on = 'img_name', right_on = url_cln_name)
    
    # no_noise_label_train_merged_df = train_merged_df[train_merged_df['Majority Q1'] != -1]
    
    # no_noise_label_boolean = (np.array()
    #
    no_noise_label_ids = np.nonzero(np.array(train_merged_df['Majority Q1'] != -1))[0].tolist()
    
    # train_merged_df.loc[]
    
    # no_noise_train_ids = np.random.permutation(list(range(no_noise_label_train_merged_df.shape[0])))
    
    np.random.shuffle(no_noise_label_ids)
    
    clean_sample_count = int(train_merged_df.shape[0]*clean_sample_ratio)
        
    clean_ids = no_noise_label_ids[0:clean_sample_count]
    
    dirty_ids = list(set(range(train_merged_df.shape[0])).difference(set(clean_ids)))
    
    # dirty_ids = train_ids[clean_sample_count:]
    
    clean_df = pd.concat([train_merged_df.iloc[clean_ids], valid_merged_df]) 
    
    dirty_df = train_merged_df.iloc[dirty_ids]
    
    
    
    
    corpus = clean_df[info_cln_name].apply(replace_nan_values)
    
    corpus_test = dirty_df[info_cln_name].apply(replace_nan_values)# df.loc[(df[url_cln_name].isin(train_img_name))][info_cln_name]
    # img_label_mappings[]
    # url_list = list(df[url_cln_name])
    #
    # info_list = list(df[info_cln_name])
    
    prob_labels = derive_prob_labels(list(corpus), clean_df['Majority Q1'], list(corpus_test), git_ignore_dir)
    
    return prob_labels, clean_ids, dirty_ids, clean_df


def partition_data_frame_train_valid_test(df):
    
    labeling_no_conflict_df = df.loc[(df['T1_Q1'] == df['T2_Q1']) & (df['T3_Q1'] == df['T2_Q1'])]
    
    labeling_conflict_df = df.loc[(df['T1_Q1'] != df['T2_Q1']) | (df['T3_Q1'] != df['T2_Q1'])]
    
    total_size = df.shape[0]

    indices = np.array(range(len(labeling_no_conflict_df)))

    np.random.seed(0)

    np.random.shuffle(indices)
    
    valid_count = int(len(labeling_no_conflict_df)*0.1)
    
    test_count = int(len(labeling_no_conflict_df)*0.2)
    
    valid_idx, test_idx = indices[test_count:valid_count + test_count], indices[0:test_count]
    
    # train_df = df.iloc[train_idx]
    
    valid_df = labeling_no_conflict_df.iloc[valid_idx]
    
    test_df = labeling_no_conflict_df.iloc[test_idx]
    
    train_df = pd.concat([labeling_conflict_df, labeling_no_conflict_df.iloc[indices[valid_count + test_count:]]])
    
    print('train_df shape::', train_df.shape)
    
    print('valid_df shape::', valid_df.shape)
    
    print('test_df shape::', test_df.shape)
    
    print(len(train_df) + len(valid_df) + len(test_df))
    
    print('df shape::', df.shape)
    
    print(((valid_df['T1_Q1'] == valid_df['T2_Q1'])&(valid_df['T3_Q1'] == valid_df['T2_Q1'])).unique())
    
    return train_df, valid_df, test_df
    

def get_loss_all_samples(model, dataset, critertion):
    loss_list = []
    
    output = model(dataset.data)
    
    loss_list = critertion(output, dataset.labels.view(-1))
    
    # for i in range(dataset.data.shape[0]):
        # X = dataset.data[i:i+1]
        #
            # # if isinstance(model, models.Logistic_regression) or isinstance(model, models.Binary_Logistic_regression):
                # # X = X.view(X.shape[0], -1)
                #
        # Y = dataset.labels[i:i+1]
        #
        # X = X.type(torch.DoubleTensor)
        #
        # output = model(X)
        #
        # curr_loss = critertion(output, Y.type(torch.double)).view(-1)
        #
        # loss_list.append(curr_loss)
        
    return loss_list


def select_valid_test_dataset2(args, output_dir, train_img_names, valid_img_names, test_img_names,  valid_ratio = 0.005, test_ratio = 0.005, select_range = 0.6):
    
    
    trans_train_dataset = torch.load(output_dir + '/' + transformed_train_dataset_file_name)
    
    trans_valid_dataset = torch.load(output_dir + '/' + transformed_val_dataset_file_name)
    
    trans_test_dataset = torch.load(output_dir + '/' + transformed_test_dataset_file_name)
    
    model = models.Logistic_regression(trans_train_dataset.data.shape[1], 2, bias = True)

    optimizer = model.get_optimizer(args.tlr, args.wd)
    
    full_features = torch.cat([trans_train_dataset.data, trans_valid_dataset.data, trans_test_dataset.data], 0)
    
    full_labels = torch.cat([trans_train_dataset.labels, trans_valid_dataset.labels, trans_test_dataset.labels], 0)

    full_img_names = pd.concat([train_img_names, valid_img_names, test_img_names], axis=0)
    
    full_features = full_features[full_labels.view(-1) != -1]
    
    full_img_names = full_img_names.iloc[(full_labels.view(-1) != -1).numpy()]
    
    full_labels = full_labels[full_labels.view(-1) != -1]
    
    full_dataset = MyDataset(full_features, full_labels)
    
    print(full_img_names.shape, full_labels.shape, full_features.shape)
    # initial_train_model(full_dataset, full_dataset, full_dataset, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
    w_list, grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, full_dataset.data, onehot(full_dataset.labels,2), args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping =False, r_weight = None, test_performance = False)
    
    
    loss_list_tensor = get_loss_all_samples(model, full_dataset, model.get_loss_function(reduction = 'none'))
    
    sorted_items, sorted_ids = torch.sort(loss_list_tensor, descending = False)
    
    valid_count = int(full_dataset.data.shape[0]*valid_ratio)
    
    test_count = int(full_dataset.data.shape[0]*test_ratio)
    
    # selected_valid_test_ids = sorted_ids[0: valid_count + test_count]
    
    
    candidate_selected_id_count = int(full_dataset.data.shape[0]*select_range)
    
    candidate_selected_ids = sorted_ids[0:candidate_selected_id_count]
    
    random_candidate_selected_ids = torch.randperm(len(candidate_selected_ids))
    
    
    # train_ids = sorted_ids[valid_count + test_count:]
    
    # shuffle_ids = torch.randperm(len(selected_valid_test_ids))
    
    valid_ids =  candidate_selected_ids[random_candidate_selected_ids[0:valid_count]]
    
    test_ids =  candidate_selected_ids[random_candidate_selected_ids[valid_count:test_count + valid_count]]
    
    all_valid_test_ids = set(valid_ids.tolist()).union(set(test_ids.tolist()))
    
    print(torch.unique(torch.sum(valid_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    
    print(torch.unique(torch.sum(test_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    
    train_ids = torch.tensor(list(set(list(range(full_dataset.data.shape[0]))).difference(all_valid_test_ids)))
    
    train_dataset = MyDataset(full_dataset.data[train_ids], full_dataset.labels[train_ids])
    
    valid_dataset = MyDataset(full_dataset.data[valid_ids], full_dataset.labels[valid_ids])
    
    test_dataset = MyDataset(full_dataset.data[test_ids], full_dataset.labels[test_ids])
    
    new_train_img_names = full_img_names.iloc[train_ids]
    
    new_valid_img_names = full_img_names.iloc[valid_ids]
    
    new_test_img_names = full_img_names.iloc[test_ids]
    
    
    torch.save(train_dataset, output_dir + '/' + transformed_train_dataset_file_name + '_new')
    
    torch.save(valid_dataset, output_dir + '/' + transformed_val_dataset_file_name + '_new')
    
    torch.save(test_dataset, output_dir + '/' + transformed_test_dataset_file_name + '_new')
    
    return train_dataset, valid_dataset, test_dataset, new_train_img_names, new_valid_img_names, new_test_img_names


def sample_data_and_select_test_samples(test_count, remaining_ids, valid_features, full_features, full_labels):
    # dist = np.random.multivariate_normal(valid_mean.numpy(), torch.diag(valid_std).numpy())
    
    selected_ids = []
    
    for k in range(test_count):
        # sampled_feature = torch.normal(mean=valid_mean, std=valid_std)
        sampled_feature = valid_features[k]
        
        min_value = None
        
        min_id = None
        
        for j in range(len(remaining_ids)):
            curr_feature = full_features[remaining_ids[j]]
            curr_label = full_labels[remaining_ids[j]]
            
            distance = -(torch.dot(curr_feature,sampled_feature)/(torch.norm(curr_feature)*torch.norm(sampled_feature))).item()#
            
            if min_value is None:
                min_value = distance
                
                min_id = j
            
            # if torch.norm(curr_feature - sampled_feature).item() < min_value:
            if distance < min_value:
                
                min_value = distance#(torch.dot(curr_feature,sampled_feature)/(torch.norm(curr_feature)*torch.norm(sampled_feature))).item()#
                
                min_id = j
            
        
        selected_ids.append(remaining_ids[min_id])
        
        print(full_labels[remaining_ids[min_id]])
        
        del remaining_ids[min_id]
        
    return selected_ids, remaining_ids

def select_valid_test_dataset4(args, output_dir, train_img_names, valid_img_names, test_img_names,  valid_ratio = 0.005, test_ratio = 0.005):
    
    
    trans_train_dataset = torch.load(output_dir + '/' + transformed_train_dataset_file_name)
    
    trans_valid_dataset = torch.load(output_dir + '/' + transformed_val_dataset_file_name)
    
    trans_test_dataset = torch.load(output_dir + '/' + transformed_test_dataset_file_name)
    
    origin_train_dataset = torch.load(output_dir + '/origin_train_dataset')
        
    origin_val_dataset = torch.load(output_dir + '/origin_val_dataset')
    
    origin_test_dataset = torch.load(output_dir + '/origin_test_dataset')
    
    model = models.Logistic_regression(trans_train_dataset.data.shape[1], 2, bias = True)

    optimizer = model.get_optimizer(args.tlr, args.wd)
    
    full_features = torch.cat([trans_train_dataset.data, trans_valid_dataset.data, trans_test_dataset.data], 0)
    
    full_labels = torch.cat([trans_train_dataset.labels, trans_valid_dataset.labels, trans_test_dataset.labels], 0)

    full_origin_features = torch.cat([origin_train_dataset.data, origin_val_dataset.data, origin_test_dataset.data], 0)
    
    full_origin_labels = torch.cat([origin_train_dataset.labels, origin_val_dataset.labels, origin_test_dataset.labels], 0)




    full_img_names = pd.concat([train_img_names, valid_img_names, test_img_names], axis=0)
    
    full_features = full_features[full_labels.view(-1) != -1]
    
    full_origin_features = full_origin_features[full_labels.view(-1) != -1]
    
    full_img_names = full_img_names.iloc[(full_labels.view(-1) != -1).numpy()]
    
    full_labels = full_labels[full_labels.view(-1) != -1]
    
    full_origin_labels = full_origin_labels[full_labels.view(-1) != -1]
    
    full_dataset = MyDataset(full_features, full_labels)
    
    full_origin_dataset = MyDataset(full_origin_features, full_origin_labels)
    
    valid_count = int(full_dataset.data.shape[0]*valid_ratio)
    
    test_count = int(full_dataset.data.shape[0]*test_ratio)
    
    all_rand_ids = torch.randperm(full_features.shape[0])
    
    valid_ids = all_rand_ids[0:valid_count]
    
    valid_features = full_features[valid_ids]
    
    valid_labels = full_labels[valid_ids]
    
    valid_mean_label_1 = torch.mean(valid_features[valid_labels.view(-1) == 1],0)
    
    valid_std_label_1 = torch.std(valid_features[valid_labels.view(-1) == 1], 0)
    
    
    valid_mean_label_0 = torch.mean(valid_features[valid_labels.view(-1) == 0],0)
    
    valid_std_label_0 = torch.std(valid_features[valid_labels.view(-1) == 0], 0)
    
    positive_ids = torch.nonzero(full_labels.view(-1) == 1).view(-1).tolist()
    
    negative_ids = torch.nonzero(full_labels.view(-1) == 0).view(-1).tolist()
    
    positive_remaining_ids = list(set(positive_ids).difference(set(valid_ids.tolist())))
    
    negative_remaining_ids = list(set(negative_ids).difference(set(valid_ids.tolist())))
    
    selected_ids_label_1, remaining_ids = sample_data_and_select_test_samples(torch.sum(valid_labels.view(-1) == 1), positive_remaining_ids, valid_features[valid_labels.view(-1) == 1], full_features, full_labels)
    
    selected_ids_label_0, remaining_ids = sample_data_and_select_test_samples(torch.sum(valid_labels.view(-1) == 0), negative_remaining_ids, valid_features[valid_labels.view(-1) == 0], full_features, full_labels)
    
    test_ids = []
    
    test_ids.extend(selected_ids_label_1)
    
    test_ids.extend(selected_ids_label_0)
    # valid_dataset = 
    
    
    
    
    
    
    
    print(full_img_names.shape, full_labels.shape, full_features.shape)
    # initial_train_model(full_dataset, full_dataset, full_dataset, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
    # w_list, grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, full_dataset.data, onehot(full_dataset.labels,2), args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping =False, r_weight = None, test_performance = False)
    #
    #
    # loss_list_tensor = get_loss_all_samples(model, full_dataset, model.get_loss_function(reduction = 'none'))
    #
    # sorted_items, sorted_ids = torch.sort(loss_list_tensor, descending = False)
    #
    # valid_count = int(full_dataset.data.shape[0]*valid_ratio)
    #
    # test_count = int(full_dataset.data.shape[0]*test_ratio)
    #
    # # selected_valid_test_ids = sorted_ids[0: valid_count + test_count]
    #
    #
    # candidate_upper_selected_id_count = int(full_dataset.data.shape[0]*select_up_range)
    #
    # candidate_lower_selected_id_count = int(full_dataset.data.shape[0]*select_low_range)
    #
    # candidate_selected_ids = sorted_ids[candidate_lower_selected_id_count:candidate_upper_selected_id_count]
    #
    # random_candidate_selected_ids = torch.randperm(len(candidate_selected_ids))
    #
    #
    # # train_ids = sorted_ids[valid_count + test_count:]
    #
    # # shuffle_ids = torch.randperm(len(selected_valid_test_ids))
    #
    # valid_ids =  candidate_selected_ids[random_candidate_selected_ids[0:valid_count]]
    #
    # test_ids =  candidate_selected_ids[random_candidate_selected_ids[valid_count:test_count + valid_count]]
    
    test_ids = torch.tensor(test_ids)
    
    all_valid_test_ids = set(valid_ids.tolist()).union(set(test_ids.tolist()))
    
    print('valid test total count::', len(all_valid_test_ids))
    
    # print(torch.unique(torch.sum(valid_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    #
    # print(torch.unique(torch.sum(test_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    
    train_ids = torch.tensor(list(set(list(range(full_dataset.data.shape[0]))).difference(all_valid_test_ids)))
    
    train_dataset = MyDataset(full_dataset.data[train_ids], full_dataset.labels[train_ids])
    
    valid_dataset = MyDataset(full_dataset.data[valid_ids], full_dataset.labels[valid_ids])
    
    test_dataset = MyDataset(full_dataset.data[test_ids], full_dataset.labels[test_ids])
    
    origin_train_dataset_new = MyDataset(full_origin_dataset.data[train_ids], full_origin_dataset.labels[train_ids])
    
    origin_valid_dataset_new = MyDataset(full_origin_dataset.data[valid_ids], full_origin_dataset.labels[valid_ids])
    
    origin_test_dataset_new = MyDataset(full_origin_dataset.data[test_ids], full_origin_dataset.labels[test_ids])
    
    new_train_img_names = full_img_names.iloc[train_ids]
    
    new_valid_img_names = full_img_names.iloc[valid_ids]
    
    new_test_img_names = full_img_names.iloc[test_ids]
    
    
    torch.save(train_dataset, output_dir + '/' + transformed_train_dataset_file_name + '_new')
    
    torch.save(valid_dataset, output_dir + '/' + transformed_val_dataset_file_name + '_new')
    
    torch.save(test_dataset, output_dir + '/' + transformed_test_dataset_file_name + '_new')
    
    torch.save(origin_train_dataset_new, output_dir + '/origin_train_dataset_new')
        
    torch.save(origin_valid_dataset_new, output_dir + '/origin_val_dataset_new')
    
    torch.save(origin_test_dataset_new, output_dir + '/origin_test_dataset_new')
    
    return train_dataset, valid_dataset, test_dataset, new_train_img_names, new_valid_img_names, new_test_img_names



def select_valid_test_dataset3(args, output_dir, train_img_names, valid_img_names, test_img_names,  valid_ratio = 0.005, test_ratio = 0.005, select_low_range = 0.4, select_up_range = 0.8):
    
    
    trans_train_dataset = torch.load(output_dir + '/' + transformed_train_dataset_file_name)
    
    trans_valid_dataset = torch.load(output_dir + '/' + transformed_val_dataset_file_name)
    
    trans_test_dataset = torch.load(output_dir + '/' + transformed_test_dataset_file_name)
    
    model = models.Logistic_regression(trans_train_dataset.data.shape[1], 2, bias = True)

    optimizer = model.get_optimizer(args.tlr, args.wd)
    
    full_features = torch.cat([trans_train_dataset.data, trans_valid_dataset.data, trans_test_dataset.data], 0)
    
    full_labels = torch.cat([trans_train_dataset.labels, trans_valid_dataset.labels, trans_test_dataset.labels], 0)

    full_img_names = pd.concat([train_img_names, valid_img_names, test_img_names], axis=0)
    
    full_features = full_features[full_labels.view(-1) != -1]
    
    full_img_names = full_img_names.iloc[(full_labels.view(-1) != -1).numpy()]
    
    full_labels = full_labels[full_labels.view(-1) != -1]
    
    full_dataset = MyDataset(full_features, full_labels)
    
    print(full_img_names.shape, full_labels.shape, full_features.shape)
    # initial_train_model(full_dataset, full_dataset, full_dataset, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
    w_list, grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, full_dataset.data, onehot(full_dataset.labels,2), args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping =False, r_weight = None, test_performance = False)
    
    
    loss_list_tensor = get_loss_all_samples(model, full_dataset, model.get_loss_function(reduction = 'none'))
    
    sorted_items, sorted_ids = torch.sort(loss_list_tensor, descending = False)
    
    valid_count = int(full_dataset.data.shape[0]*valid_ratio)
    
    test_count = int(full_dataset.data.shape[0]*test_ratio)
    
    # selected_valid_test_ids = sorted_ids[0: valid_count + test_count]
    
    
    candidate_upper_selected_id_count = int(full_dataset.data.shape[0]*select_up_range)
    
    candidate_lower_selected_id_count = int(full_dataset.data.shape[0]*select_low_range)
    
    candidate_selected_ids = sorted_ids[candidate_lower_selected_id_count:candidate_upper_selected_id_count]
    
    random_candidate_selected_ids = torch.randperm(len(candidate_selected_ids))
    
    
    # train_ids = sorted_ids[valid_count + test_count:]
    
    # shuffle_ids = torch.randperm(len(selected_valid_test_ids))
    
    valid_ids =  candidate_selected_ids[random_candidate_selected_ids[0:valid_count]]
    
    test_ids =  candidate_selected_ids[random_candidate_selected_ids[valid_count:test_count + valid_count]]
    
    all_valid_test_ids = set(valid_ids.tolist()).union(set(test_ids.tolist()))
    
    print(torch.unique(torch.sum(valid_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    
    print(torch.unique(torch.sum(test_ids.view(-1,1) == candidate_selected_ids.view(1,-1), dim = 1)))
    
    train_ids = torch.tensor(list(set(list(range(full_dataset.data.shape[0]))).difference(all_valid_test_ids)))
    
    train_dataset = MyDataset(full_dataset.data[train_ids], full_dataset.labels[train_ids])
    
    valid_dataset = MyDataset(full_dataset.data[valid_ids], full_dataset.labels[valid_ids])
    
    test_dataset = MyDataset(full_dataset.data[test_ids], full_dataset.labels[test_ids])
    
    new_train_img_names = full_img_names.iloc[train_ids]
    
    new_valid_img_names = full_img_names.iloc[valid_ids]
    
    new_test_img_names = full_img_names.iloc[test_ids]
    
    
    torch.save(train_dataset, output_dir + '/' + transformed_train_dataset_file_name + '_new')
    
    torch.save(valid_dataset, output_dir + '/' + transformed_val_dataset_file_name + '_new')
    
    torch.save(test_dataset, output_dir + '/' + transformed_test_dataset_file_name + '_new')
    
    return train_dataset, valid_dataset, test_dataset, new_train_img_names, new_valid_img_names, new_test_img_names

def select_valid_test_dataset(args, output_dir, train_img_names, valid_img_names, test_img_names,  valid_ratio = 0.005, test_ratio = 0.005, select_range = 0.1):
    
    
    trans_train_dataset = torch.load(output_dir + '/' + transformed_train_dataset_file_name)
    
    trans_valid_dataset = torch.load(output_dir + '/' + transformed_val_dataset_file_name)
    
    trans_test_dataset = torch.load(output_dir + '/' + transformed_test_dataset_file_name)
    
    model = models.Logistic_regression(trans_train_dataset.data.shape[1], 2, bias = True)

    optimizer = model.get_optimizer(args.tlr, args.wd)
    
    full_features = torch.cat([trans_train_dataset.data, trans_valid_dataset.data, trans_test_dataset.data], 0)
    
    full_labels = torch.cat([trans_train_dataset.labels, trans_valid_dataset.labels, trans_test_dataset.labels], 0)

    full_img_names = pd.concat([train_img_names, valid_img_names, test_img_names], axis=0)
    
    full_features = full_features[full_labels.view(-1) != -1]
    
    full_img_names = full_img_names.iloc[(full_labels.view(-1) != -1).numpy()]
    
    full_labels = full_labels[full_labels.view(-1) != -1]
    
    full_dataset = MyDataset(full_features, full_labels)
    
    print(full_img_names.shape, full_labels.shape, full_features.shape)
    # initial_train_model(full_dataset, full_dataset, full_dataset, args, binary=False, is_early_stopping = False, random_ids_multi_super_iterations = None)
    w_list, grad_list,random_ids_multi_super_iterations = train_model_dataset(args, model, optimizer, None, full_dataset.data, onehot(full_dataset.labels,2), args.bz, args.epochs, args.GPU, args.device, loss_func = model.soft_loss_function_reduce, val_dataset = None, test_dataset = None, f1 = False, capture_prov = False, is_early_stopping =False, r_weight = None, test_performance = False)
    
    
    loss_list_tensor = get_loss_all_samples(model, full_dataset, model.get_loss_function(reduction = 'none'))
    
    sorted_items, sorted_ids = torch.sort(loss_list_tensor, descending = False)
    
    valid_count = int(full_dataset.data.shape[0]*valid_ratio)
    
    test_count = int(full_dataset.data.shape[0]*test_ratio)
    
    selected_valid_test_ids = sorted_ids[0: valid_count + test_count]
    
    train_ids = sorted_ids[valid_count + test_count:]
    
    shuffle_ids = torch.randperm(len(selected_valid_test_ids))
    
    valid_ids =  selected_valid_test_ids[shuffle_ids[0:valid_count]]
    
    test_ids =  selected_valid_test_ids[shuffle_ids[valid_count:test_count + valid_count]]
    
    train_dataset = MyDataset(full_dataset.data[train_ids], full_dataset.labels[train_ids])
    
    valid_dataset = MyDataset(full_dataset.data[valid_ids], full_dataset.labels[valid_ids])
    
    test_dataset = MyDataset(full_dataset.data[test_ids], full_dataset.labels[test_ids])
    
    new_train_img_names = full_img_names.iloc[train_ids]
    
    new_valid_img_names = full_img_names.iloc[valid_ids]
    
    new_test_img_names = full_img_names.iloc[test_ids]
    
    
    torch.save(train_dataset, output_dir + '/' + transformed_train_dataset_file_name + '_new')
    
    torch.save(valid_dataset, output_dir + '/' + transformed_val_dataset_file_name + '_new')
    
    torch.save(test_dataset, output_dir + '/' + transformed_test_dataset_file_name + '_new')
    
    return train_dataset, valid_dataset, test_dataset, new_train_img_names, new_valid_img_names, new_test_img_names
     
    
    

    
if __name__ == '__main__':
    
    download_corpus()
    
    parser = ArgumentParser()
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'crowdsourced_dataset/Fashion/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    parser.add_argument('--compare_tars', action = 'store_true', help="flag to compare tars")
    
    parser.add_argument('--lr', type = float, default = 0.1, help="learning rate")
    
    parser.add_argument('--tlr', type = float, default = 0.01, help="learning rate")
    
    parser.add_argument('--bz', type = int, default = 1000, help="batch size")
    
    parser.add_argument('--wd', type = float, default = 0.001, help="l2 norm regularization coefficient")
    
    parser.add_argument('--clean_ratio', type = float, default = 0.1, help="l2 norm regularization coefficient")
    
    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('--epochs', type = int, default=50, help="epoch count for training")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")
    # parser.add_argument('--dataset', action = 'store_true', help="flag to compare tars")
    
    args = parser.parse_args()
    
    clean_sample_ratio = args.clean_ratio
    
    
    if not args.GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    args.device = device
    
    
    data_prepare = Data_preparer()
    
    
    dataset_name = 'fashion'
    
#     full_output_dir = os.path.join(args.output_dir, dataset_name)
#         
#     if not os.path.exists(full_output_dir):
#         os.makedirs(full_output_dir)
    
    # num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    #
    # args.num_class = num_class
    
    
    # obtain_data_function = getattr(sys.modules[__name__], 'obtain_' + args.dataset.lower() + '_examples')
    
    
    
    
    annotated_label_file = os.path.join(os.path.join(args.output_dir, 'Annotations'), fashion_csv_file_name)

    train_DL, valid_DL, test_DL, train_sample_names, valid_sample_names, test_sample_names = data_prepare.prepare_fashion(args.output_dir, bz=200, load = True)

    train_dataset, valid_dataset, test_dataset, train_sample_names, valid_sample_names, test_sample_names = select_valid_test_dataset4(args, args.output_dir, train_sample_names, valid_sample_names, test_sample_names)

    print('input sample names statistics::', len(train_sample_names), len(valid_sample_names), len(test_sample_names))
    
    print('input DL statistics::', train_dataset.lenth, valid_dataset.lenth, test_dataset.lenth)
    
    
    if True:
        train_sample_names, valid_sample_names, test_sample_names, train_prob_labels, train_annotated_labels, valid_labels, test_labels = load_labels_per_image(clean_sample_ratio, annotated_label_file, args.output_dir, train_sample_names, valid_sample_names, test_sample_names, compare_tars = args.compare_tars)
        
        torch.save(train_prob_labels, os.path.join(args.output_dir, 'train_prob_labels'))
        
        torch.save(train_annotated_labels, os.path.join(args.output_dir, 'train_annotated_labels'))
        
        torch.save(valid_labels, os.path.join(args.output_dir, 'valid_labels'))
        
        torch.save(test_labels, os.path.join(args.output_dir, 'test_labels'))
        
        torch.save(train_sample_names, os.path.join(args.output_dir, 'train_names_new'))
        
        torch.save(valid_sample_names, os.path.join(args.output_dir, 'valid_names_new'))
        
        torch.save(test_sample_names, os.path.join(args.output_dir, 'test_names_new'))
    
        print('output statistics::')
        
        print(len(train_sample_names), len(valid_sample_names), len(test_sample_names))
        
        print(train_prob_labels.shape, train_annotated_labels.shape, valid_labels.shape, test_labels.shape)
    
    
    # else:
    #     clean_sample_ids = torch.load(os.path.join(args.output_dir, 'clean_sample_ids'))
    #
    #     noisy_sample_ids = torch.load(os.path.join(args.output_dir, 'noisy_sample_ids'))
    #
    #
    #     # twitter_without_gt_tensor = pd.read_csv(os.path.join(args.output_dir, 'twitter_without_gt.csv'))
    #
    #     training_ids_with_derived_ground_truth = torch.load(os.path.join(args.output_dir, 'training_ids_with_derived_ground_truth'))
    #
    #     training_ids_without_derived_ground_truth = torch.load(os.path.join(args.output_dir, 'training_ids_without_derived_ground_truth'))
    #
    #     train_features = torch.load(os.path.join(args.output_dir, 'train_features'))
    #
    #     test_features = torch.load(os.path.join(args.output_dir, 'test_features'))
    #
    #     train_id = 4420
    #
    #     exp_label = 1
    #
    #     selected_tuple = twitter_without_gt_tensor.iloc[training_ids_without_derived_ground_truth[train_id-len(training_ids_with_derived_ground_truth)]]
    #
    #     selected_tuple_text = selected_tuple['text']
    #
    #     selected_tuple_label = selected_tuple['derived_labels']
    #
    #     valid_test_feature_list, twitter_with_gt0 = get_valid_test_features(full_twitter_csv_file_name)
    #
    #     twitter_with_gt0['airline_sentiment_gold'] = twitter_with_gt0['airline_sentiment_gold'].apply(replace_rates_to_numeric_values) 
    #
    #     min_id, _ = get_closest_samples(valid_test_feature_list, train_features[train_id], twitter_with_gt0, exp_label=exp_label)
    #
    #
    #     selected_tuple_text_embedding_list = replace_twitter_text_with_embeddings(selected_tuple_text)
    #
    #
    #     selected_tuple_text_embedding = align_tensor_embedding_list([selected_tuple_text_embedding_list])
    #
    #     print(selected_tuple_text, selected_tuple_label)
    #
    #     print(twitter_with_gt0.iloc[min_id]['text'], twitter_with_gt0.iloc[min_id]['airline_sentiment_gold'])
    #
    #     print('here')
    
    # 
    #
    # # read_twitter_csv(full_twitter_csv_file_name, default_output_dir)
    # label_file = load_labels_per_image(annotated_label_file)



