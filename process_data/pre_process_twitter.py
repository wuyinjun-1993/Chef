'''
Created on Apr 23, 2021

'''

# import pickle
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

from scipy import stats

import re

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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')

from gensim.test.utils import datapath



try:
    from utils.utils import *
    from process_data.pre_process_fact import *
    from interactive_weak_supervision.iws_utils import *
    from interactive_weak_supervision.iws import InteractiveWeakSupervision
    from interactive_weak_supervision.torchmodels import TorchMLP
except ImportError:
    from utils import *
    from pre_process_fact import *
    from iws_utils import *
    from iws import InteractiveWeakSupervision
    from torchmodels import TorchMLP

twitter_csv_file_names = 'Airline-Full-Non-Ag-DFE-Sentiment.csv'

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')   # Download vocabulary from S3 and cache.


# def initialize_labels_df(qids, raters):
    # init_data = np.ones((len(qids), len(raters)))*(-1)
    #
    # df = pd.DataFrame(data = init_data, index = qids, columns = raters)
    #
    #
    # return df


def set_crowdsourced_label_df(crowdsourced_label_df, all_worker_ids, df):
    for id in all_worker_ids:
        
        curr_worker_rated_rows = df.loc[df['_worker_id'] == id][['_unit_id', 'airline_sentiment']]
        
        # curr_worker_rated_row_unit_id_worker_id_rates = 
        
        crowdsourced_label_df.loc[curr_worker_rated_rows['_unit_id'],id] = list(curr_worker_rated_rows['airline_sentiment'])
    
        print('here')
    

def replace_rates_to_numeric_values(x):
    if x == 'negative':
        return 0
    else:
        if x == 'positive':
            return 1
        else:
            if x == 'neutral':
                return 2
            else:
                return -1
        
def replace_twitter_text_with_embeddings(text):
    text = text.replace('@', '')
    
    curr_embedding = convert_str_to_embeddings(tokenizer, model, text)
    
    return curr_embedding
    
def remove_twitter_text_special_char(text):
    # text = text.replace('@', '')
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    # curr_embedding = convert_str_to_embeddings(tokenizer, model, text)
    
    return text    

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
    
    svd = TruncatedSVD(n_components=min(50, LFs.shape[0] - 1), n_iter=20, random_state=42)
    LFfeatures = svd.fit_transform(LFs.T).astype(np.float32)
    
    
    positive_sentiment_words = ['amazing', 'great','awesome', 'thank']
    negative_sentimen_words = ['bad', 'shit', 'terrible','frustrating']
    # neutral_sentimatent_words = ['the', 'will']
    start_idxs = []
    halfway= int(len(lf_descriptions)/2)# we generated positive LFs first, then negative
    # iterate over positive LFs
    for i,desc in enumerate(lf_descriptions[:halfway]):
        for word in positive_sentiment_words:
            if word in desc:
                start_idxs.append(i)
                if len(start_idxs)==4:
                    break
        if len(start_idxs)==4:
            break
    
    # iterate over negative LFs
    for i,desc in enumerate(lf_descriptions[halfway:2*halfway]):
        idx = halfway+i
        for word in negative_sentimen_words:
            if word in desc:
                start_idxs.append(idx)
                if len(start_idxs)==8:
                    break
        if len(start_idxs)==8:
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
    
    print('lf sets lenth::', len(LFsets))
    
    downstream_results = train_end_classifier(Xsvd,Xtestsvd,IWSsession,LFsets,gap=20,verbose=True)
    
    return downstream_results[1][num_iter-1]

def obtain_derived_ground_truth(sentiment_list):

    sentiment_array = np.array(sentiment_list)
    
    m = stats.mode(sentiment_array)
    
    return m[0][0]
    # twitter_without_gt['_unit_id']


# def partition_valid_test_df(twitter_with_gt0):
    # all_ids = list(range(twitter_with_gt0.shape[0]))
    #
    # np.random.shuffle(all_ids)
    #
    # valid_count = 

def print_text(df):
    for k in range(df.shape[0]):
        print(k)
        print('text::', df.iloc[k]['text'])
        print('label::', df.iloc[k]['airline_sentiment_gold'])


def get_valid_test_features(file_name):
    df = pd.read_csv(file_name)
    
    twitter_with_gt0 = df[(df['_golden'] == True) & (~(df['airline_sentiment_gold'] == 'neutral'))][['_unit_id','airline_sentiment_gold','text']].drop_duplicates()

    twitter_with_gt0['text_embedding'] = list(twitter_with_gt0['text'].apply(replace_twitter_text_with_embeddings))
    
    return align_tensor_embedding_list(list(twitter_with_gt0['text_embedding'])), twitter_with_gt0

def get_closest_samples(valid_test_feature_list, curr_train_feature, twitter_with_gt0, exp_label):
    
    min_distance = None
    
    min_id = -1
    
    for k in range(valid_test_feature_list.shape[0]):
        
        curr_test_feature = valid_test_feature_list[k]
        
        if exp_label is not None:
        
            if twitter_with_gt0['airline_sentiment_gold'].iloc[k] != exp_label:
                continue
        
        distance = -torch.dot(curr_test_feature.view(-1)[-768:], curr_train_feature.view(-1)[-768:])/(torch.norm(curr_test_feature[-768:])*torch.norm(curr_train_feature[-768:]))
        
        # distance = -torch.dot(curr_test_feature.view(-1), curr_train_feature.view(-1))/(torch.norm(curr_test_feature)*torch.norm(curr_train_feature))

        # distance = torch.norm(curr_test_feature.view(-1) - curr_train_feature.view(-1))
        
        if min_distance is None:
            min_distance = distance
            min_id = k
        else:
            if distance < min_distance:
                min_distance = distance
                min_id = k
         
    return min_id, min_distance

def read_twitter_csv(file_name, output_dir, training_count_with_gt_ratio = 0):
    
    # rid = 0
    
    
    df = pd.read_csv(file_name)
    
    df['airline_sentiment'] = df['airline_sentiment'].apply(replace_rates_to_numeric_values)
    
    df['airline_sentiment_gold'] = df['airline_sentiment_gold'].apply(replace_rates_to_numeric_values) 
    
    df['text'] = df['text'].apply(remove_twitter_text_special_char)

    twitter_with_gt0 = df[df['_golden'] == True][['_unit_id','airline_sentiment_gold','text']].drop_duplicates()
    
    twitter_with_gt0 = twitter_with_gt0[(twitter_with_gt0['airline_sentiment_gold'] == 1) | (twitter_with_gt0['airline_sentiment_gold'] == 0)]
    
    all_twitter_ids = list(range(twitter_with_gt0.shape[0]))
    
    all_twitter_ids_positive = list(np.nonzero(list(twitter_with_gt0['airline_sentiment_gold'] == 1))[0])
    
    all_twitter_ids_negative = list(np.nonzero(list(twitter_with_gt0['airline_sentiment_gold'] == 0))[0])
    
    np.random.shuffle(all_twitter_ids)
    
    np.random.shuffle(all_twitter_ids_positive)
    
    np.random.shuffle(all_twitter_ids_negative)
    
    valid_count = int(len(all_twitter_ids)*0.5)
    
    valid_pos_count = int(len(all_twitter_ids_positive)*0.5)
    
    valid_neg_count = int(len(all_twitter_ids_negative)*0.5)
    
    valid_ids_pos = all_twitter_ids_positive[0:valid_pos_count]
    
    valid_ids_neg = all_twitter_ids_negative[0:valid_neg_count]
    
    valid_ids = []
    
    valid_ids.extend(valid_ids_pos)
    
    valid_ids.extend(valid_ids_neg)
    
    test_ids = list(set(all_twitter_ids).difference(set(valid_ids)))
    
    valid_df = twitter_with_gt0.iloc[valid_ids]
    
    test_df = twitter_with_gt0.iloc[test_ids]
    
    print('valid data frame::')
    
    print_text(valid_df)
    
    print('test data frame::')
    
    print_text(test_df)
    
    
    torch.save(torch.tensor(valid_ids), os.path.join(output_dir, 'valid_ids'))
    
    torch.save(torch.tensor(test_ids), os.path.join(output_dir, 'test_ids'))
    
    twitter_without_gt = df[df['_golden'] == False][['_unit_id','text']].drop_duplicates()
    
    annotated_labels = df.loc[df['_golden'] == False].groupby('_unit_id')['airline_sentiment'].apply(list)
    
    # annotated_labels.rename(columns = {'airline_sentiment' : 'airline_sentiment_list'})
    
    twitter_without_gt = twitter_without_gt.merge(annotated_labels, left_on = '_unit_id', right_index = True)
    
    twitter_without_gt['derived_labels'] = twitter_without_gt['airline_sentiment'].apply(obtain_derived_ground_truth)
    
    twitter_without_gt = twitter_without_gt[(twitter_without_gt['derived_labels'] == 1) | (twitter_without_gt['derived_labels'] == 0)]
    
    # id_list = list(range(200))
    #
    # twitter_without_gt = twitter_without_gt.iloc[id_list]
    
    training_ids = list(range(twitter_without_gt.shape[0]))
    
    np.random.shuffle(training_ids) 
    
    training_count_with_derived_ground_truth = int(len(training_ids)*training_count_with_gt_ratio)
    
    # valid_count = int(len(training_ids)*valid_count_with_gt_ratio)
    
    ids_with_derived_ground_truth = training_ids[0:training_count_with_derived_ground_truth]
    
    training_ids_with_derived_ground_truth = training_ids[0:training_count_with_derived_ground_truth]
    
    # valid_ids_with_derived_ground_truth = training_ids[training_count_with_derived_ground_truth:valid_count + training_count_with_derived_ground_truth]
    
    training_ids_without_derived_ground_truth = training_ids[training_count_with_derived_ground_truth:]
    
    df_with_derived_gt = twitter_without_gt.iloc[ids_with_derived_ground_truth]
    
    train_df_with_derived_gt = twitter_without_gt.iloc[training_ids_with_derived_ground_truth]
    
    # valid_df_with_derived_gt = twitter_without_gt.iloc[valid_ids_with_derived_ground_truth]
    
    train_df_without_derived_gt = twitter_without_gt.iloc[training_ids_without_derived_ground_truth]
    
    
    full_df_with_derived_gt = pd.concat([train_df_with_derived_gt, valid_df])[['_unit_id', 'derived_labels', 'text']].rename(columns={'derived_labels':'airline_sentiment_gold'})
    
    # twitter_with_gt = twitter_with_gt0[(twitter_with_gt0['airline_sentiment_gold'] == 0) | (twitter_with_gt0['airline_sentiment_gold'] == 1)]
    
    text_prob_labels = derive_prob_labels(list(full_df_with_derived_gt['text']), list(full_df_with_derived_gt['airline_sentiment_gold']), train_df_without_derived_gt['text'], output_dir)

    # twitter_without_gt['']

    train_valid_labels = torch.zeros([len(training_ids),2])
    
    if df_with_derived_gt['derived_labels'].shape[0] > 0:
        train_valid_labels[ids_with_derived_ground_truth] = onehot(torch.tensor(list(df_with_derived_gt['derived_labels'])), 2)
    
    train_valid_labels[training_ids_without_derived_ground_truth,1] = torch.from_numpy(text_prob_labels)
    
    train_valid_labels[training_ids_without_derived_ground_truth,0] = 1 - torch.from_numpy(text_prob_labels)
    
    valid_labels = torch.tensor(list(valid_df['airline_sentiment_gold']))
    # valid_labels = train_valid_labels[valid_ids_with_derived_ground_truth]
    
    # train_labels =  train_valid_labels[valid_ids_with_derived_ground_truth]
        
    # twitter_text_list = df[['_unit_id','text']].drop_duplicates()
    
    # max_worker_count_per_twit = max(list(df.loc[df['_golden'] == False].groupby('_unit_id').size()))
    
    # all_worker_ids = list(df.loc[df['_golden'] == False,'_worker_id'].unique())
    
    # all_twitter_ids = list(df.loc[df['_golden'] == False,'_unit_id'].unique())
    #
    # crowdsourced_label_df = initialize_labels_df(all_twitter_ids, all_worker_ids)
    #
    # set_crowdsourced_label_df(crowdsourced_label_df, all_worker_ids, df[df['_golden'] == False])
    # crowdsourced_lables = df[df['_golden'] == False].groupby(['_unit_id'])[['airline_sentiment']].apply(list)
    
    train_df_without_derived_gt['text_embedding'] = list(train_df_without_derived_gt['text'].apply(replace_twitter_text_with_embeddings))
    
    # valid_df_with_derived_gt['text_embedding'] = list(valid_df_with_derived_gt['text'].apply(replace_twitter_text_with_embeddings))
    
    train_df_with_derived_gt['text_embedding'] = list(train_df_with_derived_gt['text'].apply(replace_twitter_text_with_embeddings))
    
    valid_df['text_embedding'] = list(valid_df['text'].apply(replace_twitter_text_with_embeddings))
    
    test_df['text_embedding'] = list(test_df['text'].apply(replace_twitter_text_with_embeddings))
    
    full_train_df = pd.concat([train_df_with_derived_gt, train_df_without_derived_gt])
    
    full_train_labels = torch.cat([train_valid_labels[training_ids_with_derived_ground_truth], train_valid_labels[training_ids_without_derived_ground_truth]], dim = 0)
    
    
    full_train_annotated_labels = list(train_df_with_derived_gt['airline_sentiment'])
    
    full_train_annotated_labels.extend(list(train_df_without_derived_gt['airline_sentiment']))
    # full_train_labels =  torch.tensor(list(train_valid_labels[training_ids_with_derived_ground_truth]).extend(list(train_valid_labels[training_ids_without_derived_ground_truth])))
    
    
    # twitter_without_gt['text_embedding'] = twitter_without_gt['text'].apply(replace_twitter_text_with_embeddings)
    
    test_label_tensor = torch.tensor(list(test_df['airline_sentiment_gold']))
    
    
    
    # valid_labels = torch.argmax(valid_labels, dim = 1)
    
    train_origin_labels = torch.cat([torch.tensor(list(twitter_without_gt.iloc[training_ids_with_derived_ground_truth]['derived_labels'])), torch.tensor(list(twitter_without_gt.iloc[training_ids_without_derived_ground_truth]['derived_labels']))], dim = 0)
    
    clean_sample_ids = torch.tensor(list(range(len(training_ids_with_derived_ground_truth))))
    
    noise_sample_ids = torch.tensor(list(range(len(training_ids_with_derived_ground_truth), len(train_origin_labels))))
    
    torch.save(clean_sample_ids, os.path.join(output_dir, 'clean_sample_ids'))
    
    torch.save(training_ids_with_derived_ground_truth, os.path.join(output_dir, 'training_ids_with_derived_ground_truth'))
    
    torch.save(training_ids_without_derived_ground_truth, os.path.join(output_dir, 'training_ids_without_derived_ground_truth'))
    
    
    
    twitter_without_gt.to_csv(os.path.join(output_dir, 'twitter_without_gt.csv'))
    
    torch.save(noise_sample_ids, os.path.join(output_dir, 'noisy_sample_ids'))
    
    print('train origin labels::', train_origin_labels)
    
    print(len(full_train_labels), len(train_origin_labels), len(full_train_annotated_labels))
    # train_origin_labels = twitter_without_gt['derived_labels']
    
    train_features = align_tensor_embedding_list(list(full_train_df['text_embedding']))
    
    valid_features = align_tensor_embedding_list(list(valid_df['text_embedding']))
    
    test_features = align_tensor_embedding_list(list(test_df['text_embedding']))
    
    return train_features, full_train_labels, train_origin_labels, full_train_annotated_labels, valid_features, valid_labels, test_features, test_label_tensor
    # return torch.stack(list(twitter_without_gt['text_embedding']),0), list(annotated_labels), torch.tensor(text_prob_labels), torch.stack(list(twitter_with_gt0['text_embedding']),0), torch.tensor(list(twitter_with_gt0['airline_sentiment_gold']))
        
    # embedding1 = convert_str_to_embeddings(tokenizer, model, ' '.join([sub_text, property_str, obj_text]))
    
    
    
    # true_labels = df[df['_golden'] == True][['_unit_id', 'airline_sentiment_gold']]
    #
    # print('here')
    
    # with open(file_name, newline='') as csvfile:
        # spamreader = csv.reader(csvfile, delimiter=',')
        # for row in spamreader:
        #
            # if rid <= 0:
                # rid += 1
                # continue
                #
            # qid = int(row[0])
            
            
          


if __name__ == '__main__':
    
    download_corpus()
    
    parser = ArgumentParser()
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'crowdsourced_dataset/twitter/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    args = parser.parse_args()
    
    # clean_sample_ids = torch.load(os.path.join(args.output_dir, 'clean_sample_ids'))
    
    full_twitter_csv_file_name = os.path.join(args.output_dir, twitter_csv_file_names)
    
    # features_without_gt, annotated_labels_without_gt, prob_labels_without_gt, features_with_gt, labels_with_gt = read_twitter_csv(full_twitter_csv_file_name, default_output_dir)
    
    if True:
    
        train_origin_features, train_labels, train_origin_labels, full_train_annotated_labels, valid_origin_features, valid_labels, test_origin_features, test_labels  = read_twitter_csv(full_twitter_csv_file_name, default_output_dir)
        
        train_features, valid_features, test_features = train_origin_features.view(train_origin_features.shape[0], -1), valid_origin_features.view(valid_origin_features.shape[0], -1), test_origin_features.view(test_origin_features.shape[0], -1)
        
        print(train_features.shape, valid_features.shape, test_features.shape)
        
        print(train_origin_features.shape, valid_origin_features.shape, test_origin_features.shape)
        
        torch.save(train_features, os.path.join(args.output_dir, 'train_features'))
        
        torch.save(train_origin_features, os.path.join(args.output_dir, 'train_origin_features'))
        
        torch.save(valid_origin_features, os.path.join(args.output_dir, 'valid_origin_features'))
        
        torch.save(test_origin_features, os.path.join(args.output_dir, 'test_origin_features'))
        
        torch.save(train_labels, os.path.join(args.output_dir, 'train_labels'))
        
        torch.save(train_origin_labels, os.path.join(args.output_dir, 'train_origin_labels'))
        
        torch.save(full_train_annotated_labels, os.path.join(args.output_dir, 'full_train_annotated_labels'))
        
        torch.save(valid_features, os.path.join(args.output_dir, 'valid_features'))
        
        torch.save(valid_labels, os.path.join(args.output_dir, 'valid_labels'))
        
        torch.save(test_features, os.path.join(args.output_dir, 'test_features'))
        
        torch.save(test_labels, os.path.join(args.output_dir, 'test_labels'))
    
    else:
        
        clean_sample_ids = torch.load(os.path.join(args.output_dir, 'clean_sample_ids'))
        
        noisy_sample_ids = torch.load(os.path.join(args.output_dir, 'noisy_sample_ids'))
        
        twitter_without_gt_tensor = pd.read_csv(os.path.join(args.output_dir, 'twitter_without_gt.csv'))
        
        training_ids_with_derived_ground_truth = torch.load(os.path.join(args.output_dir, 'training_ids_with_derived_ground_truth'))
    
        training_ids_without_derived_ground_truth = torch.load(os.path.join(args.output_dir, 'training_ids_without_derived_ground_truth'))
        
        train_features = torch.load(os.path.join(args.output_dir, 'train_features'))
        
        test_features = torch.load(os.path.join(args.output_dir, 'test_features'))
        
        train_id = 4420
        
        exp_label = 1
        
        selected_tuple = twitter_without_gt_tensor.iloc[training_ids_without_derived_ground_truth[train_id-len(training_ids_with_derived_ground_truth)]]
        
        selected_tuple_text = selected_tuple['text']
        
        selected_tuple_label = selected_tuple['derived_labels']
        
        valid_test_feature_list, twitter_with_gt0 = get_valid_test_features(full_twitter_csv_file_name)
        
        twitter_with_gt0['airline_sentiment_gold'] = twitter_with_gt0['airline_sentiment_gold'].apply(replace_rates_to_numeric_values) 
        
        min_id, _ = get_closest_samples(valid_test_feature_list, train_features[train_id], twitter_with_gt0, exp_label=exp_label)
        
        
        selected_tuple_text_embedding_list = replace_twitter_text_with_embeddings(selected_tuple_text)
        
        
        selected_tuple_text_embedding = align_tensor_embedding_list([selected_tuple_text_embedding_list])
        
        print(selected_tuple_text, selected_tuple_label)
        
        print(twitter_with_gt0.iloc[min_id]['text'], twitter_with_gt0.iloc[min_id]['airline_sentiment_gold'])
        
        print('here')
        
        
        # if len(clean_sample_ids) <= 0:
        #     train_labels = torch.load(os.path.join(args.output_dir, 'train_labels'))
        #
        #     train_origin_labels = torch.load(os.path.join(args.output_dir, 'train_origin_labels'))
        #
        #     all_train_ids = torch.randperm(train_origin_labels.shape[0])
        #
        #     random_clean_count = int(train_origin_labels.shape[0]*0.01)
        #
        #     random_clean_ids = all_train_ids[0:random_clean_count]
        #
        #     train_labels[random_clean_ids] = onehot(train_origin_labels[random_clean_ids].type(torch.long), 2)
        #
        #     torch.save(random_clean_ids, os.path.join(args.output_dir, 'clean_sample_ids'))
        #
        #     torch.save(train_labels, os.path.join(args.output_dir, 'train_labels'))
        #
        #
        #     noisy_ids = set(list(range(train_origin_labels.shape[0]))).difference(set(random_clean_ids))
        #
        #     torch.save(noisy_ids, os.path.join(args.output_dir, 'noisy_sample_ids'))
            # torch.save(train_origin_labels, )
        
    # torch.save(features_without_gt, os.path.join(args.output_dir, 'features_without_gt'))
    #
    # torch.save(annotated_labels_without_gt, os.path.join(args.output_dir, 'annotated_labels_without_gt'))
    #
    # torch.save(prob_labels_without_gt, os.path.join(args.output_dir, 'prob_labels_without_gt'))
    #
    # torch.save(features_with_gt, os.path.join(args.output_dir, 'features_with_gt'))
    #
    # torch.save(labels_with_gt, os.path.join(args.output_dir, 'labels_with_gt'))
    
    
    