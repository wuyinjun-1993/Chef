'''
Created on Dec 18, 2020

'''

import numpy as np

import torch

import torch.nn.functional as F
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/pytorch_influence_functions')


from sklearn.metrics import roc_auc_score
from sklearn import metrics

from torch.utils.data import Dataset, DataLoader

from snorkel.preprocess import preprocessor
from textblob import TextBlob

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel


import pandas as pd
import csv
import re

try:
    from utils.utils import *
    from models.utils_real import *
    from train import *

except ImportError:
    from utils import *
    from utils_real import *
    from train import *
    
ABSTAIN = -1
HAM = 0
SPAM = 1

parser = '::'

folder_name = '/home/wuyinjun/workspace/ML_provenance_application/.gitignore/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0'


columns = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']

def get_train_valid_test_split():
    
    file_name = folder_name + '/mimic-cxr-2.0.0-split.csv'
    
    split_info = pd.read_csv(file_name)
    
    train_ids = split_info.loc[split_info.split == 'train']
    
    valid_ids = split_info.loc[split_info.split == 'validate']
    
    test_ids = split_info.loc[split_info.split == 'test']
    
    return train_ids, valid_ids, test_ids

def join_id_df_with_label_df(train_ids, label_file):
    
    file_name = folder_name + '/' + label_file
    
    label_df = pd.read_csv(file_name)
    
    result_df = pd.merge(label_df, train_ids, on = ['subject_id', 'study_id'])
    
    return result_df

def valid_label_one_rule(joined_valid_labeled_df, cln):
    return ((joined_valid_labeled_df[cln + '_1'] == 1) & (joined_valid_labeled_df[cln + '_2'] == 1)) | ((joined_valid_labeled_df[cln + '_1'] == 1) & (joined_valid_labeled_df[cln + '_2'].isnull())) | ((joined_valid_labeled_df[cln + '_1'].isnull()) & (joined_valid_labeled_df[cln + '_2'] == 1)) 


def valid_label_zero_rule(joined_valid_labeled_df, cln):
    return ((joined_valid_labeled_df[cln + '_1'] == 0) & (joined_valid_labeled_df[cln + '_2'] == 0)) | ((joined_valid_labeled_df[cln + '_1'] == 0) & (joined_valid_labeled_df[cln + '_2'].isnull())) | ((joined_valid_labeled_df[cln + '_1'].isnull()) & (joined_valid_labeled_df[cln + '_2'] == 0)) 


def valid_label_minus_one_rule(joined_valid_labeled_df, cln):
    return ((joined_valid_labeled_df[cln + '_1'] == -1)) | ((joined_valid_labeled_df[cln + '_2'] == -1)) | ((joined_valid_labeled_df[cln + '_1'] == 1) & (joined_valid_labeled_df[cln + '_2'] == 0)) | ((joined_valid_labeled_df[cln + '_1'] == 0) & (joined_valid_labeled_df[cln + '_2'] == 1)) 

def valid_label_nan_rule(joined_valid_labeled_df, cln):
    return ((joined_valid_labeled_df[cln + '_1'].isnull())) & (joined_valid_labeled_df[cln + '_2'].isnull()) 



def valid_give_label(valid_df, joined_valid_labeled_df, cln):

    valid_df_copy = valid_df.copy()

    one_id = joined_valid_labeled_df.loc[valid_label_one_rule(joined_valid_labeled_df, cln),['id']]
    
    zero_id = joined_valid_labeled_df.loc[valid_label_zero_rule(joined_valid_labeled_df, cln),['id']]
    
    minus_one_id = joined_valid_labeled_df.loc[valid_label_minus_one_rule(joined_valid_labeled_df, cln),['id']]
    
    nan_id = joined_valid_labeled_df.loc[valid_label_nan_rule(joined_valid_labeled_df, cln),['id']]
    
    valid_df_copy.loc[valid_df_copy['id'].isin(one_id.values.reshape(-1)),[cln]] = 1
    
    valid_df_copy.loc[valid_df_copy['id'].isin(zero_id.values.reshape(-1)),[cln]] = 0
    
    valid_df_copy.loc[valid_df_copy['id'].isin(minus_one_id.values.reshape(-1)),[cln]] = -1
    
#     valid_df.loc[valid_df['id'].isin(nan_id.values.reshape[-1]),[cln]] = np.nan

    print(one_id.shape[0] + zero_id.shape[0] + minus_one_id.shape[0] + nan_id.shape[0])
    
    print(valid_df_copy.shape[0])

    
    return valid_df_copy



def filter_empty_records_valid_test(valid_labeled_df1,valid_labeled_df2, tag):
    
    valid_df = valid_labeled_df1.copy()
    
    joined_valid_labeled_df = pd.merge(valid_labeled_df1, valid_labeled_df2, on = ['id'], suffixes=('_1', '_2'))
    
    for col in columns:
        valid_df_copy = valid_give_label(valid_df, joined_valid_labeled_df, col)
                
        valid_df_copy = valid_df_copy.loc[(valid_df_copy[col] == 0) | (valid_df_copy[col] == 1),['id', col]]
        
        print('df unique values::', np.unique(valid_df_copy[col].values))
        
        print('final ' + tag + ' size', valid_df_copy.shape[0])
        
        valid_df_copy.to_csv(folder_name + '/' + tag + '_' + col + '.csv')
         
        
#         valid_df1_label_1 = valid_labeled_df1[valid_labeled_df1[col] == 1]
#         valid_df1_label_0 = valid_labeled_df1[valid_labeled_df1[col] == 0]
#         valid_df1_label_minus_1 = valid_labeled_df1[valid_labeled_df1[col] == -1]
#         valid_df1_label_none = valid_labeled_df1[valid_labeled_df1[col].isnull()]
#         
#         
#         valid_df2_label_1 = valid_labeled_df1[valid_labeled_df2[col] == 1]
#         valid_df2_label_0 = valid_labeled_df1[valid_labeled_df2[col] == 0]
#         valid_df2_label_minus_1 = valid_labeled_df1[valid_labeled_df2[col] == -1]
#         valid_df2_label_none = valid_labeled_df1[valid_labeled_df2[col].isnull()]
#         
# #         valid_df.iloc[ & (valid_labeled_df2[col] == 1)][col] = 1
#         valid_df.iloc[(valid_labeled_df1[col] == 0) & (valid_labeled_df2[col] == 0)][col] = 0
#         valid_df.iloc[(valid_labeled_df1[col] == -1) & (valid_labeled_df2[col] == -1)][col] = -1





def obtain_prob_labels_training_origin(train_labeled_df1, train_labeled_df2):
    
    ids = train_labeled_df1['id'].values
    
    sub_id = train_labeled_df1['subject_id'].values
    
    study_id = train_labeled_df1['study_id'].values
    
    dicom_id = train_labeled_df1['dicom_id'].values
    
    
    
    for cln in columns:
    
        
        
        sub_id = train_labeled_df1.loc[:,['id']].values
    
#         train_labeled_df1.loc[train_labeled_df1[cln].isnull(), [cln]] = ABSTAIN
#         
#         train_labeled_df2.loc[train_labeled_df2[cln].isnull(), [cln]] = ABSTAIN
    
        label1 = train_labeled_df1.loc[:, [cln]].values
        
#         label1[np.isnan(label1)] = ABSTAIN
        
        label2 = train_labeled_df2.loc[:,[cln]].values

        label = np.copy(label1)
        
        label[(label1 == 1) & np.isnan(label2)] = 1
        
        label[(label2 == 1) & np.isnan(label1)] = 1
        
        label[(label1 == 0) & np.isnan(label2)] = 0
        
        label[(label2 == 0) & np.isnan(label1)] = 0
        
        label[(label1 == -1) & np.isnan(label2)] = -1
        
        label[(label2 == -1) & np.isnan(label1)] = -1
        
        label[(label1 == -1) & (label2 == 0)] = -1
        
        label[(label1 == -1) & (label2 == 1)] = -1
        
        label[(label2 == -1) & (label1 == 0)] = -1
        
        label[(label2 == -1) & (label1 == 1)] = -1
        
        labeled_ids = ((label == 1) | (label == 0) | (label == -1))
        
        label = label[((label == 1) | (label == 0) | (label == -1))]
        
#         label[label == -1] = 0
        
        selected_sub_ids = sub_id[labeled_ids]
        
        df = pd.DataFrame({'id': selected_sub_ids, 'label': label})
        
        print(df.shape)
        
        print('train data file name::', 'train_' + cln  + '.csv')
        
        df.to_csv(folder_name + '/train_' + cln  + '_origin.csv')

def obtain_prob_labels_training(train_labeled_df1, train_labeled_df2):
    
    ids = train_labeled_df1['id'].values
    
    sub_id = train_labeled_df1['subject_id'].values
    
    study_id = train_labeled_df1['study_id'].values
    
    dicom_id = train_labeled_df1['dicom_id'].values
    
    
    
    for cln in columns:
    
        
        
        sub_id = train_labeled_df1.loc[:,['id']].values
    
#         train_labeled_df1.loc[train_labeled_df1[cln].isnull(), [cln]] = ABSTAIN
#         
#         train_labeled_df2.loc[train_labeled_df2[cln].isnull(), [cln]] = ABSTAIN
    
        label1 = train_labeled_df1.loc[:, [cln]].values
        
#         label1[np.isnan(label1)] = ABSTAIN
        
        label2 = train_labeled_df2.loc[:,[cln]].values

        label = np.copy(label1)
        
        label[(label1 == 1) & np.isnan(label2)] = 1
        
        label[(label2 == 1) & np.isnan(label1)] = 1
        
        label[(label1 == 0) & np.isnan(label2)] = 0
        
        label[(label2 == 0) & np.isnan(label1)] = 0
        
        label[(label1 == -1) & np.isnan(label2)] = -1
        
        label[(label2 == -1) & np.isnan(label1)] = -1
        
        label[(label1 == -1) & (label2 == 0)] = -1
        
        label[(label1 == -1) & (label2 == 1)] = -1
        
        label[(label2 == -1) & (label1 == 0)] = -1
        
        label[(label2 == -1) & (label1 == 1)] = -1
        
        labeled_ids = ((label == 1) | (label == 0) | (label == -1))
        
        label = label[((label == 1) | (label == 0) | (label == -1))]
        
        label[label == -1] = 0
        
        selected_sub_ids = sub_id[labeled_ids]
        
        df = pd.DataFrame({'id': selected_sub_ids, 'label': label})
        
        print(df.shape)
        
        print('train data file name::', 'train_' + cln  + '.csv')
        
        df.to_csv(folder_name + '/train_' + cln  + '.csv')
        
#         label2[label2.isnull()] = ABSTAIN
        
#         all_labels = np.zeros((label1.shape[0], 2))
#         
#         all_labels[:,0] = label1.reshape(-1)
#         
#         all_labels[:,1] = label2.reshape(-1)
#         
#         label_model = LabelModel(cardinality=2, verbose=True)
#         label_model.fit(L_train=all_labels, n_epochs=500, log_freq=100, seed=123)
#         probs_train = label_model.predict_proba(L=all_labels)
#     
#         train_labeled_df1[cln+'_final'] = probs_train
    

    
def create_identifiers(df):
    
    df['id'] = df['subject_id'].astype(str) + parser + df['study_id'].astype(str) + parser + df['dicom_id']
    
    return df 

if __name__ == '__main__':
    args = parse_optim_del_args()
    
    train_ids, valid_ids, test_ids = get_train_valid_test_split()
    
    labeled_file1 = 'mimic-cxr-2.0.0-chexpert.csv'
    
    labeled_file2 = 'mimic-cxr-2.0.0-negbio.csv'
    
    train_labeled_df1 = create_identifiers(join_id_df_with_label_df(train_ids, labeled_file1))
    
    train_labeled_df2 = create_identifiers(join_id_df_with_label_df(train_ids, labeled_file2))
    
    train_labeled_df1 = train_labeled_df1.sort_values(by = ['id'])
    
    train_labeled_df2 = train_labeled_df2.sort_values(by = ['id'])
    
    valid_labeled_df1 = create_identifiers(join_id_df_with_label_df(valid_ids, labeled_file1))
    
    valid_labeled_df2 = create_identifiers(join_id_df_with_label_df(valid_ids, labeled_file2))
    
    valid_labeled_df1 = valid_labeled_df1.sort_values(by = ['id'])
    
    valid_labeled_df2 = valid_labeled_df2.sort_values(by = ['id'])
    
    test_labeled_df1 = create_identifiers(join_id_df_with_label_df(test_ids, labeled_file1))
    
    test_labeled_df2 = create_identifiers(join_id_df_with_label_df(test_ids, labeled_file2))
    
    test_labeled_df1 = test_labeled_df1.sort_values(by = ['id'])
    
    test_labeled_df2 = test_labeled_df2.sort_values(by = ['id'])
    
    filter_empty_records_valid_test(valid_labeled_df1, valid_labeled_df2, 'valid')
     
    filter_empty_records_valid_test(test_labeled_df1, test_labeled_df2, 'test')
    
#     obtain_prob_labels_training(train_labeled_df1, train_labeled_df2)
    
    obtain_prob_labels_training_origin(train_labeled_df1, train_labeled_df2)
    
    
    