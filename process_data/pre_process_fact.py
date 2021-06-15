'''
Created on Apr 20, 2021

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
import torch.nn.functional as F
from torchtext.vocab import CharNGram
from torchtext.vocab import GloVe

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Reweight_examples')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')

from gensim.test.utils import datapath

from torch import optim


try:
    from utils.utils import *
    from process_data.cage import *
    from models.util_func import *
    # from real_examples.utils_real import *
    # from models.util_func import *
    # from Reweight_examples.utils_reweight import *
except ImportError:
    from utils import *
    from cage import *
    from util_func import *
    # from utils_real import *
    # from util_func import *
    # from utils_reweight import *
    
    
    
judge_json_files = 'Google_KG_institution_CrowdScale_v3_questions_full_data.csv'

gold_label_file = 'gold.public.csv'

freebase_files = 'mid2name.tsv'

def download_corpus():
    # corpus = api.load('text8')
    info = api.info()
    print(json.dumps(info, indent=4))
    for model_name, model_data in sorted(info['models'].items()):
        print(
            '%s (%d records): %s' % (
                model_name,
                model_data.get('num_records', -1),
                model_data['description'][:40] + '...',
            )
        )
    



def convert_string_to_json(text_str):
    final_dic = json.loads(text_str)
    
    return final_dic

def get_all_judge_raters(all_raters, curr_json_obj):
    judge_list = curr_json_obj["judgments"]
    
    for j in range(len(judge_list)):
        all_raters.add(judge_list[j]["rater"])
    
def initialize_labels_df(qids, raters):
    init_data = np.ones((len(qids), len(raters)))*(-1)
    
    df = pd.DataFrame(data = init_data, index = qids, columns = raters)
        
    
    return df
    

def set_df_values(df, judge_mappings):
    for qid in judge_mappings:
        curr_judge_list = judge_mappings[qid]['judgments']
        
        print('curr qid::', qid)
        
        for k in range(len(curr_judge_list)):
            
            # print(curr_judge_list)
            
            curr_rater = curr_judge_list[k]["rater"]
            
            curr_rate = curr_judge_list[k]["judgment"]
            
            df.loc[qid][curr_rater] = curr_rate
            
    return df
            
def convert_property_to_str(word):
    property_str_list = word.split('/')
    
    existing_word = set()
    
    str_list = []
    
    for w in property_str_list:
        
        if len(w) <= 0:
            continue
        
        w = re.sub(r'\W+', '', w)
    
        if w in existing_word:
            continue
        existing_word.add(w)
        
        
        # print('word use::', w)
        
        
        str_list.append(w)
        # w = w.replace(',','')
        
        # vec_form = convert_word_2_vec(w)
        #
        # embedding_list.append(vec_form)
        
    return str_list, ' '.join(str_list)

def convert_str_to_embedding2(model_glove_twitter, word):
    
    # word_list, _ = convert_property_to_str(word)
    word_list = word.split(' ')
    
    average_score = 0
    
    for k in range(len(word_list)):
        
        curr_word = word_list[k].lower()
        
        curr_word = re.sub(r'\W+', '', curr_word)
        
        average_score += model_glove_twitter[curr_word] 
    
    return average_score/len(word_list)

def convert_str_to_embeddings(tokenizer, model, curr_string):
    
    
    # word_list = curr_string.split(' ')
    #
    # embedding_list = []
    #
    # for i in range(len(word_list)):
    #
        # vec_form = convert_word_2_vec(word_list[i])
        #
        # embedding_list.append(vec_form)
    
    # import sentencepiece as spm
    # s = spm.SentencePieceProcessor(model_file='spm.model')
    # for n in range(5):
        # s.encode(curr_string, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
    # max_length = 64
    
    indexed_tokens  = tokenizer.encode(curr_string, max_length = 512)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    
    with torch.no_grad():
        output = model(tokens_tensor)
        encoded_layers = output.last_hidden_state
    
    print(output.last_hidden_state.shape)
    # print(output.shape, encoded_layers.shape)
    
    return encoded_layers[0]

def align_single_tensor_embedding(curr_embedding, min_len):
    if curr_embedding.shape[0] < min_len:
        new_embedding = torch.zeros(min_len, curr_embedding.shape[1])
        new_embedding[-curr_embedding.shape[0]:] = curr_embedding
        return new_embedding
            # embedding_list.append(new_embedding)
    else:
        return curr_embedding[-min_len:]
        # embedding_list.append()
        
    

def align_tensor_embedding_list(df_text_embedding_list, min_len = 20):
    
    # min_len = 512
    embedding_list = []
    
    for j in range(len(df_text_embedding_list)):
        # curr_embedding = df['text_embedding'].iloc[j]
        curr_embedding = df_text_embedding_list[j]
        if curr_embedding.shape[0] < min_len:
            new_embedding = torch.zeros(min_len, curr_embedding.shape[1])
            new_embedding[-curr_embedding.shape[0]:] = curr_embedding
            embedding_list.append(new_embedding)
        else:
            embedding_list.append(curr_embedding[-min_len:])
        
        
        # min_len = min(min_len, curr_embedding.shape[0])
        
    
    return torch.stack(embedding_list, 0)
    
    



# def generate_bert_representations(embedding_lists):
    
def l_func1(subj, obj, pred, evidence, tokenizer, model):
    evid_words = evidence.split(' ')
    
    pred_embedding = convert_str_to_embedding2(model, pred)
    
    max_prob = 1e-4
    
    for word in evid_words:
        
        # print(subj, obj, pred, word)
        
        curr_embedding = convert_str_to_embedding2(model, word)
        
        sim = torch.dot(pred_embedding, curr_embedding)/(torch.norm(pred_embedding)*torch.norm(curr_embedding)) 
        
        max_prob = max(max_prob, sim)
    
    return max_prob, (max_prob > 0.7)

def l_func2(subj, obj, pred, gold_sub_pred_obj_set, tokenizer, model):
    # evid_words = evidence.split(' ')
    #
    pred_embedding = convert_str_to_embedding2(model, pred)
    
    subj_embedding = convert_str_to_embedding2(model, subj)
    
    obj_embedding = convert_str_to_embedding2(model, obj)
    #
    max_prob = 1e-4
    
    for gold_sub_pred_obj in gold_sub_pred_obj_set:
        
        gold_sub = gold_sub_pred_obj[0]
        
        gold_pred = gold_sub_pred_obj[1]
        
        gold_obj = gold_sub_pred_obj[2]
        
        gold_sub_embedding = convert_str_to_embedding2(model, gold_sub)
        
        gold_pred_embedding = convert_str_to_embedding2(model, gold_pred)
        
        gold_obj_embedding = convert_str_to_embedding2(model, gold_obj)
        
        sim1 = torch.dot(gold_sub_embedding, subj_embedding)/(torch.norm(gold_sub_embedding)*torch.norm(subj_embedding))
        
        sim2 = torch.dot(gold_pred_embedding, pred_embedding)/(torch.norm(gold_pred_embedding)*torch.norm(pred_embedding))
        
        sim3 = torch.dot(gold_obj_embedding, obj_embedding)/(torch.norm(gold_obj_embedding)*torch.norm(obj_embedding))
        
        min_sim = min(sim1, sim2, sim3)
        
        # sim = torch.dot(pred_embedding, curr_embedding)/(torch.norm(pred_embedding)*torch.norm(curr_embedding)) 
        
        max_prob = max(max_prob, min_sim)
    
    return max_prob, (max_prob > 0.7)


def l_func3(subj, obj, pred, evidence, tokenizer, model):
    evid_words = evidence.split(' ')
    
    subj_embedding = convert_str_to_embedding2(model, subj)
    
    obj_embedding = convert_str_to_embedding2(model, obj)
    
    max_prob1 = 1e-4
    
    max_prob2 = 1e-4
    
    for word in evid_words:
        
        curr_embedding = convert_str_to_embedding2(model, word)
        
        sim1 = torch.dot(subj_embedding, curr_embedding)/(torch.norm(subj_embedding)*torch.norm(curr_embedding))
        
        sim2 = torch.dot(obj_embedding, curr_embedding)/(torch.norm(obj_embedding)*torch.norm(curr_embedding)) 
        
        max_prob1 = max(max_prob1, sim1)
        
        max_prob2 = max(max_prob2, sim2)
    
    max_prob = min(max_prob1, max_prob2)
    
    return max_prob, (max_prob > 0.7)


def read_and_obtain_all_linking(file_name):
    rid = 0
    
    label_mappings = {}
    
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            
            if rid <= 0:
                rid += 1
                continue
            
            qid = int(row[0])
            if qid == 2:
                continue
            
            label = int(row[1])
            
            label_mappings[qid] = label
            
            
    return label_mappings


def read_and_obtain_positive_linking(file_name):
    
    rid = 0
    
    label_mappings = {}
    
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            
            if rid <= 0:
                rid += 1
                continue
            
            qid = int(row[0])
            if qid != 1:
                continue
            
            label = int(row[1])
            
            label_mappings[qid] = label
            
            
    return label_mappings
                

def obtain_valid_test_parition(gold_label_mappings):
    all_gold_labeled_ids = list(gold_label_mappings.keys())
    
    indices = list(range(len(all_gold_labeled_ids)))
    
    np.random.seed(0)
    
    np.random.shuffle(indices)
    
    valid_count = int(len(indices)/2)
    
    test_count = int(len(indices)/2)
    
    valid_idx, test_idx = indices[0:valid_count], indices[valid_count:valid_count + test_count]
    
    valid_gold_label_ids = [all_gold_labeled_ids[x] for x in valid_idx]
    
    test_gold_label_ids = [all_gold_labeled_ids[x] for x in test_idx]
    
    return valid_gold_label_ids, test_gold_label_ids
  

def check_label_2_samples(rating_list):
    all_judgements = rating_list['judgments']
    
    for i in range(len(all_judgements)):
        if all_judgements[i]['judgment'] == 2:
            return True
        
    return False

def run_cage(l, s, continuous_mask, n_lfs, n_classes = 2):
    
    
    pi = torch.ones((n_classes, n_lfs)).double()
    pi.requires_grad = True
    
    theta = torch.ones((n_classes, n_lfs)).double() * 1
    theta.requires_grad = True
    
    pi_y = torch.ones(n_classes).double()
    pi_y.requires_grad = True
    
    k = torch.ones(l.shape[1])
    
    optimizer = optim.Adam([theta, pi], lr=0.001, weight_decay=0)
    for epoch in range(100):
        print('cage epoch::', epoch)
        optimizer.zero_grad()
        loss = log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
        prec_loss = 0#precision_loss(theta, k, n_classes, a)
        loss += prec_loss
        y_pred = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
        
        # y_pred = np.argmax(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask).detach().numpy(), 1)
        # print("Epoch: {}\tf1_score: {}".format(epoch, f1_score(y_true_test, y_pred, average="binary")))
    
        loss.backward()
        optimizer.step()
        
    return F.softmax(y_pred, 1)

def get_annotated_labels(json_obj):
    all_judgements = json_obj['judgments']
    
    all_annotated_labels = []
    
    for k in range(len(all_judgements)):
        curr_annotated_label = all_judgements[k]['judgment']
        
        if curr_annotated_label == 2:
            curr_annotated_label = -1
        
        all_annotated_labels.append(curr_annotated_label)
        
    return all_annotated_labels
    


def obtain_train_fake_real_labels(label_list):
    label_list_tensor = torch.tensor(label_list)
    
    return torch.mode(label_list_tensor)[0]


def post_process_valid_test_dataset():
    valid_feature_tensor = torch.load(os.path.join(args.output_dir, 'valid_feature_tensor'))
    
    test_feature_tensor = torch.load(os.path.join(args.output_dir, 'test_feature_tensor'))
    
    valid_label_tensor = torch.load(os.path.join(args.output_dir, 'valid_labels'))
    
    test_label_tensor = torch.load(os.path.join(args.output_dir, 'test_labels'))
    
    valid_test_feature_tensor = torch.cat([valid_feature_tensor, test_feature_tensor], 0)
    
    valid_test_label_tensor = torch.cat([valid_label_tensor, test_label_tensor], 0)
    
    random_ids = torch.randperm(valid_test_feature_tensor.shape[0])
    
    valid_ids = random_ids[0:int(random_ids.shape[0]/2)]
    
    test_ids = random_ids[int(random_ids.shape[0]/2):]
    
    new_valid_feature_tensor = valid_test_feature_tensor[valid_ids]
    
    new_test_feature_tensor = valid_test_feature_tensor[test_ids]
    
    new_valid_label_tensor = valid_test_label_tensor[valid_ids]
    
    new_test_label_tensor = valid_test_label_tensor[test_ids]
    
    print(len(new_valid_feature_tensor), len(new_valid_label_tensor))
    
    print(len(new_test_feature_tensor), len(new_test_label_tensor))
    
    torch.save(new_valid_feature_tensor, os.path.join(args.output_dir, 'valid_feature_tensor'))
    
    torch.save(new_test_feature_tensor, os.path.join(args.output_dir, 'test_feature_tensor'))
    
    torch.save(new_valid_label_tensor, os.path.join(args.output_dir, 'valid_labels'))
    
    torch.save(new_test_label_tensor, os.path.join(args.output_dir, 'test_labels'))


def calculate_num_raters(file_name):
    rater_set = set()
    rid = 0
    with open(file_name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                
                if rid <= 0:
                    rid += 1
                    continue
        
                # if rid >= 200:
                    # break
                    #
                qid = int(row[0])
                
                # if qid >= 200:
                    # # print('here')
                    # break
                
                print('qid::', qid)
                
                json_obj = convert_string_to_json(row[1])
                
                all_annotated_labels = get_annotated_labels(json_obj)
                
                # if check_label_2_samples(json_obj):
                    # print('has ambiguous labels::', )
                    # continue
                
                # judge_mappings[qid] = json_obj
                
                for k in range(len(json_obj['judgments'])):
                    curr_rater_id = json_obj['judgments'][k]['rater']
                    
                    rater_set.add(curr_rater_id)
                
                # get_all_judge_raters(all_raters, json_obj)
                
                # obj_text, sub_text = map_obj_sub_to_free_base(json_obj['obj'], json_obj['sub'], freebase_mappings)
                
                # if obj_text is None or sub_text is None:
                #     print('mid no mappings ', no_mapping_count)
                #     no_mapping_count += 1
                #     continue
                #
                # # print(qid, sub_text, json_obj['pred'], obj_text)
                # _,property_str = convert_property_to_str(json_obj['pred'])
                #
                #
                # origin_embedding1 = align_single_tensor_embedding(convert_str_to_embeddings(tokenizer, model, ' '.join([sub_text, property_str, obj_text])), min_len = 5)
                #
                # origin_embedding2 = align_single_tensor_embedding(convert_str_to_embeddings(tokenizer, model, json_obj['evidences'][0]['snippet']), min_len = 15)
                #
                # # curr_embedding = torch.cat([embedding1, embedding2], 0)
                #
                # curr_origin_embedding = torch.cat([origin_embedding1, origin_embedding2], 0)
                #
                # print('embedding shape::', curr_origin_embedding.shape)
                # # embedding_list.append()
                #
                # if qid in valid_gold_label_ids:
                #     # valid_feature_list.append(curr_embedding)
                #     valid_origin_feature_list.append(curr_origin_embedding)
                #     valid_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                #     valid_label_list.append(gold_label_mappings[qid])
                # else:
                #     if qid in test_gold_label_ids:
                #         # test_feature_list.append(curr_embedding)
                #         test_origin_feature_list.append(curr_origin_embedding)
                #         test_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                #         test_label_list.append(gold_label_mappings[qid])
                #     else:
                #         # train_feature_list.append(curr_embedding)
                #
                #         train_qid_list.append(qid)
                #         train_origin_feature_list.append(curr_origin_embedding)
                #         train_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                #         train_evidence_list.append(json_obj['evidences'][0]['snippet'])
                #         train_annotated_label_list.append(all_annotated_labels)
                #         derived_train_real_label_list.append(obtain_train_fake_real_labels(all_annotated_labels))
                # # sub_embeddings = convert_str_to_embeddings(sub_text) 
                #
                # # print(row)
                rid += 1
                
                
    print('number of raters::', len(rater_set))

def repartition_valid_test_samples(valid_label_tensor, test_label_tensor, valid_feature_tensor, test_feature_tensor, origin_valid_feature_tensor, origin_test_feature_tensor, valid_ids, test_ids):
    valid_test_labels = torch.cat([valid_label_tensor, test_label_tensor])
    
    valid_test_feature_tensor = torch.cat([valid_feature_tensor, test_feature_tensor])
    
    origin_valid_test_feature_tensor = torch.cat([origin_valid_feature_tensor, origin_test_feature_tensor])
    
    valid_test_ids = torch.cat([valid_ids, test_ids])
    
    pos_ids = torch.nonzero(valid_test_labels == 1).view(-1)
    
    neg_ids = torch.nonzero(valid_test_labels == 0).view(-1)
    
    pos_rand_ids = torch.randperm(pos_ids.shape[0])
    
    pos_val_count = int(pos_rand_ids.shape[0]/2)
    
    neg_rand_ids = torch.randperm(neg_ids.shape[0])
    
    neg_val_count = int(neg_rand_ids.shape[0]/2)
    
    valid_pos_ids = pos_ids[pos_rand_ids[0:pos_val_count]]
    
    valid_neg_ids = neg_ids[neg_rand_ids[0:neg_val_count]]
    
    test_pos_ids = pos_ids[pos_rand_ids[pos_val_count:]]
    
    test_neg_ids = neg_ids[neg_rand_ids[neg_val_count:]]
    
    valid_pos_features = valid_test_feature_tensor[valid_pos_ids]
    
    valid_neg_features = valid_test_feature_tensor[valid_neg_ids]
    
    test_pos_features = valid_test_feature_tensor[test_pos_ids]
    
    test_neg_features = valid_test_feature_tensor[test_neg_ids]
    
    valid_new_features = torch.cat([valid_pos_features, valid_neg_features])
    
    test_new_features = torch.cat([test_pos_features, test_neg_features])
    
    valid_pos_features = valid_origin_feature_tensor[valid_pos_ids]
    
    valid_neg_features = valid_feature_tensor[valid_neg_ids]
    
    test_pos_features = test_feature_tensor[test_pos_ids]
    
    test_neg_features = test_feature_tensor[test_neg_ids]
    
    
    
    
    

def read_judge_file(file_name, gold_label_mappings, freebase_mappings, valid_gold_label_ids, test_gold_label_ids, recompute = True, training_clean_sample_ratio = 0):
    if recompute:
    
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        
        config.max_position_embeddings = 1024
        
        model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')   # Download vocabulary from S3 and cache.
        
        test_feature_tensor = None 
        
        test_label_tensor = None
        
        rid = 0
        
        judge_mappings = {}
        
        all_raters = set()
        
        # embedding_list = []
        
        no_mapping_count = 0
        
        # valid_feature_list = []
        
        # test_feature_list = []
        
        valid_origin_feature_list = []
        
        test_origin_feature_list = []
        
        # train_feature_list = []
        
        train_origin_feature_list = []
    
        train_sub_pred_obj_list = []
        
        valid_sub_pred_obj_list = []
        
        test_sub_pred_obj_list = []
    
        train_evidence_list = []
        
        valid_label_list = []
        
        test_label_list = []
        
        train_annotated_label_list = []
        
        derived_train_real_label_list = []
        
        train_qid_list = []
        
        with open(file_name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                
                if rid <= 0:
                    rid += 1
                    continue
        
                # if rid >= 200:
                    # break
                    #
                qid = int(row[0])
                
                # if qid >= 200:
                    # # print('here')
                    # break
                
                print('qid::', qid)
                
                json_obj = convert_string_to_json(row[1])
                
                all_annotated_labels = get_annotated_labels(json_obj)
                
                # if check_label_2_samples(json_obj):
                    # print('has ambiguous labels::', )
                    # continue
                
                judge_mappings[qid] = json_obj
                
                # get_all_judge_raters(all_raters, json_obj)
                
                obj_text, sub_text = map_obj_sub_to_free_base(json_obj['obj'], json_obj['sub'], freebase_mappings)
                
                if obj_text is None or sub_text is None:
                    print('mid no mappings ', no_mapping_count)
                    no_mapping_count += 1
                    continue
                
                # print(qid, sub_text, json_obj['pred'], obj_text)
                _,property_str = convert_property_to_str(json_obj['pred'])
                
                
                origin_embedding1 = align_single_tensor_embedding(convert_str_to_embeddings(tokenizer, model, ' '.join([sub_text, property_str, obj_text])), min_len = 5)
                
                origin_embedding2 = align_single_tensor_embedding(convert_str_to_embeddings(tokenizer, model, json_obj['evidences'][0]['snippet']), min_len = 15)
                
                # curr_embedding = torch.cat([embedding1, embedding2], 0)
                
                curr_origin_embedding = torch.cat([origin_embedding1, origin_embedding2], 0)
                
                print('embedding shape::', curr_origin_embedding.shape)
                # embedding_list.append()
                
                if qid in valid_gold_label_ids:
                    # valid_feature_list.append(curr_embedding)
                    valid_origin_feature_list.append(curr_origin_embedding)
                    valid_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                    valid_label_list.append(gold_label_mappings[qid])
                else:
                    if qid in test_gold_label_ids:
                        # test_feature_list.append(curr_embedding)
                        test_origin_feature_list.append(curr_origin_embedding)
                        test_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                        test_label_list.append(gold_label_mappings[qid])
                    else:
                        # train_feature_list.append(curr_embedding)
                        
                        train_qid_list.append(qid)
                        train_origin_feature_list.append(curr_origin_embedding)
                        train_sub_pred_obj_list.append([sub_text, property_str, obj_text])
                        train_evidence_list.append(json_obj['evidences'][0]['snippet'])
                        train_annotated_label_list.append(all_annotated_labels)
                        derived_train_real_label_list.append(obtain_train_fake_real_labels(all_annotated_labels))
                # sub_embeddings = convert_str_to_embeddings(sub_text) 
                
                # print(row)
                rid += 1
        
        # embedding_list_tensor = torch.stack(embedding_list, 0)
        
        # print(all_raters)
        
        # label_df = initialize_labels_df(list(judge_mappings.keys()), all_raters)
        #
        # label_df = set_df_values(label_df, judge_mappings)
        #
        # # print(label_df)
        #
        # label_df_array = np.array(label_df)
        #
        # label_df_array_for_prob_labels = label_df_array[:, 0:int(label_df_array.shape[1]/2)]
        #
        # label_df_array_remaining_labels = label_df_array[:, int(label_df_array.shape[1]/2):]
        #
        # label_model = LabelModel(cardinality=3, verbose=True)
        #
        # label_model.fit(L_train=label_df_array_for_prob_labels, n_epochs=500, log_freq=100, seed=123)
        #
        # probs_train = label_model.predict_proba(L=label_df_array_for_prob_labels)
        
        train_prob_labels = []
        
        prob_label_list = []
        
        derived_label_list = []
        
        # model_glove_twitter = api.load("glove-twitter-25")
        
        embedding_glove = GloVe(name='6B', dim=100)
    
        
        for k in range(len(train_sub_pred_obj_list)):
            print('train sample id::', k)
            train_sub_pred_obj = train_sub_pred_obj_list[k]
            train_evidence = train_evidence_list[k]
            prob_label1, label1 = l_func1(train_sub_pred_obj[0], train_sub_pred_obj[2], train_sub_pred_obj[1], train_evidence, tokenizer, embedding_glove)
            
            prob_label2, label2 = l_func2(train_sub_pred_obj[0], train_sub_pred_obj[2], train_sub_pred_obj[1], valid_sub_pred_obj_list, tokenizer, embedding_glove)
            
            prob_label3, label3 = l_func3(train_sub_pred_obj[0], train_sub_pred_obj[2], train_sub_pred_obj[1], train_evidence, tokenizer, embedding_glove)
        
            prob_label_list.append([prob_label1, prob_label2, prob_label3])
            
            derived_label_list.append([label1, label2, label3])
    

        if len(derived_train_real_label_list) > 0:
            derived_train_real_label_list_tensor = torch.tensor(derived_train_real_label_list)
            torch.save(derived_train_real_label_list_tensor, os.path.join(args.output_dir, 'derived_train_real_label_list_tensor'))
            print('derived train shape::', derived_train_real_label_list_tensor.shape)
    
        if len(train_origin_feature_list) > 0:
            # train_origin_feature_tensor = torch.stack(train_origin_feature_list, 0)
            
            train_origin_feature_tensor = torch.stack(train_origin_feature_list,0)# align_tensor_embedding_list(train_origin_feature_list, min_len=20)
        
            
            torch.save(train_origin_feature_tensor, os.path.join(args.output_dir, 'train_origin_feature_tensor'))

            train_feature_tensor = train_origin_feature_tensor.view(train_origin_feature_tensor.shape[0],-1) #torch.stack(train_origin_feature_list,0)
            
            torch.save(train_feature_tensor, os.path.join(args.output_dir, 'train_feature_tensor'))
            
            print('train shape::', train_feature_tensor.shape)
            
        if len(valid_origin_feature_list) > 0:
            # valid_origin_feature_tensor = torch.stack(valid_origin_feature_list, 0)

            valid_origin_feature_tensor = torch.stack(valid_origin_feature_list, 0)# align_tensor_embedding_list(valid_origin_feature_list, min_len=20)
        
            
            torch.save(valid_origin_feature_tensor, os.path.join(args.output_dir, 'valid_origin_feature_tensor'))

            valid_feature_tensor = valid_origin_feature_tensor.view(valid_origin_feature_tensor.shape[0],-1) #torch.stack(train_origin_feature_list,0)
            
            torch.save(valid_feature_tensor, os.path.join(args.output_dir, 'valid_feature_tensor'))
            
            print('valid shape::', valid_feature_tensor.shape)
        
        
        if len(test_origin_feature_list) > 0:
            # test_origin_feature_tensor = torch.stack(test_origin_feature_list, 0)
            test_origin_feature_tensor = torch.stack(test_origin_feature_list, 0)# align_tensor_embedding_list(test_origin_feature_list, min_len=20)
            
            torch.save(test_origin_feature_tensor, os.path.join(args.output_dir, 'test_origin_feature_tensor'))

            test_feature_tensor = test_origin_feature_tensor.view(test_origin_feature_tensor.shape[0],-1) #torch.stack(train_origin_feature_list,0)
            
            torch.save(test_feature_tensor, os.path.join(args.output_dir, 'test_feature_tensor'))
            
            print('test shape::', test_feature_tensor.shape)
        
        # if len(valid_origin_feature_list) > 0:
            # valid_feature_tensor = torch.stack(valid_origin_feature_list, 0)
            #
            # torch.save(valid_feature_tensor, os.path.join(args.output_dir, 'valid_feature_tensor'))
            #
            # valid_origin_feature_tensor = torch.stack(valid_origin_feature_list,0)
            #
            # torch.save(valid_origin_feature_tensor, os.path.join(args.output_dir, 'valid_origin_feature_tensor'))
            #
            # print('valid shape::', valid_feature_tensor.shape)
            #
        # if len(test_feature_list) > 0:
            # test_feature_tensor = torch.stack(test_feature_list, 0)
            #
            # torch.save(test_feature_tensor, os.path.join(args.output_dir, 'test_feature_tensor'))
            #
            # test_origin_feature_tensor = torch.stack(test_origin_feature_list,0)
            #
            # torch.save(test_origin_feature_tensor, os.path.join(args.output_dir, 'test_origin_feature_tensor'))
            #
            # print('test shape::', test_feature_tensor.shape)
        
        if len(valid_label_list) > 0:
            valid_label_tensor = torch.tensor(valid_label_list)
            
            torch.save(valid_label_tensor, os.path.join(args.output_dir, 'valid_labels'))
            
            print('valid label shape::', valid_label_tensor.shape)
        
        if len(test_label_list) > 0:
            test_label_tensor = torch.tensor(test_label_list)
            
            torch.save(test_label_tensor, os.path.join(args.output_dir, 'test_labels'))
        
            print('test label shape::', test_label_tensor.shape)
        
        if len(prob_label_list) > 0:
            prob_label_tensor = torch.tensor(prob_label_list)
        
            prob_label_tensor[prob_label_tensor >= 1-1e-4] = 1-1e-4
        
            torch.save(prob_label_tensor, os.path.join(args.output_dir, 'prob_label_tensor'))
    
        if len(derived_label_list) > 0:    
            derived_label_tensor = torch.tensor(derived_label_list).float()
            
            torch.save(derived_label_tensor, os.path.join(args.output_dir, 'derived_label_tensor'))
    
        if len(train_annotated_label_list) > 0:
            torch.save(train_annotated_label_list, os.path.join(args.output_dir, 'train_annotated_label_list'))
        
        
        torch.save(torch.tensor(train_qid_list), os.path.join(args.output_dir, 'train_qid_list'))
        
    else:
         
        train_feature_tensor = torch.load(os.path.join(args.output_dir, 'train_feature_tensor'))
        
        valid_feature_tensor = torch.load(os.path.join(args.output_dir, 'valid_feature_tensor'))
        
        test_feature_tensor = torch.load(os.path.join(args.output_dir, 'test_feature_tensor'))
        
        # valid_origin_feature_tensor = torch.stack(valid_origin_feature_list,0)
            
        valid_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'valid_origin_feature_tensor'))
        
        train_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'train_origin_feature_tensor'))
        
        test_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'test_origin_feature_tensor'))
        
        valid_label_tensor = torch.load(os.path.join(args.output_dir, 'valid_labels'))
        
        test_label_tensor = torch.load(os.path.join(args.output_dir, 'test_labels'))
        
        prob_label_tensor = torch.load(os.path.join(args.output_dir, 'prob_label_tensor'))
        
        derived_label_tensor = torch.load(os.path.join(args.output_dir, 'derived_label_tensor'))
        
        train_annotated_label_list = torch.load(os.path.join(args.output_dir, 'train_annotated_label_list'))
        
        derived_label_tensor = (prob_label_tensor >= 0.7).float()
        
        prob_label_tensor[prob_label_tensor >= 1-1e-4] = 1-1e-4
    
    
    
    
    
    
    
    
    agg_prob_label_tensor = run_cage(derived_label_tensor, prob_label_tensor, torch.tensor([1,1,1]), prob_label_tensor.shape[1], n_classes=2)
    
    
    
    print('train prob label shape::', agg_prob_label_tensor.shape)
    
    print('train annotated label shape::', len(train_annotated_label_list))
    
    
    no_conflict_train_label_booleans = (derived_train_real_label_list_tensor != -1)
    
    no_conflict_train_label_ids = torch.nonzero(no_conflict_train_label_booleans).numpy()
    
    np.random.shuffle(no_conflict_train_label_ids)
    
    selected_no_conflict_train_label_ids = no_conflict_train_label_ids[0:int(training_clean_sample_ratio*len(no_conflict_train_label_ids))]
    
    torch.save(torch.tensor(selected_no_conflict_train_label_ids.reshape(-1)), os.path.join(args.output_dir, 'clean_sample_ids'))
    
    dirty_ids = list(set(range(len(derived_train_real_label_list_tensor))).difference(set(list(selected_no_conflict_train_label_ids.reshape(-1)))))
    
    torch.save(torch.tensor(dirty_ids), os.path.join(args.output_dir, 'noisy_sample_ids'))
    
    if len(selected_no_conflict_train_label_ids) > 0:
        agg_prob_label_tensor[selected_no_conflict_train_label_ids.reshape(-1)] = onehot(derived_train_real_label_list_tensor[selected_no_conflict_train_label_ids.reshape(-1)], 2).type(torch.double)
    
    torch.save(agg_prob_label_tensor, os.path.join(args.output_dir, 'train_prob_labels'))
       
    return train_feature_tensor, valid_feature_tensor, test_feature_tensor, agg_prob_label_tensor, train_annotated_label_list, valid_label_tensor, test_label_tensor, train_origin_feature_tensor, valid_origin_feature_tensor, test_origin_feature_tensor


def load_freebase(file_name):
    
    freebase_mappings = {}
    
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            print(row, len(row))
            
            mid = row[0]
            text = row[1]
            freebase_mappings[mid] = text
            
    return freebase_mappings

def map_obj_sub_to_free_base(obj, subj, freebase_mappings):
    try:
        return freebase_mappings[obj], freebase_mappings[subj]
    except:
        return None, None
    
def convert_word_2_vec(content):
    
    model = api.load("glove-wiki-gigaword-50")

    
    # model = gensim.models.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=True)
        
    return model[content]


def obtain_partitioned_tensor(tensor, min_len1 = 5, min_lenth2 = 15, origin_min_len1 = 10, origin_min_len2 = 30):
    tensor1 = tensor[:,origin_min_len1 - min_len1: origin_min_len1]
    
    tensor2 = tensor[:,origin_min_len1 + origin_min_len2 - min_lenth2: origin_min_len1 + origin_min_len2]
    
    return torch.cat([tensor1, tensor2], 1)
    
    

if __name__ == '__main__':
    
    download_corpus()
    
    parser = ArgumentParser()
    
    default_git_ignore_dir = get_default_git_ignore_dir()
    
    print('default git ignore dir::', default_git_ignore_dir)
    
    default_output_dir = os.path.join(default_git_ignore_dir, 'crowdsourced_dataset/fact/')
    
    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)
    
    
    parser.add_argument('--output_dir', type = str, default = default_output_dir, help="output directory")
    
    parser.add_argument('--recompute', action = 'store_true')
    
    args = parser.parse_args()
    
    if os.path.exists(os.path.join(args.output_dir, 'freebase_mappings')):
        
        with open(os.path.join(args.output_dir, 'freebase_mappings')) as json_file:
            freebase_mappings = json.load(json_file)
    
    else:
        freebase_mappings = load_freebase(os.path.join(args.output_dir, freebase_files))
            
        with open(os.path.join(args.output_dir, 'freebase_mappings'), 'w') as outfile:
            json.dump(freebase_mappings, outfile)
            
        print('json dump done!!')
    
    
    full_gold_file_name = os.path.join(args.output_dir,gold_label_file)
    
    gold_label_mappings = read_and_obtain_all_linking(full_gold_file_name)
    
    valid_gold_label_ids, test_gold_label_ids = obtain_valid_test_parition(gold_label_mappings)
    
    # post_process_valid_test_dataset()
    
    full_sample_file = os.path.join(args.output_dir,judge_json_files)
    
    calculate_num_raters(full_sample_file)
    # full_df = pd.read_csv(full_sample_file)
    
    if True:
        train_feature_tensor, valid_feature_tensor, test_feature_tensor, agg_prob_label_tensor, train_annotated_label_list, valid_label_tensor, test_label_tensor, train_origin_feature_tensor, valid_origin_feature_tensor, test_origin_feature_tensor = read_judge_file(os.path.join(args.output_dir,judge_json_files), gold_label_mappings, freebase_mappings, valid_gold_label_ids, test_gold_label_ids, recompute = args.recompute)
    
    else:
        train_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'train_origin_feature_tensor'))
        
        valid_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'valid_origin_feature_tensor'))
        
        test_origin_feature_tensor = torch.load(os.path.join(args.output_dir, 'test_origin_feature_tensor'))
        
        print(train_origin_feature_tensor.shape)
    
        train_origin_feature_tensor = obtain_partitioned_tensor(train_origin_feature_tensor)
        
        valid_origin_feature_tensor = obtain_partitioned_tensor(valid_origin_feature_tensor)
        
        test_origin_feature_tensor = obtain_partitioned_tensor(test_origin_feature_tensor)
        
        print(train_origin_feature_tensor.shape, valid_origin_feature_tensor.shape[0], test_origin_feature_tensor.shape)
        
        train_feature_tensor = train_origin_feature_tensor.view(train_origin_feature_tensor.shape[0], -1)
        
        valid_feature_tensor = valid_origin_feature_tensor.view(valid_origin_feature_tensor.shape[0], -1) 
        
        test_feature_tensor = test_origin_feature_tensor.view(test_origin_feature_tensor.shape[0], -1)
        
        print(train_feature_tensor.shape, valid_feature_tensor.shape, test_feature_tensor.shape)
        
        torch.save(train_feature_tensor, os.path.join(args.output_dir, 'train_feature_tensor'))
        
        torch.save(valid_feature_tensor, os.path.join(args.output_dir, 'valid_feature_tensor'))
        
        torch.save(test_feature_tensor, os.path.join(args.output_dir, 'test_feature_tensor'))
        
        # train_origin_feature_tensor = train_origin_feature_tensor.view(-1, 40, 768).clone()
        #
        # valid_origin_feature_tensor = valid_origin_feature_tensor.view(-1, 40, 768).clone()
        #
        # test_origin_feature_tensor = test_origin_feature_tensor.view(-1, 40, 768).clone()
        #
        # torch.save(train_origin_feature_tensor, os.path.join(args.output_dir, 'train_origin_feature_tensor'))
        #
        # torch.save(valid_origin_feature_tensor, os.path.join(args.output_dir, 'valid_origin_feature_tensor'))
        #
        # torch.save(test_origin_feature_tensor, os.path.join(args.output_dir, 'test_origin_feature_tensor'))
        
        
        # clean_sample_ids = torch.load(os.path.join(args.output_dir, 'clean_sample_ids'))
        #
        # if len(clean_sample_ids) <= 0:
        #
            # agg_prob_label_tensor = torch.load(os.path.join(args.output_dir, 'train_prob_labels'))
            #
            # derived_label_tensor = torch.load(os.path.join(args.output_dir, 'derived_label_tensor'))
            #
            # all_train_ids = torch.randperm(agg_prob_label_tensor.shape[0])
            #
            # random_clean_count = int(agg_prob_label_tensor.shape[0]*0.01)
            #
            # random_clean_ids = all_train_ids[0:random_clean_count]
            #
            # agg_prob_label_tensor = agg_prob_label_tensor.detach()
            #
            # agg_prob_label_tensor[random_clean_ids] = onehot(torch.mode(derived_label_tensor[random_clean_ids].type(torch.long), dim = 1)[0], 2).type(torch.double)
            #
            # torch.save(random_clean_ids, os.path.join(args.output_dir, 'clean_sample_ids'))
            #
            # torch.save(agg_prob_label_tensor, os.path.join(args.output_dir, 'train_prob_labels'))
            #
            #
            # noisy_ids = set(list(range(agg_prob_label_tensor.shape[0]))).difference(set(random_clean_ids))
            #
            # torch.save(noisy_ids, os.path.join(args.output_dir, 'noisy_sample_ids'))
        
    
    print('train feature shape')
    

    
    # torch.save(embedding_list_tensor, os.path.join(args.output_dir, 'fact_embedding'))
    #
    # torch.save(probs_train, os.path.join(args.output_dir, 'fact_prob_labels'))
    #
    # torch.save(torch.tensor(label_df_array_remaining_labels), os.path.join(args.output_dir, 'fact_remaining_annotations'))
     
    
    
    
    
    
    