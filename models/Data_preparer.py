'''
Created on Jan 3, 2020


'''
from torchvision.datasets.mnist import *
import torchvision.transforms as transforms
import os, sys

import matplotlib.pyplot as plt
import pandas as pd

import pickle

from torch import nn, optim
import torch
import requests

import bz2
from bz2 import decompress

import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.optim import Adam

import cv2

import csv
from shutil import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/utils')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
# 
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import *
try:
    from utils.utils import *
#     from Models.DNN_transfer import *
# 
except ImportError:
    from utils import *
#     from Models.DNN_transfer import *


# class MyDataset(Dataset):
#     def __init__(self, samples, labels, name_list, lb=None, ub = None):
#         
#         self.data = samples
#         self.labels = labels
#         self.name_list = name_list
#         
#         self.lb = lb
#         
#         self.ub = ub
#         
#     def __getitem__(self, index):
#         data = self.data[index]
#         target = self.labels[index]
#         
# #         data = data.view(-1)
#         
#         # Your transformations here (or set it in CIFAR10)
#         if self.lb is None and self.ub is None:
#             return data.type(torch.DoubleTensor), target, index
#         else:
#             return data.type(torch.DoubleTensor), self.lb[index].type(torch.DoubleTensor),self.ub[index].type(torch.DoubleTensor), target, index
#         
# 
#     def __len__(self):
#         return len(self.data)
# mimic_type = 'Lung Lesion'

mimic_type = 'Lung Opacity'

def save_obj(obj, file_name):
    with open(file_name, "wb") as fp:   #Pickling
        pickle.dump(obj, fp)

def load_obj(file_name):
    with open(file_name, "rb") as fp:   #Pickling
        object = pickle.load(fp)
    return object
        # pickle.dump(obj, fp)

def replace_url_with_image_name(url):
    image_name = url.split('/')[-1]
    real_image_name = image_name.split('_')[0]
    return real_image_name

def replace_annotation_with_labels(text):
    if text == 'no':
        return 0
    else:
        if text == 'yes':
            return 1
        else:
            return -1


class MyDataset(Dataset):
    def __init__(self, samples, labels, name_list=None):
        
        self.data = samples
        self.labels = labels
        self.name_list = name_list
        self.lenth = self.data.shape[0] 
        self.skipped_dataset = None
        
    def __getitem__(self, index):
        data = self.data[index]
        target = self.labels[index]
        
#         data = data.view(-1)
        
        # Your transformations here (or set it in CIFAR10)
#         if self.skipped_dataset is None or (index not in self.skipped_dataset):
        return data.type(torch.DoubleTensor), target, index#.numpy().tolist()
#         else:
#             return data.type(torch.DoubleTensor), target, np.nan#.numpy().tolist()
        

    def __len__(self):
        return self.lenth

class Data_preparer:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        return
    
    
    
#     def prepare

    def get_file_count0(self, data_dir):
        
        all_folders = os.listdir(data_dir)
        
        img_count = 0
        
        for folder in all_folders:
        
            if os.path.isdir(os.path.join(data_dir, folder)):
                all_imgs = os.listdir(os.path.join(data_dir, folder))
                
                img_count += len(all_imgs)
                
        return img_count
    
    
    def get_mimic_file_count(self, data_dir):
        all_pat_folders = os.listdir(data_dir)
        
        full_ids_list =[]
        
        full_dir_list =[]
        
        for pat_head in all_pat_folders:
            pat_head_id = pat_head[1:]
            
            pat_head_dir = os.path.join(data_dir, pat_head)
            
            if os.path.isdir(pat_head_dir):
                all_pat_folders = os.listdir(pat_head_dir)
                
                for pat in all_pat_folders:
                    
                    pat_dir = os.path.join(pat_head_dir, pat)
                    
                    pat_id = pat[1:]
                    
                    
                    
                    if os.path.isdir(pat_dir):
                        all_study_folders = os.listdir(pat_dir)

                        for study in all_study_folders:
                            study_id = study[1:]
                            
                            study_dir = os.path.join(pat_dir, study)
                        
                        
                            if os.path.isdir(study_dir):
                                all_img_files = os.listdir(study_dir)
                            
                                for img in all_img_files:
                                    if img.endswith('.jpg'):
                                        img_name = img[0:-4]
                                        
                                        full_id = pat_id + '::' + study_id + '::' + img_name
                                        
                                        full_ids_list.append(full_id)
                                        
                                        full_dir_list.append(os.path.join(study_dir, img_name))
                                    
        return full_ids_list, full_dir_list
             
        
    
    def get_file_count(self, data_dir):
        
        all_files = os.listdir(data_dir)
        
        img_count = 0
        
        img_names = []
        
        for i in range(len(all_files)):
            
#             print(data_dir, all_files[i])
            if all_files[i].endswith('jpeg'):
                img_names.append(all_files[i][0:-len('.jpeg')])
                
                
                img_count += 1
        
#         for folder in all_folders:
#         
#             if os.path.isdir(os.path.join(data_dir, folder)):
#                 all_imgs = os.listdir(os.path.join(data_dir, folder))
#                 
#                 img_count += len(all_imgs)
                
        return img_count, img_names
                

    def read_retina_labels(self, label_file):
    
        mappings = {}
        
        with open(label_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    mappings[row[0]] = row[1]
                    line_count += 1
        
        return mappings

    def pre_process_dataset_retina(self, data_dir, val_data_dir, label_file, valid_label_file):
    
        if check_done_file(data_dir):
            return
        
        
    #     data_preparer = models.Data_preparer()
        
    #     get_label_function=getattr(data_preparer, "read_" + args.dataset + '_labels')
        
        label_mappings = self.read_retina_labels(label_file)
        all_imgs = os.listdir(data_dir)
        
        all_labels = set()
        
        for img in all_imgs:
            
            if os.path.isdir(os.path.join(data_dir, img)):
                continue
            
            
            img_name = img.split('.')[0]
            label = label_mappings[img_name]
            
            target_dir = os.path.join(data_dir, str(label))
            
            src_file = os.path.join(data_dir, img)
            
            if (not label in all_imgs) and (not label in all_labels):
                print(label)
                os.mkdir(target_dir)
                all_labels.add(label)
            
            move(src_file, target_dir)
            
        setup_done_file(data_dir)

    
    class DataSet_retina(Dataset):
        def __init__(self, data_dir, label_file, transform, file_names, idx):
            
#             self.label_mapping = {'': 0, '-1.0': 1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            self.label_mappings = self.read_retina_labels(label_file, file_names)
            
            label_id_list = list(self.label_mappings.keys())
            
            self.img_names = [label_id_list[index] for index in idx]
            
            self.transform = transform
            
            self.lenth = len(self.img_names)
            
#             self.data, self.labels = self.read_retina_images() 
            
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def scalRadius(self, img,scale):
            x = img[int(img.shape[0]/2),:,:].sum(1)
            r = (x>x.mean()/10).sum()/2
            s = scale*1.0/r
            return cv2.resize(img,None,fx=s,fy=s)
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            
#             return self.data[idx], self.labels[idx]
            img_loc = os.path.join(self.data_dir, self.img_names[idx] + '.jpeg')
            labels = self.label_mappings[self.img_names[idx]]
            image = Image.open(img_loc).convert("RGB")
            
#             image = cv2.imread(img_loc)
# 
#             print(img_loc, image.shape)
#             
#             scale = 300
# #             a = cv2.imread(f)
#             a = self.scalRadius(image,scale)
#             a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
#             b = np.zeros(a.shape)
#             cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#             a = a*b+128*(1-b)
#             aa = cv2.resize(a,(512,512))
            
            
            tensor_image =self.transform(image)
#             return tensor_image.type(torch.DoubleTensor), (labels >= 3).type(torch.LongTensor), idx
            return tensor_image.type(torch.DoubleTensor), labels.type(torch.LongTensor), idx
        
        def read_retina_images(self):
            
            all_data = []
            
            all_labels = []
            
            for i in range(self.lenth):
                print('index::', i)
                img_loc = os.path.join(self.data_dir, self.img_names[i] + '.jpeg')
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                all_data.append(tensor_image)
                all_labels.append(self.label_mappings[self.img_names[i]])
                
                
            return torch.stack(all_data), torch.tensor(all_labels)
                 
                
                

        def read_retina_labels(self, label_file, img_names):
    
#         observe_ids = [13, 8, ]
#             key_prefix = 'CheXpert-v1.0-small/train/'
    
#             observe_ids = []
#     
#             selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
            mappings = {}
            
            with open(label_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        
#                         for name in selected_observation:
#                             id = [i for i,x in enumerate(row) if x == name]
#                             
#                             observe_ids.extend(id)
                        
                        line_count += 1
                        continue
                    else:
                        
                        curr_key = row[0]
                        
                        if not curr_key in img_names:
                            continue
                        mappings[curr_key] = torch.tensor(int(row[1]))
                        line_count += 1
            
            return mappings




    
    def prepare_retina(self, data_dir, bz, test = False):
        train_label_file = data_dir + '/trainLabels.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        if test:
            img_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(10),
                transforms.ToTensor(),
                transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
            ])
        else:
            img_transform = transforms.Compose([
#                 transforms.RandomResizedCrop((587,587)),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
#         else:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
        
        train_data_dir = os.path.join(data_dir, 'train/')
        
        figure_count, img_names = self.get_file_count(train_data_dir)
        
        print('figure_count::', figure_count)
        
        indices = list(range(figure_count))
        valid_size = 0.1
        split = int(np.floor(valid_size * figure_count))
        shuffle = True
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        
        train_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, train_idx)
        
        valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
        
        
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return train_DL, valid_DL
    
    
    
    
    def prepare_test_retina(self, data_dir, bz):
        test_label_file = data_dir + '/testLabels.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        img_transform = transforms.Compose([
#                 transforms.RandomCrop((587,587)),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        
        test_data_dir = os.path.join(data_dir, 'test/')
        
        figure_count, img_names = self.get_file_count(test_data_dir)
        
        print('figure_count::', figure_count)
        
        indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        test_dataset = self.DataSet_retina(test_data_dir, test_label_file, img_transform, img_names, indices)
        
#         valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
        
        
        test_DL = DataLoader(test_dataset, batch_size=bz)
        
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return test_DL
    
    def prepare_retina0(self, data_dir, bz):
         
         
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
         
        train_label_file = data_dir + '/trainLabels.csv'
         
#         train_label_mappings = self.read_retina_labels(train_label_file)
         
        figure_count = self.get_file_count(data_dir)
         
        print('total figure count::', figure_count)
         
        indices = list(range(figure_count))
        valid_size = 0.1
        split = int(np.floor(valid_size * figure_count))
        shuffle = True
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
         
         
#         test_label_file = data_dir + '/testLabels.csv'
#         
#         test_label_mappings = self.read_retina_labels(test_label_file)
         
#         datadir = '/data/fundus/train'
#         datadir = self.data_dir
        dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
#                 transforms.RandomCrop((587,587))
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
         
        valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
         
#         if self.isGPU:
#             dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#              
#             train_DL = DataLoader(train_dataset, sampler=dist_sampler, batch_size=16, num_workers=4)
#         else:
        train_DL = DataLoader(train_dataset, batch_size=bz)
         
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
         
        return train_DL, valid_DL
        
    
    
    def get_hyperparameters_retina(self, parameters, init_lr, regularization_rate=0.0):
    
#         criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        
        optimizer = Adam(parameters, lr=init_lr)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_retina(self):
        return 2
    

    class DataSet_fact(Dataset):
        def __init__(self, data_dir, prefix):
            
#             self.label_mapping = {'': 0, '-1.0': 1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            
            self.data = torch.load(os.path.join(data_dir, prefix + '_feature_tensor'))
            
            if prefix == 'train':
                self.labels = torch.load(os.path.join(data_dir, prefix + '_labels'))
            else:
                self.labels = torch.load(os.path.join(data_dir, prefix + '_prob_labels'))
            
            self.lenth = self.data.shape[0]
            
            # self.label_mappings = label_mappings
            # # self.label_mappings = self.read_fashion_labels(label_file, file_names)
            #
            # # label_id_list = list(self.label_mappings.keys())
            # #
            #
            # self.img_names = file_names
            # self.img_names = list(self.label_mappings['img_name'])
            # self.img_names = [self.img_names[index] for index in idx]
            
            
            # self.transform = transform
            #
            # self.lenth = len(self.img_names)
            #
            # self.img_sub_dir_mapppings = img_sub_dir_mapppings
            
#             self.data, self.labels = self.read_retina_images() 
            
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def scalRadius(self, img,scale):
            x = img[int(img.shape[0]/2),:,:].sum(1)
            r = (x>x.mean()/10).sum()/2
            s = scale*1.0/r
            return cv2.resize(img,None,fx=s,fy=s)
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            
#             return self.data[idx], self.labels[idx]

            return self.data[idx], self.labels[idx], idx
            # sub_dir = self.img_sub_dir_mapppings[self.img_names[idx]]
            #
            # img_loc = os.path.join(os.path.join(self.data_dir, sub_dir), self.img_names[idx] + '.jpg')
            # labels = torch.tensor(list(self.label_mappings.loc[self.label_mappings['img_name'] == self.img_names[idx], 'label_gold'])).view(-1)
            #
            # print(img_loc, idx, self.img_names[idx], labels)
            #
            # image = Image.open(img_loc).convert("RGB")
            #
# #             image = cv2.imread(img_loc)
# # 
# #             print(img_loc, image.shape)
# #             
# #             scale = 300
# # #             a = cv2.imread(f)
# #             a = self.scalRadius(image,scale)
# #             a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
# #             b = np.zeros(a.shape)
# #             cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
# #             a = a*b+128*(1-b)
# #             aa = cv2.resize(a,(512,512))
            #
            #
            # tensor_image =self.transform(image)
# #             return tensor_image.type(torch.DoubleTensor), (labels >= 3).type(torch.LongTensor), idx
            # return tensor_image.type(torch.DoubleTensor), labels.type(torch.LongTensor), idx
        
        def read_fact_images(self):
            
            all_data = []
            
            all_labels = []
            
            for i in range(self.lenth):
                print('index::', i)
                img_loc = os.path.join(self.data_dir, self.img_names[i] + '.jpeg')
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                all_data.append(tensor_image)
                all_labels.append(self.label_mappings[self.img_names[i]])
                
                
            return torch.stack(all_data), torch.tensor(all_labels)
                 
                

    def read_fact_labels(self, label_file, file_names = None):

#         observe_ids = [13, 8, ]
#             key_prefix = 'CheXpert-v1.0-small/train/'

#             observe_ids = []
#     
#             selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
        df = pd.read_csv(label_file)
        
        df['label_1'] = df['T1_Q1'].apply(replace_annotation_with_labels)
        
        df['label_2'] = df['T2_Q1'].apply(replace_annotation_with_labels)
        
        df['label_3'] = df['T3_Q1'].apply(replace_annotation_with_labels)
        
        df['label_gold'] = df['Majority Q1'].apply(replace_annotation_with_labels)
        
        df['img_name'] = df['PictureURL'].apply(replace_url_with_image_name)
        # return df[['img_name', 'T1_Q1','T2_Q1','T3_Q1','Majority Q1']] 
    
        
        if file_names is not None:
            mappings = df.loc[df['img_name'].isin(file_names), ['img_name', 'label_1','label_2','label_3','label_gold']]
        else:
            mappings = df[['img_name', 'label_1','label_2','label_3','label_gold']]
        
        # with open(label_file) as csv_file:
            # csv_reader = csv.reader(csv_file, delimiter=',')
            # line_count = 0
            # for row in csv_reader:
                # if line_count == 0:
                #
# #                         for name in selected_observation:
# #                             id = [i for i,x in enumerate(row) if x == name]
# #                             
# #                             observe_ids.extend(id)
                    #
                    # line_count += 1
                    # continue
                # else:
                #
                    # curr_key = row[0]
                    #
                    # if not curr_key in img_names:
                        # continue
                    # mappings[curr_key] = torch.tensor(int(row[1]))
                    # line_count += 1
        
        return mappings


    def get_fact_image_count(self, train_data_dir):
        
        img_names = []
        
        img_count = 0
        
        img_sub_dir_mapppings = {}
        
        for sub_folder in os.listdir(train_data_dir):
            
            full_dir = os.path.join(train_data_dir, sub_folder)
            
            if os.path.isdir(full_dir):
                
                for image_files in os.listdir(full_dir):
                    if image_files.endswith('jpg'):
                        
                        # print(image_files)
                        
                        try:
                            Image.open(os.path.join(full_dir, image_files)).convert("RGB")
                            
                        except:
                            # os.remove()
                            continue
                        
                        curr_img_name = image_files[0:-len('.jpg')]
                        
                        img_names.append(curr_img_name)
                        
                        img_sub_dir_mapppings[curr_img_name] = sub_folder
                        
                        img_count += 1
                
        return img_count, img_names, img_sub_dir_mapppings

    # def select_fashion_valid_test_ids(self, no_conflict_samples, count):
    #
    #
        # no_conflict_samples
    
    def prepare_fact(self, data_dir, bz, load = True):
        if load and os.path.exists(os.path.join(data_dir, 'train_dataset')) and os.path.exists(os.path.join(data_dir, 'valid_dataset')) and os.path.exists(os.path.join(data_dir, 'test_dataset')):
            train_dataset = torch.load(os.path.join(data_dir, 'train_dataset'))
    
            valid_dataset = torch.load(os.path.join(data_dir, 'valid_dataset'))
            
            test_dataset = torch.load(os.path.join(data_dir, 'test_dataset'))
            
            return train_dataset, valid_dataset, test_dataset
            
            # train_sample_names = load_obj(os.path.join(data_dir, 'train_sample_names'))
            #
            # valid_sample_names = load_obj(os.path.join(data_dir, 'valid_sample_names'))
            #
            # test_sample_names = load_obj(os.path.join(data_dir, 'test_sample_names'))
            #
            # return train_DL, valid_DL, test_DL, train_sample_names, valid_sample_names, test_sample_names
        
        # train_label_file = data_dir + '/Annotations/Annotation_PerImage_All.csv'
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        # std=[0.229, 0.224, 0.225])
                                        #
                                        #
# #         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        # # if test:
            # # img_transform = transforms.Compose([
                # # transforms.Resize(64),
                # # transforms.CenterCrop(10),
                # # transforms.ToTensor(),
                # # transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
            # # ])
        # # else:
        # img_transform = transforms.Compose([
# #                 transforms.RandomResizedCrop((587,587)),
            # transforms.Resize((64, 64)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            # normalize,
        # ])
#         else:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
        
        # train_data_dir = os.path.join(data_dir, 'train/')
        # train_data_dir = os.path.join(data_dir, 'Fashion10000/Photos')
        #
        # _, img_names, img_sub_dir_mapppings = self.get_fashion_image_count(train_data_dir)
        #
        #
        # label_mappings = self.read_fashion_labels(train_label_file, img_names)
        # # label_mappings = self.read_fashion_labels(train_label_file, None)
        #
        # img_names = list(set(img_names).intersection(set(label_mappings['img_name'])))
        #
        # label_mappings = label_mappings.loc[label_mappings['img_name'].isin(img_names)]
        #
        # figure_count = len(img_names)
        #
        # print('figure_count::', figure_count)
        
        
        # valid_size = 0.1
        # test_size = 0.2
        #
        # no_conflict_samples = label_mappings.loc[(label_mappings['label_1'] == label_mappings['label_2']) & (label_mappings['label_3'] == label_mappings['label_2']) & (label_mappings['label_gold'] != -1)]
        #
        # conflict_samples = label_mappings.loc[(label_mappings['label_1'] != label_mappings['label_2']) | (label_mappings['label_3'] != label_mappings['label_2'])]
        #
        # no_conflict_samples_img_names = list(no_conflict_samples['img_name'])
        #
        # indices = list(range(no_conflict_samples.shape[0]))
        #
        # split1 = int(np.floor(valid_size * figure_count))
        # split2 = int(np.floor(test_size * figure_count))
        # shuffle = True
        # if shuffle:
            # np.random.seed(0)
            # np.random.shuffle(indices)
        # valid_sample_names, test_sample_names = no_conflict_samples.iloc[indices[split2:split1+split2]], no_conflict_samples.iloc[indices[:split2]]
        #
        # train_sample_names1 = no_conflict_samples.iloc[indices[split1 + split2:]]
        #
        # # train_sample_names =  list(set(img_names).difference(set(valid_sample_names)).difference(set(test_sample_names)))
        # train_sample_names = pd.concat([train_sample_names1, conflict_samples])
        # train_idx = set(range(figure_count)).difference(valid_idx).difference(test_idx)
        
        train_dataset = self.DataSet_fact(data_dir, 'train')
        
        valid_dataset = self.DataSet_fact(data_dir, 'valid')
        
        test_dataset = self.DataSet_fact(data_dir, 'test')
        
        torch.save(train_dataset, os.path.join(data_dir, 'train_dataset'))
    
        torch.save(valid_dataset, os.path.join(data_dir, 'valid_dataset'))
        
        torch.save(test_dataset, os.path.join(data_dir, 'test_dataset'))
        
        return train_dataset, valid_dataset, test_dataset
    
    
    
    
    def prepare_test_fact(self, data_dir, bz):
        test_label_file = data_dir + '/testLabels.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        img_transform = transforms.Compose([
#                 transforms.RandomCrop((587,587)),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        
        test_data_dir = os.path.join(data_dir, 'test/')
        
        figure_count, img_names = self.get_file_count(test_data_dir)
        
        print('figure_count::', figure_count)
        
        indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        test_dataset = self.DataSet_retina(test_data_dir, test_label_file, img_transform, img_names, indices)
        
#         valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
        
        
        test_DL = DataLoader(test_dataset, batch_size=bz)
        
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return test_DL
    


    def get_num_class_fact(self):
        return 2


    def get_num_class_twitter(self):
        return 2
   
    class DataSet_fashion(Dataset):
        def __init__(self, data_dir, label_file, transform, file_names, label_mappings, img_sub_dir_mapppings):
            
#             self.label_mapping = {'': 0, '-1.0': 1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            self.label_mappings = label_mappings
            # self.label_mappings = self.read_fashion_labels(label_file, file_names)
            
            # label_id_list = list(self.label_mappings.keys())
            #
            
            self.img_names = file_names
            # self.img_names = list(self.label_mappings['img_name'])
            # self.img_names = [self.img_names[index] for index in idx]
            
            
            self.transform = transform
            
            self.lenth = len(self.img_names)
            
            self.img_sub_dir_mapppings = img_sub_dir_mapppings
            
#             self.data, self.labels = self.read_retina_images() 
            
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def scalRadius(self, img,scale):
            x = img[int(img.shape[0]/2),:,:].sum(1)
            r = (x>x.mean()/10).sum()/2
            s = scale*1.0/r
            return cv2.resize(img,None,fx=s,fy=s)
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            
#             return self.data[idx], self.labels[idx]
            sub_dir = self.img_sub_dir_mapppings[self.img_names[idx]]

            img_loc = os.path.join(os.path.join(self.data_dir, sub_dir), self.img_names[idx] + '.jpg')
            labels = torch.tensor(list(self.label_mappings.loc[self.label_mappings['img_name'] == self.img_names[idx], 'label_gold'])).view(-1)
            
            print(img_loc, idx, self.img_names[idx], labels)
            
            image = Image.open(img_loc).convert("RGB")
            
#             image = cv2.imread(img_loc)
# 
#             print(img_loc, image.shape)
#             
#             scale = 300
# #             a = cv2.imread(f)
#             a = self.scalRadius(image,scale)
#             a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
#             b = np.zeros(a.shape)
#             cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#             a = a*b+128*(1-b)
#             aa = cv2.resize(a,(512,512))
            
            
            tensor_image =self.transform(image)
#             return tensor_image.type(torch.DoubleTensor), (labels >= 3).type(torch.LongTensor), idx
            return tensor_image.type(torch.DoubleTensor), labels.type(torch.LongTensor), idx
        
        def read_fashion_images(self):
            
            all_data = []
            
            all_labels = []
            
            for i in range(self.lenth):
                print('index::', i)
                img_loc = os.path.join(self.data_dir, self.img_names[i] + '.jpeg')
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                all_data.append(tensor_image)
                all_labels.append(self.label_mappings[self.img_names[i]])
                
                
            return torch.stack(all_data), torch.tensor(all_labels)
                 
                

    def read_fashion_labels(self, label_file, file_names = None):

#         observe_ids = [13, 8, ]
#             key_prefix = 'CheXpert-v1.0-small/train/'

#             observe_ids = []
#     
#             selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
        df = pd.read_csv(label_file)
        
        df['label_1'] = df['T1_Q1'].apply(replace_annotation_with_labels)
        
        df['label_2'] = df['T2_Q1'].apply(replace_annotation_with_labels)
        
        df['label_3'] = df['T3_Q1'].apply(replace_annotation_with_labels)
        
        df['label_gold'] = df['Majority Q1'].apply(replace_annotation_with_labels)
        
        df['img_name'] = df['PictureURL'].apply(replace_url_with_image_name)
        # return df[['img_name', 'T1_Q1','T2_Q1','T3_Q1','Majority Q1']] 
    
        
        if file_names is not None:
            mappings = df.loc[df['img_name'].isin(file_names), ['img_name', 'label_1','label_2','label_3','label_gold']]
        else:
            mappings = df[['img_name', 'label_1','label_2','label_3','label_gold']]
        
        # with open(label_file) as csv_file:
            # csv_reader = csv.reader(csv_file, delimiter=',')
            # line_count = 0
            # for row in csv_reader:
                # if line_count == 0:
                #
# #                         for name in selected_observation:
# #                             id = [i for i,x in enumerate(row) if x == name]
# #                             
# #                             observe_ids.extend(id)
                    #
                    # line_count += 1
                    # continue
                # else:
                #
                    # curr_key = row[0]
                    #
                    # if not curr_key in img_names:
                        # continue
                    # mappings[curr_key] = torch.tensor(int(row[1]))
                    # line_count += 1
        
        return mappings


    def get_fashion_image_count(self, train_data_dir):
        
        img_names = []
        
        img_count = 0
        
        img_sub_dir_mapppings = {}
        
        for sub_folder in os.listdir(train_data_dir):
            
            full_dir = os.path.join(train_data_dir, sub_folder)
            
            if os.path.isdir(full_dir):
                
                for image_files in os.listdir(full_dir):
                    if image_files.endswith('jpg'):
                        
                        # print(image_files)
                        
                        try:
                            Image.open(os.path.join(full_dir, image_files)).convert("RGB")
                            
                        except:
                            # os.remove()
                            continue
                        
                        curr_img_name = image_files[0:-len('.jpg')]
                        
                        img_names.append(curr_img_name)
                        
                        img_sub_dir_mapppings[curr_img_name] = sub_folder
                        
                        img_count += 1
                
        return img_count, img_names, img_sub_dir_mapppings

    # def select_fashion_valid_test_ids(self, no_conflict_samples, count):
    #
    #
        # no_conflict_samples
    
    def prepare_fashion(self, data_dir, bz, load = False):
        if load and os.path.exists(os.path.join(data_dir, 'train_DL')) and os.path.exists(os.path.join(data_dir, 'valid_DL')) and os.path.exists(os.path.join(data_dir, 'test_DL')) and os.path.exists(os.path.join(data_dir, 'train_sample_names')) and os.path.exists(os.path.join(data_dir, 'valid_sample_names')) and os.path.exists(os.path.join(data_dir, 'test_sample_names')):
            train_DL = torch.load(os.path.join(data_dir, 'train_DL'))
    
            valid_DL = torch.load(os.path.join(data_dir, 'valid_DL'))
            
            test_DL = torch.load(os.path.join(data_dir, 'test_DL'))
            
            train_sample_names = load_obj(os.path.join(data_dir, 'train_sample_names'))
        
            valid_sample_names = load_obj(os.path.join(data_dir, 'valid_sample_names'))
            
            test_sample_names = load_obj(os.path.join(data_dir, 'test_sample_names'))
        
            return train_DL, valid_DL, test_DL, train_sample_names, valid_sample_names, test_sample_names
        
        train_label_file = data_dir + '/Annotations/Annotation_PerImage_All.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        # if test:
            # img_transform = transforms.Compose([
                # transforms.Resize(64),
                # transforms.CenterCrop(10),
                # transforms.ToTensor(),
                # transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
            # ])
        # else:
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop((64,64)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
#         else:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
        
        # train_data_dir = os.path.join(data_dir, 'train/')
        train_data_dir = os.path.join(data_dir, 'Fashion10000/Photos')
        
        _, img_names, img_sub_dir_mapppings = self.get_fashion_image_count(train_data_dir)
        
        
        label_mappings = self.read_fashion_labels(train_label_file, img_names)
        # label_mappings = self.read_fashion_labels(train_label_file, None)
        
        img_names = list(set(img_names).intersection(set(label_mappings['img_name'])))
        
        label_mappings = label_mappings.loc[label_mappings['img_name'].isin(img_names)]

        figure_count = len(img_names)

        print('figure_count::', figure_count)
        
        
        valid_size = 0.05
        test_size = 0.05
        
        # no_conflict_samples = label_mappings.loc[(label_mappings['label_1'] == label_mappings['label_2']) & (label_mappings['label_3'] == label_mappings['label_2']) & (label_mappings['label_gold'] != -1)]
        #
        # conflict_samples = label_mappings.loc[(label_mappings['label_1'] != label_mappings['label_2']) | (label_mappings['label_3'] != label_mappings['label_2'])]
        
        no_conflict_samples = label_mappings.loc[(label_mappings['label_gold'] != -1)]
        
        no_conflict_samples_img_names = list(no_conflict_samples['img_name'])
        
        indices = list(range(no_conflict_samples.shape[0]))
        
        split1 = int(np.floor(valid_size * figure_count))
        split2 = int(np.floor(test_size * figure_count))
        shuffle = True
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(indices)
        valid_sample_names, test_sample_names = no_conflict_samples.iloc[indices[split2:split1+split2]], no_conflict_samples.iloc[indices[:split2]]
        
        train_sample_names1 = no_conflict_samples.iloc[indices[split1 + split2:]]
        
        # train_sample_names =  list(set(img_names).difference(set(valid_sample_names)).difference(set(test_sample_names)))
        train_sample_names = train_sample_names1#pd.concat([train_sample_names1, conflict_samples])
        # train_idx = set(range(figure_count)).difference(valid_idx).difference(test_idx)
        
        train_dataset = self.DataSet_fashion(train_data_dir, train_label_file, img_transform, list(train_sample_names['img_name']), label_mappings, img_sub_dir_mapppings)
        
        valid_dataset = self.DataSet_fashion(train_data_dir, train_label_file, img_transform, list(valid_sample_names['img_name']), label_mappings, img_sub_dir_mapppings)
        
        test_dataset = self.DataSet_fashion(train_data_dir, train_label_file, img_transform, list(test_sample_names['img_name']), label_mappings, img_sub_dir_mapppings)
        
        
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        test_DL = DataLoader(test_dataset, batch_size=bz)
        
        torch.save(train_DL, os.path.join(data_dir, 'train_DL'))
    
        torch.save(valid_DL, os.path.join(data_dir, 'valid_DL'))
        
        torch.save(test_DL, os.path.join(data_dir, 'test_DL'))
        
        save_obj(train_sample_names, os.path.join(data_dir, 'train_sample_names'))
    
        save_obj(valid_sample_names, os.path.join(data_dir, 'valid_sample_names'))
        
        save_obj(test_sample_names, os.path.join(data_dir, 'test_sample_names'))
        
        
        return train_DL, valid_DL, test_DL, train_sample_names[['img_name']], valid_sample_names[['img_name']], test_sample_names[['img_name']]
    
    
    
    
    def prepare_test_fashion(self, data_dir, bz):
        test_label_file = data_dir + '/testLabels.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        img_transform = transforms.Compose([
#                 transforms.RandomCrop((587,587)),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        
        test_data_dir = os.path.join(data_dir, 'test/')
        
        figure_count, img_names = self.get_file_count(test_data_dir)
        
        print('figure_count::', figure_count)
        
        indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        test_dataset = self.DataSet_retina(test_data_dir, test_label_file, img_transform, img_names, indices)
        
#         valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
        
        
        test_DL = DataLoader(test_dataset, batch_size=bz)
        
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return test_DL
    
    def get_num_class_fashion(self):
        return 2
    
    def get_hyperparameters_fashion(self, parameters, init_lr, regularization_rate=0.0):
    
#         criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        
        optimizer = Adam(parameters, lr=init_lr)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    
    
    def read_mimic_labels(self, label_file):
    
        mappings = {}
        
        with open(label_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    mappings[row[0]] = row[1]
                    line_count += 1
        
        return mappings

    def pre_process_dataset_mimic(self, data_dir, val_data_dir, label_file, valid_label_file):
    
        if check_done_file(data_dir):
            return
        
        
    #     data_preparer = models.Data_preparer()
        
    #     get_label_function=getattr(data_preparer, "read_" + args.dataset + '_labels')
        
        label_mappings = self.read_retina_labels(label_file)
        all_imgs = os.listdir(data_dir)
        
        all_labels = set()
        
        for img in all_imgs:
            
            if os.path.isdir(os.path.join(data_dir, img)):
                continue
            
            
            img_name = img.split('.')[0]
            label = label_mappings[img_name]
            
            target_dir = os.path.join(data_dir, str(label))
            
            src_file = os.path.join(data_dir, img)
            
            if (not label in all_imgs) and (not label in all_labels):
                print(label)
                os.mkdir(target_dir)
                all_labels.add(label)
            
            move(src_file, target_dir)
            
        setup_done_file(data_dir)

    
    class DataSet_mimic(Dataset):
        def __init__(self, data_dir, label_file, transform, full_ids_list, full_dir_list):
            
#             self.label_mapping = {'': 0, '-1.0': 1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            self.label_list, self.id_list = self.read_mimic_labels(label_file, full_ids_list)
            
#             label_id_list = list(self.label_mappings.keys())
            
            self.img_names = full_dir_list#[label_id_list[index] for index in idx]
            
            self.transform = transform
            
            self.lenth = len(self.id_list)
            
#             self.data, self.labels = self.read_retina_images() 
            
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def scalRadius(self, img,scale):
            x = img[int(img.shape[0]/2),:,:].sum(1)
            r = (x>x.mean()/10).sum()/2
            s = scale*1.0/r
            return cv2.resize(img,None,fx=s,fy=s)
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            
#             return self.data[idx], self.labels[idx]
            img_loc = self.img_names[self.id_list[idx]] + '.jpg'
            labels = self.label_list[idx]
            image = Image.open(img_loc).convert("RGB")
            
#             image = cv2.imread(img_loc)
# 
#             print(img_loc, image.shape)
#             
#             scale = 300
# #             a = cv2.imread(f)
#             a = self.scalRadius(image,scale)
#             a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
#             b = np.zeros(a.shape)
#             cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#             a = a*b+128*(1-b)
#             aa = cv2.resize(a,(512,512))
            
            
            tensor_image =self.transform(image)
            
#             self.visualize(tensor_image)
            return tensor_image.type(torch.DoubleTensor), labels.type(torch.LongTensor), idx
        
        def visualize(self, data):
            fig1 = plt.figure(facecolor='white')
            
            data = data.numpy().transpose((1,2,0))
            
            plt.figure(1)
            plt.imshow(data, cmap='gray', vmin=0, vmax=1)
            
            print('here')
        
        def read_mimic_images(self):
            
            all_data = []
            
            all_labels = []
            
            for i in range(self.lenth):
                print('index::', i)
                img_loc = os.path.join(self.data_dir, self.img_names[i] + '.jpeg')
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                all_data.append(tensor_image)
                all_labels.append(self.label_mappings[self.img_names[i]])
                
                
            return torch.stack(all_data), torch.tensor(all_labels)
                 
                
                

        def read_mimic_labels(self, label_file, img_names):
    
#         observe_ids = [13, 8, ]
#             key_prefix = 'CheXpert-v1.0-small/train/'
    
#             observe_ids = []
#     
#             selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
#             mappings = {}
            label_list = []
            
            id_list = []
#             img_name_set = set(img_names)
            
            with open(label_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        
#                         for name in selected_observation:
#                             id = [i for i,x in enumerate(row) if x == name]
#                             
#                             observe_ids.extend(id)
                        
                        line_count += 1
                        continue
                    else:
                        
                        curr_key = row[1]
                        
                        if not curr_key in img_names:
                            continue
                        
                        curr_index = img_names.index(curr_key)
                        id_list.append(curr_index)
                        label_list.append(torch.tensor(int(float(row[-1]))))
                        line_count += 1
            
            print(line_count)
            
            return label_list, id_list




    class DataSet_OCT(Dataset):
        def __init__(self, data_dir, label_file, transform, full_ids_list, full_dir_list):
            
#             self.label_mapping = {'': 0, '-1.0': 1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            self.label_list, self.id_list = self.read_mimic_labels(label_file, full_ids_list)
            
#             label_id_list = list(self.label_mappings.keys())
            
            self.img_names = full_dir_list#[label_id_list[index] for index in idx]
            
            self.transform = transform
            
            self.lenth = len(self.id_list)
            
#             self.data, self.labels = self.read_retina_images() 
            
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def scalRadius(self, img,scale):
            x = img[int(img.shape[0]/2),:,:].sum(1)
            r = (x>x.mean()/10).sum()/2
            s = scale*1.0/r
            return cv2.resize(img,None,fx=s,fy=s)
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            
#             return self.data[idx], self.labels[idx]
            img_loc = self.img_names[self.id_list[idx]] + '.jpg'
            labels = self.label_list[idx]
            image = Image.open(img_loc).convert("RGB")
            
#             image = cv2.imread(img_loc)
# 
#             print(img_loc, image.shape)
#             
#             scale = 300
# #             a = cv2.imread(f)
#             a = self.scalRadius(image,scale)
#             a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
#             b = np.zeros(a.shape)
#             cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#             a = a*b+128*(1-b)
#             aa = cv2.resize(a,(512,512))
            
            
            tensor_image =self.transform(image)
            
#             self.visualize(tensor_image)
            return tensor_image.type(torch.DoubleTensor), labels.type(torch.LongTensor), idx
        
        def visualize(self, data):
            fig1 = plt.figure(facecolor='white')
            
            data = data.numpy().transpose((1,2,0))
            
            plt.figure(1)
            plt.imshow(data, cmap='gray', vmin=0, vmax=1)
            
            print('here')
        
        def read_mimic_images(self):
            
            all_data = []
            
            all_labels = []
            
            for i in range(self.lenth):
                print('index::', i)
                img_loc = os.path.join(self.data_dir, self.img_names[i] + '.jpeg')
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                all_data.append(tensor_image)
                all_labels.append(self.label_mappings[self.img_names[i]])
                
                
            return torch.stack(all_data), torch.tensor(all_labels)
                 
                
                

        def read_mimic_labels(self, label_file, img_names):
    
#         observe_ids = [13, 8, ]
#             key_prefix = 'CheXpert-v1.0-small/train/'
    
#             observe_ids = []
#     
#             selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
#             mappings = {}
            label_list = []
            
            id_list = []
#             img_name_set = set(img_names)
            
            with open(label_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        
#                         for name in selected_observation:
#                             id = [i for i,x in enumerate(row) if x == name]
#                             
#                             observe_ids.extend(id)
                        
                        line_count += 1
                        continue
                    else:
                        
                        curr_key = row[1]
                        
                        if not curr_key in img_names:
                            continue
                        
                        curr_index = img_names.index(curr_key)
                        id_list.append(curr_index)
                        label_list.append(torch.tensor(int(float(row[-1]))))
                        line_count += 1
            
            print(line_count)
            
            return label_list, id_list


    
    def target_transform_OCT(self, origin_target):
        target_mappings = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
        
        return target_mappings[origin_target]
    
    def obtain_transformed_dataset_OCT(self, DL):
        
        all_samples = []
        
        all_target = []
        
        i = 0
        
        for img, target in DL:
            print('data idx::', i)
            
            all_samples.append(img)
            all_target.append(target)
            
            i += img.shape[0]
            
        all_sample_tensor = torch.cat(all_samples, dim = 0)
        
        all_target_tensor = torch.cat(all_target, dim = 0)
        
        transformed_dataset = MyDataset(all_sample_tensor, all_target_tensor)
        
        return transformed_dataset
        
        
        
    
    def prepare_OCT(self, data_dir, bz, origin=False):
        
        
        train_data_file = os.path.join(data_dir, 'train/')
        
        test_data_file = os.path.join(data_dir, 'test/')
        
        
        
#         if origin:
#             train_label_file = data_dir + '/train_' + mimic_type + '.csv'
#         else:
#             train_label_file = data_dir + '/train_' + mimic_type + '_origin.csv'
#         valid_label_file = data_dir + '/valid_' + mimic_type + '.csv'
#         test_label_file = data_dir + '/test_' + mimic_type + '.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
#         if test:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(10),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
#         else:
        img_transform = transforms.Compose([
#                 transforms.RandomResizedCrop((587,587)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = datasets.ImageFolder(train_data_file, transform = img_transform)#, target_transform = self.target_transform_OCT)
        
        test_dataset = datasets.ImageFolder(test_data_file, transform = img_transform)#, target_transform = self.target_transform_OCT)
        
        target_mappings = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
        
        train_dataset.class_to_idx = target_mappings
        
        test_dataset.class_to_idx = target_mappings
#         else:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
        
#         train_data_dir = os.path.join(data_dir, 'files/')
#         
#         full_ids_list, full_dir_list = self.get_mimic_file_count(train_data_dir)
#         
#         figure_count = len(full_ids_list)
#         
#         print('figure_count::', figure_count)
#         
# #         indices = list(range(figure_count))
# #         valid_size = 0.1
# #         split = int(np.floor(valid_size * figure_count))
# #         shuffle = True
# #         if shuffle:
# #             np.random.seed(0)
# #             np.random.shuffle(indices)
# #         train_idx, valid_idx = indices[split:], indices[:split]
#         
#         train_dataset = self.DataSet_mimic(train_data_dir, train_label_file, img_transform, full_ids_list, full_dir_list)
#         
#         print('training data size::', train_dataset.lenth)
#         
#         valid_dataset = self.DataSet_mimic(train_data_dir, valid_label_file, img_transform, full_ids_list, full_dir_list)
# 
#         print('valid data size::', valid_dataset.lenth)
#         
#         test_dataset = self.DataSet_mimic(train_data_dir, test_label_file, img_transform, full_ids_list, full_dir_list)
#         
#         print('test data size::', test_dataset.lenth)
        
                
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        test_DL = DataLoader(test_dataset, batch_size = bz)
        
        transformed_train_dataset = self.obtain_transformed_dataset_OCT(train_DL)
        
        transformed_test_dataset = self.obtain_transformed_dataset_OCT(test_DL)

        random_train_ids = torch.randperm(transformed_train_dataset.data.shape[0])
        
        valid_size = int(random_train_ids.shape[0]*0.1)
        
        valid_ids = random_train_ids[0:valid_size]
        
        train_ids = random_train_ids[valid_size:]

#         indices = list(range(train_dataset.data.shape[0]))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))

        
        valid_data = transformed_train_dataset.data[valid_ids]
        
        valid_labels = transformed_train_dataset.labels[valid_ids] 
        
        train_data = transformed_train_dataset.data[train_ids]
        
        train_labels = transformed_train_dataset.labels[train_ids]
        
        new_train_dataset = MyDataset(train_data, train_labels)
        
        valid_dataset = MyDataset(valid_data, valid_labels)
        
        train_DL = DataLoader(new_train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        test_DL = DataLoader(transformed_test_dataset, batch_size = bz)
#         transformed_test_dataset = self.obtain_transformed_dataset_OCT(test_DL)
        
        
        return train_DL, valid_DL, test_DL
    
    
    
    
#     def prepare_test_OCT(self, data_dir, bz):
# #         test_label_file = data_dir + '/testLabels.csv'
#         train_label_file = data_dir + '/train_' + mimic_type + '.csv'
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#         
#         
# #         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
#         img_transform = transforms.Compose([
# #                 transforms.RandomCrop((587,587)),
#                 transforms.Resize((224, 224)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
#         
#         
#         test_data_dir = os.path.join(data_dir, 'test/')
#         
#         figure_count, img_names = self.get_file_count(test_data_dir)
#         
#         print('figure_count::', figure_count)
#         
#         indices = list(range(figure_count))
# #         valid_size = 0.1
# #         split = int(np.floor(valid_size * figure_count))
# #         shuffle = True
# #         if shuffle:
# #             np.random.seed(0)
# #             np.random.shuffle(indices)
# #         train_idx, valid_idx = indices[split:], indices[:split]
#         
#         test_dataset = self.DataSet_retina(test_data_dir, test_label_file, img_transform, img_names, indices)
#         
# #         valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
#         
#         
#         test_DL = DataLoader(test_dataset, batch_size=bz)
#         
# #         valid_DL = DataLoader(valid_dataset, batch_size=bz)
#         
#         return test_DL
#     
    
    
    def get_hyperparameters_OCT(self, parameters, init_lr, regularization_rate=0.0):
    
#         criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        
        optimizer = Adam(parameters, lr=init_lr)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_OCT(self):
        return 4

    
    def prepare_mimic(self, data_dir, bz, origin=False):
        
        
        if origin:
            train_label_file = data_dir + '/train_' + mimic_type + '.csv'
        else:
            train_label_file = data_dir + '/train_' + mimic_type + '_origin.csv'
        valid_label_file = data_dir + '/valid_' + mimic_type + '.csv'
        test_label_file = data_dir + '/test_' + mimic_type + '.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
#         if test:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(10),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
#         else:
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop((64,64)),
#             transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
#         else:
#             img_transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
#             ])
        
        train_data_dir = os.path.join(data_dir, 'files/')
        
        full_ids_list, full_dir_list = self.get_mimic_file_count(train_data_dir)
        
        figure_count = len(full_ids_list)
        
        print('figure_count::', figure_count)
        
#         indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        train_dataset = self.DataSet_mimic(train_data_dir, train_label_file, img_transform, full_ids_list, full_dir_list)
        
        print('training data size::', train_dataset.lenth)
        
        valid_dataset = self.DataSet_mimic(train_data_dir, valid_label_file, img_transform, full_ids_list, full_dir_list)

        print('valid data size::', valid_dataset.lenth)
        
        test_dataset = self.DataSet_mimic(train_data_dir, test_label_file, img_transform, full_ids_list, full_dir_list)
        
        print('test data size::', test_dataset.lenth)
        
                
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        test_DL = DataLoader(test_dataset, batch_size = bz)
        
        return train_DL, valid_DL, test_DL
    
    
    
    
    def prepare_test_mimic(self, data_dir, bz):
        test_label_file = data_dir + '/testLabels.csv'
        # train_label_file = data_dir + '/train_' + mimic_type + '.csv'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        
#         data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        img_transform = transforms.Compose([
#                 transforms.RandomCrop((587,587)),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        
        test_data_dir = os.path.join(data_dir, 'test/')
        
        figure_count, img_names = self.get_file_count(test_data_dir)
        
        print('figure_count::', figure_count)
        
        indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        test_dataset = self.DataSet_retina(test_data_dir, test_label_file, img_transform, img_names, indices)
        
#         valid_dataset = self.DataSet_retina(train_data_dir, train_label_file, img_transform, img_names, valid_idx)
        
        
        test_DL = DataLoader(test_dataset, batch_size=bz)
        
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return test_DL
    
    
    
    def get_hyperparameters_mimic(self, parameters, init_lr, regularization_rate=0.0):
    
#         criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        
        optimizer = Adam(parameters, lr=init_lr)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_mimic(self):
        return 2
    
    
    def prepare_alarm(self, data_dir, bz, origin=False):
        full_output_dir = os.path.join(data_dir, 'alarm')
    
    #     train_dataset = torch.load(full_output_dir + '/preprocessed_train')
    #     
    #     valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
    #     
    #     test_dataset = torch.load(full_output_dir + '/preprocessed_test')
        
        train_dataset = torch.load(full_output_dir + '/train_dataset')
        
        valid_dataset = torch.load(full_output_dir + '/valid_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
    
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
#         test_DL = DataLoader(test_dataset, batch_size = bz)
        
        return train_DL, valid_DL
    
    def prepare_test_alarm(self, data_dir, bz, origin=False):
        full_output_dir = os.path.join(data_dir, 'alarm')
    
    #     train_dataset = torch.load(full_output_dir + '/preprocessed_train')
    #     
    #     valid_dataset = torch.load(full_output_dir + '/preprocessed_valid')
    #     
    #     test_dataset = torch.load(full_output_dir + '/preprocessed_test')
        
        train_dataset = torch.load(full_output_dir + '/train_dataset')
        
        valid_dataset = torch.load(full_output_dir + '/valid_dataset')
        
        test_dataset = torch.load(full_output_dir + '/test_dataset')
    
#         train_DL = DataLoader(train_dataset, batch_size=bz)
#         
#         valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        test_DL = DataLoader(test_dataset, batch_size = bz)
        
        return test_DL
    
    
    def get_num_class_alarm(self):
        return 2
#     def determine_chexpert_labels(self, label):
#         
# #         print(label)
#         
#         if label == '':
#             label = 0
#         
#         else:
#             if label == '-1.0':
#                 label = 1
#             
#             else:
#                 label = int(float(label))
#         
#         
#         return label
    
    
    class DataSet_chexpert(Dataset):
        def __init__(self, data_dir, label_file, transform, all_pat_dirs):
            
            self.label_mapping = {'': -1, '-1.0': -1, '0.0': 0, '1.0': 1}
            
            self.data_dir = data_dir
            self.label_mappings = self.read_chexpert_labels(label_file, all_pat_dirs)
            
            self.img_names = list(self.label_mappings.keys())
            
            self.transform = transform
            
            self.lenth = len(self.img_names)
#             all_imgs = os.listdir(main_dir)
#             self.total_imgs = natsort.natsorted(all_imgs)
    
        def __len__(self):
            return self.lenth
    
        def __getitem__(self, idx):
#             img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            img_loc = os.path.join(self.data_dir, self.img_names[idx])
            labels = self.label_mappings[self.img_names[idx]]
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image.type(torch.DoubleTensor), labels, idx
        
        def get_pat_ids(self, key):
            id = key.index('/')
            return key[0:id]
        
        def read_chexpert_labels(self, label_file, all_pat_dirs):
    
#         observe_ids = [13, 8, ]
            key_prefix = 'CheXpert-v1.0-small/train/'
    
            observe_ids = []
    
            selected_observation = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
            mappings = {}
            
#             unique_labels = []
#             
#             for k in range(len(selected_observation)):
#                 unique_labels.append(set())
            
            
            
            with open(label_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        
                        for name in selected_observation:
                            id = [i for i,x in enumerate(row) if x == name]
                            
                            observe_ids.extend(id)
                        
                        line_count += 1
                    else:
                        
                        curr_labels = []
                        
                        for r in range(len(observe_ids)):
                            
                            curr_label = row[observe_ids[r]]
                            
                            curr_label = self.label_mapping[curr_label]
    
#                             curr_label = self.determine_chexpert_labels(curr_label)
                            
    #                         try:
    #                             if int(row[observe_ids[r]]) < 0:
    #                                 curr_label = 0
    #                         except:
    #                             pass
                            
                            curr_labels.append(curr_label)
#                             unique_labels[r].add(curr_label)
                        
                        curr_key = row[0][len(key_prefix):]
                        
                        assert curr_key.startswith("patient")
                        
                        pat_id = self.get_pat_ids(curr_key)
                        
                        if not pat_id in all_pat_dirs:
                            continue
                        
                        mappings[curr_key] = torch.tensor(curr_labels)
                        line_count += 1
            
            return mappings
    
    
    def get_all_patients_chexpert(self, data_dir):
        patient_dirs = os.listdir(data_dir)
        
#         full_patient_dirs = []
        
#         for dir in patient_dirs:
#             full_patient_dirs.append(flag, dir)
        
        return set(patient_dirs)
        
    
    def pre_process_dataset_chexpert(self, data_dir, val_data_dir, label_file, valid_label_file):
    
        if check_done_file(data_dir):
            return
        
        
    #     data_preparer = models.Data_preparer()
        
    #     get_label_function=getattr(data_preparer, "read_" + args.dataset + '_labels')
        
        label_mappings = self.read_chexpert_labels(label_file)
        
        val_label_mappings = self.read_chexpert_labels(valid_label_file)
        
        
        
        all_imgs = os.listdir(data_dir)
        
        all_labels = set()
        
        for img in all_imgs:
            
            if os.path.isdir(os.path.join(data_dir, img)):
                continue
            
            
            img_name = img.split('.')[0]
            label = label_mappings[img_name]
            
            target_dir = os.path.join(data_dir, str(label))
            
            src_file = os.path.join(data_dir, img)
            
            if (not label in all_imgs) and (not label in all_labels):
                print(label)
                os.mkdir(target_dir)
                all_labels.add(label)
            
            move(src_file, target_dir)
            
        setup_done_file(data_dir)
    
    
    
    def prepare_chexpert(self, data_dir, bz):
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        train_label_file = data_dir + '/train.csv'
        
        valid_label_file = data_dir + '/valid.csv'
        
#         train_label_mappings = self.read_retina_labels(train_label_file)
        
#         figure_count = self.get_file_count(data_dir)
#         
#         print('total figure count::', figure_count)
#         
#         indices = list(range(figure_count))
#         valid_size = 0.1
#         split = int(np.floor(valid_size * figure_count))
#         shuffle = True
#         if shuffle:
#             np.random.seed(0)
#             np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
        
        
#         test_label_file = data_dir + '/testLabels.csv'
#         
#         test_label_mappings = self.read_retina_labels(test_label_file)
        
#         datadir = '/data/fundus/train'
#         datadir = self.data_dir
        img_transform = transforms.Compose([
#                 transforms.RandomCrop((224,224)),
                
                transforms.Resize((64, 64)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        
        train_data_dir = os.path.join(data_dir, 'train/')
        
        valid_data_dir = os.path.join(data_dir, 'valid/')
        
        train_all_pat_dirs = self.get_all_patients_chexpert(train_data_dir)
        
        valid_all_pat_dirs = self.get_all_patients_chexpert(valid_data_dir)
        
        train_dataset = self.DataSet_chexpert(data_dir + '/train/', train_label_file, img_transform, train_all_pat_dirs)
        
        valid_dataset = self.DataSet_chexpert(data_dir + '/valid/', valid_label_file, img_transform, valid_all_pat_dirs)
        
#         train_dataset = torch.utils.data.Subset(dataset, train_idx)
#         
#         valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
        
#         if self.isGPU:
#             dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#              
#             train_DL = DataLoader(train_dataset, sampler=dist_sampler, batch_size=16, num_workers=4)
#         else:
        train_DL = DataLoader(train_dataset, batch_size=bz)
        
        valid_DL = DataLoader(valid_dataset, batch_size=bz)
        
        return train_DL, valid_DL
        
    
    
    def get_hyperparameters_chexpert(self, parameters, init_lr, regularization_rate=0.0):
    
#         criterion = nn.NLLLoss()
        criterion = nn.BCEWithLogitsLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        
        optimizer = Adam(parameters, lr=init_lr)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_chexpert(self):
        return 2


    
    # def prepare_MNIST2(self):
    #
    #
        # resnet = models.resnet50(pretrained=True)
        # # freeze all model parameters
        # for param in resnet.parameters():
            # param.requires_grad = False
            #
            #
# #         print(resnet.fc.in_features)
        #
        #
        # mnist = MNIST(git_ignore_folder + '/mnist', download=True, train=False).train_data.float()
        #
# #         data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.float().mean()/255,), (mnist.float().std()/255,))])
        #
        # data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])
        #
        # train_data = MNIST(git_ignore_folder + '/mnist',
                   # download=True,
                   # transform=data_transform)
                   #
        # test_data = MNIST(git_ignore_folder + '/mnist',
                      # train=False,
                      # download=True,
                      # transform=data_transform)
                      #
                      #
# #         train_X = train_data.transforms.transform.transforms[0](train_data.data.data.numpy())
        #
# #         train_X = data_transform(train_data)
#
        # train_X = self.compose_train_test_data(train_data, resnet)
        #
# #         train_X = train_data.transforms.transform(transforms.ToPILImage()(train_data.data.float()))
        #
        # train_X = train_X.transpose(0,1).transpose(1,2)
        #
        # train_Y = train_data.targets
        #
        #
        # test_data = MNIST(git_ignore_folder + '/mnist',
                      # train=False,
                      # download=True,
                      # transform=transforms.Compose([
# #                           transforms.Resize((32, 32)),
                          # transforms.ToTensor()]))
                          #
                          #
# #         test_X = test_data.transforms.transform.transforms[0](test_data.data.data.numpy())
        # test_X = data_transform(test_data.data.data.numpy())
        #
        # test_X = test_X.transpose(0,1).transpose(1,2)
        #
        # test_Y = test_data.targets
        #
        # return train_X.type(torch.DoubleTensor).view(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2]), train_Y.view(train_X.shape[0],-1), test_X.type(torch.DoubleTensor).view(test_X.shape[0], 1, test_X.shape[1], test_X.shape[2]), test_Y.view(test_X.shape[0], -1)
        #
    
    
    def get_hyperparameters_MNIST2(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    # def prepare_MNIST3(self):
        # train_data = MNIST(git_ignore_folder + '/mnist',
                   # download=True,
                   # transform=transforms.Compose([
# #                        transforms.Resize((32, 32)),
                       # transforms.ToTensor()]))
                       #
        # test_data = MNIST(git_ignore_folder + '/mnist',
                      # train=False,
                      # download=True,
                      # transform=transforms.Compose([
# #                           transforms.Resize((32, 32)),
                          # transforms.ToTensor()]))
                          #
        # return train_data, test_data
        
    
    def get_hyperparameters_MNIST3(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler    


    # def prepare_MNIST4(self):
        # train_data = MNIST(git_ignore_folder + '/mnist',
                   # download=True,
                   # transform=transforms.Compose([
# #                        transforms.Resize((32, 32)),
                       # transforms.ToTensor()]))
                       #
        # test_data = MNIST(git_ignore_folder + '/mnist',
                      # train=False,
                      # download=True,
                      # transform=transforms.Compose([
# #                           transforms.Resize((32, 32)),
                          # transforms.ToTensor()]))
                          #
        # return train_data, test_data
        #
    
    
    def get_hyperparameters_MNIST4(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler

    
    def prepare_MNIST(self, git_ignore_folder):
        
#         configs = load_config_data(config_file)
#     
# #     print(configs)
#         git_ignore_folder = configs['git_ignore_folder']
        
        
        train_data = MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        train_X = train_data.transforms.transform.transforms[0](train_data.data.data.numpy())
        
        train_X = train_X.transpose(0,1).transpose(1,2)
        
        train_Y = train_data.targets
        
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        
        test_X = test_data.transforms.transform.transforms[0](test_data.data.data.numpy())
        
        test_X = test_X.transpose(0,1).transpose(1,2)
        
        test_Y = test_data.targets
        
        
        train_X = train_X.reshape([train_X.shape[0], -1])
        
        
        train_X = train_X.type(torch.DoubleTensor)
        
        test_X = test_X.type(torch.DoubleTensor)
        
        test_X = test_X.reshape([test_X.shape[0], -1])
        
#         print(train_X.shape)

        
#         train_X = extended_by_constant_terms(train_X, False)
#         
#         test_X = extended_by_constant_terms(test_X, False)
        
#         torch.save(train_X, git_ignore_folder + 'noise_X')
#         
#         torch.save(train_Y, git_ignore_folder + 'noise_Y')
        
        
        return train_X, train_Y, test_X, test_Y
        
    
    
    def get_hyperparameters_MNIST(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_MNIST(self):
        return 10
    
    
    
    # def prepare_covtype(self):
    #
        # configs = load_config_data(config_file)
        #
# #     print(configs)
        # git_ignore_folder = configs['git_ignore_folder']
        #
        # directory_name = configs['directory']
        #
        # train_X, train_Y, test_X, test_Y = load_data_multi_classes(True, directory_name + "covtype")
        #
        # train_Y = train_Y.view(-1)
        #
        # test_Y = test_Y.view(-1)
        #
        # train_X = extended_by_constant_terms(train_X, False)
        #
        # test_X = extended_by_constant_terms(test_X, False)
        #
        # torch.save(train_X, git_ignore_folder + 'noise_X')
        #
        # torch.save(train_Y, git_ignore_folder + 'noise_Y')
        #
# #         train_data = MNIST(git_ignore_folder + '/mnist',
# #                    download=True,
# #                    transform=transforms.Compose([
# # #                         transforms.Resize((32, 32)),
# #                        transforms.ToTensor()]))
# #         
# #         test_data = MNIST(git_ignore_folder + '/mnist',
# #                       train=False,
# #                       download=True,
# #                       transform=transforms.Compose([
# # #                         transforms.Resize((32, 32)),
# #                           transforms.ToTensor()]))
        #
        # return train_X, train_Y.type(torch.LongTensor), test_X, test_Y.type(torch.LongTensor)
        #
    
    
    def get_hyperparameters_covtype(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    
    # def prepare_higgs(self, git_ignore_folder):
    #
    #
    #
        # if not os.path.exists(git_ignore_folder):
            # os.makedirs(git_ignore_folder)
            #
        # if not os.path.exists(git_ignore_folder + '/higgs'):
            # os.makedirs(git_ignore_folder + '/higgs')
            #
        # curr_file_name = git_ignore_folder + '/higgs/HIGGS'
        #
        # if not os.path.exists(git_ignore_folder + '/higgs/HIGGS.bz2'):
            # print('start downloading higgs dataset')
            # url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2'
            # r = requests.get(url, allow_redirects=True)
            #
            # open(curr_file_name + '.bz2', 'wb').write(r.content)
            # print('end downloading higgs dataset')
            #
            # print('start uncompressing higgs dataset')
            # zipfile = bz2.BZ2File(curr_file_name + '.bz2') # open the file
            # data = zipfile.read() # get the decompressed data
# #             newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
            # open(curr_file_name, 'wb').write(data) # write a uncompressed file
            #
            # print('end uncompressing higgs dataset')
            #
            #
            #
# #         if not os.path.exists(git_ignore_folder + '/rcv1/rcv1_test.binary'):
# #             print('start downloading rcv1 test dataset')
# #             url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
# #             r = requests.get(url, allow_redirects=True)
# #             curr_file_name = git_ignore_folder + 'rcv1/rcv1_test.binary'
# #             open(curr_file_name + '.bz2', 'wb').write(r.content)
# #             print('end downloading rcv1 test dataset')
# #             
# #             print('start uncompressing rcv1 test dataset')
# #             zipfile = bz2.BZ2File(curr_file_name + '.bz2') # open the file
# #             data = zipfile.read() # get the decompressed data
# # #             newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
# #             open(curr_file_name, 'wb').write(data) # write a uncompressed file
# #             print('end uncompressing rcv1 test dataset')
        #
        #
        #
        #
        #
# #         configs = load_config_data(config_file)
# #     
# # #     print(configs)
# #         git_ignore_folder = configs['git_ignore_folder']
# #         
# #         directory_name = configs['directory']
        #
        # num_feature = 28
        #
        # train_X, train_Y, test_X, test_Y =  clean_sensor_data0(git_ignore_folder + 'higgs/HIGGS', True, num_feature, -500000)
        #
# #         train_X, train_Y, test_X, test_Y = load_data_multi_classes(, , )
        #
        # train_Y = train_Y.view(-1)
        #
        # test_Y = test_Y.view(-1)
        #
        # train_X = extended_by_constant_terms(train_X, False)
        #
        # test_X = extended_by_constant_terms(test_X, False)
        #
# #         torch.save(train_X, git_ignore_folder + 'noise_X')
# #         
# #         torch.save(train_Y, git_ignore_folder + 'noise_Y')
        #
        #
        # print(train_X.shape)
        #
        # print(test_X.shape)
# #         train_data = MNIST(git_ignore_folder + '/mnist',
# #                    download=True,
# #                    transform=transforms.Compose([
# # #                         transforms.Resize((32, 32)),
# #                        transforms.ToTensor()]))
# #         
# #         test_data = MNIST(git_ignore_folder + '/mnist',
# #                       train=False,
# #                       download=True,
# #                       transform=transforms.Compose([
# # #                         transforms.Resize((32, 32)),
# #                           transforms.ToTensor()]))
        #
        # return train_X, train_Y.type(torch.LongTensor), test_X, test_Y.type(torch.LongTensor)
        #
    
    
    def get_hyperparameters_higgs(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_higgs(self):
        return 2
    
    def get_num_class_covtype(self):
        return 7
    
    
    
    
    
    
    def prepare_rcv1(self, git_ignore_folder):
        
        
        
        if not os.path.exists(git_ignore_folder):
            os.makedirs(git_ignore_folder)
            
        if not os.path.exists(git_ignore_folder + '/rcv1'):
            os.makedirs(git_ignore_folder + '/rcv1')
        
        
        if not os.path.exists(git_ignore_folder + '/rcv1/rcv1_train.binary'):
            print('start downloading rcv1 training dataset')
            url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
            r = requests.get(url, allow_redirects=True)
            curr_file_name = git_ignore_folder + 'rcv1/rcv1_train.binary'
            open(curr_file_name + '.bz2', 'wb').write(r.content)
            print('end downloading rcv1 training dataset')
            
            print('start uncompressing rcv1 training dataset')
            zipfile = bz2.BZ2File(curr_file_name + '.bz2') # open the file
            data = zipfile.read() # get the decompressed data
#             newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
            open(curr_file_name, 'wb').write(data) # write a uncompressed file
                        
            print('end uncompressing rcv1 training dataset')
            
            
        
        if not os.path.exists(git_ignore_folder + '/rcv1/rcv1_test.binary'):
            print('start downloading rcv1 test dataset')
            url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
            r = requests.get(url, allow_redirects=True)
            curr_file_name = git_ignore_folder + 'rcv1/rcv1_test.binary'
            open(curr_file_name + '.bz2', 'wb').write(r.content)
            print('end downloading rcv1 test dataset')
            
            print('start uncompressing rcv1 test dataset')
            zipfile = bz2.BZ2File(curr_file_name + '.bz2') # open the file
            data = zipfile.read() # get the decompressed data
#             newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
            open(curr_file_name, 'wb').write(data) # write a uncompressed file
            print('end uncompressing rcv1 test dataset')
#         configs = load_config_data(config_file)
    
#     print(configs)
#         git_ignore_folder = configs['git_ignore_folder']
        
#         directory_name = configs['directory']
        
        X_train, y_train = load_svmlight_file(git_ignore_folder + "/rcv1/rcv1_train.binary")
        
#         X_test, y_test = load_svmlight_file(git_ignore_folder + "/rcv1/rcv1_test.binary")
        
        
        train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)
        
        train_Y = torch.from_numpy(y_train).type(torch.DoubleTensor).view(y_train.shape[0], -1)
        
        train_Y = (train_Y + 1)/2
        
#         test_X = torch.from_numpy(X_test.todense()).type(torch.DoubleTensor)
#         
#         test_Y = torch.from_numpy(y_test).type(torch.DoubleTensor).view(y_test.shape[0], -1)
#         
#         test_Y = (test_Y + 1)/2        
        
        
#         train_X, train_Y, test_X, test_Y = load_data_multi_classes_rcv1()
        
#         train_X, train_Y = load_data_multi_classes_single(True, directory_name + "rcv1_test.multiclass")
#         
#         test_X, test_Y = load_data_multi_classes_single(True, directory_name + "rcv1_train.multiclass")
#         
# #         train_X, train_Y, test_X, test_Y = load_data_multi_classes(True, "../../../data/covtype")
#         
#         
#         train_X = extended_by_constant_terms(train_X, False)
#         
#         test_X = extended_by_constant_terms(test_X, False)
        
#         torch.save(train_X, git_ignore_folder + 'noise_X')
#         
#         torch.save(train_Y, git_ignore_folder + 'noise_Y')
#         train_data = MNIST(git_ignore_folder + '/mnist',
#                    download=True,
#                    transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
#         
#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        return train_X, train_Y.type(torch.LongTensor), train_X, train_Y.type(torch.LongTensor)
        
    
    
    def get_hyperparameters_rcv1(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_rcv1(self):
        return 2
    
    
    
    # def prepare_FashionMNIST(self):
        # train_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                   # download=True,
                   # transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
                       # transforms.ToTensor()]))
                       #
        # test_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                      # train=False,
                      # download=True,
                      # transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
                          # transforms.ToTensor()]))
                          #
        # return train_data, test_data
        #
    
    
    def get_hyperparameters_FashionMNIST(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    # def prepare_FashionMNIST2(self):
        # train_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                   # download=True,
                   # transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
                       # transforms.ToTensor()]))
                       #
        # test_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                      # train=False,
                      # download=True,
                      # transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
                          # transforms.ToTensor()]))
                          #
        # return train_data, test_data
        #
    
    
    def get_hyperparameters_FashionMNIST2(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def transform_cifar10(self, data, transform, transform_num):
#         data_array = np.array(data)
        data_array = data
        
        transformed_tensor = []
        
        for i in range(data_array.shape[0]):
            print('data::', i)
            curr_array = transforms.ToPILImage()(data_array[i])
            
            for k in range(transform_num):
                curr_new_array = transform.transforms[k](curr_array)
                
                del curr_array
                
                curr_array = curr_new_array
            
            transformed_tensor.append(curr_array)
            
        return torch.stack(transformed_tensor, dim = 0)
    
    
    def prepare_cifar10(self, git_ignore_folder):
        
        
#         configs = load_config_data(config_file)
#     
# #     print(configs)
# #         git_ignore_folder = configs['git_ignore_folder']
# #         
# #         directory_name = configs['directory']
#         
#         X_train, y_train = load_svmlight_file(directory_name + "cifar10")
#         
#         X_test, y_test = load_svmlight_file(directory_name + "cifar10.t")
#         
#         
#         train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)/255
#         
#         train_Y = torch.from_numpy(y_train).type(torch.LongTensor).view(y_train.shape[0], -1)
#         
#         test_X = torch.from_numpy(X_test.todense()).type(torch.DoubleTensor)/255
#         
#         test_Y = torch.from_numpy(y_test).type(torch.LongTensor).view(y_test.shape[0], -1)
        
        
        
#         transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#         transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar_dir = git_ignore_folder + '/cifar10'

#         if os.path.exists(os.path.join(cifar_dir, 'train_X')):
#             train_X = torch.load(os.path.join(cifar_dir, 'train_X')).type(torch.DoubleTensor)
#             
#             test_X = torch.load(os.path.join(cifar_dir, 'test_X')).type(torch.DoubleTensor)
# 
#             train_Y = torch.load(os.path.join(cifar_dir, 'train_Y'))
#             
#             test_Y = torch.load(os.path.join(cifar_dir, 'test_Y'))
#             
#             test_Y = torch.tensor(test_Y)
#         
#             train_Y = torch.tensor(train_Y)
#             
#             return train_X, train_Y, test_X, test_Y

        data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                       download=True,
                       transform=transform_train)
        data_test = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                          train=False,
                          download=True,
                          transform=transform_test)
        
        total_ids = torch.randperm(data_train.data.data.shape[0])
        
        training_id_count = int(data_train.data.data.shape[0]*0.1)
        
        valid_ids = total_ids[0:training_id_count]
        
        training_ids = total_ids[training_id_count:]
        
        train_X = self.transform_cifar10(np.array(data_train.data.data)[training_ids.numpy()], transform_train, len(transform_train.transforms)).type(torch.DoubleTensor)
        
        train_Y = torch.tensor(np.array(data_train.targets)[training_ids.numpy()])
        
        print('train_X shape::,', train_X.shape, train_Y.shape)
        
        valid_X = self.transform_cifar10(np.array(data_train.data.data)[valid_ids.numpy()], transform_test, len(transform_test.transforms)).type(torch.DoubleTensor)

        valid_Y = torch.tensor(np.array(data_train.targets)[valid_ids.numpy()])
        
        print('valid_X shape::,', valid_X.shape, valid_Y.shape)
        
        test_X = self.transform_cifar10(np.array(data_test.data.data), transform_test, len(transform_test.transforms)).type(torch.DoubleTensor)
        
        print('test_X shape::', test_X.shape)
        
        test_Y = torch.tensor(data_test.targets)
        
        
        torch.save(train_X, os.path.join(cifar_dir, 'train_X'))
            
        torch.save(test_X, os.path.join(cifar_dir, 'test_X'))

        torch.save(train_Y, os.path.join(cifar_dir, 'train_Y'))
            
        torch.save(test_Y, os.path.join(cifar_dir, 'test_Y'))
        
        torch.save(valid_Y, os.path.join(cifar_dir, 'valid_Y'))
            
        torch.save(valid_X, os.path.join(cifar_dir, 'valid_X'))

#         data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                        ]))
#         data_test = torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                           ]))

        
        return train_X, train_Y, (valid_X, test_X), (valid_Y, test_Y)
        
    
    
    def get_hyperparameters_cifar10(self, parameters, init_lr, regularization_rate):
    
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def get_num_class_cifar10(self):
        return 10
    
    def compute_output_before_last_layer(self, input, tl, transfer_model, transfer_model_modules, get_transfer_model_func, last_layer, is_GPU, device):
        
        this_input = input.clone()
        
        if is_GPU:
            this_input = this_input.to(device)
#         expect_output = model.forward(input)
#         for i in range(len(list(model.children()))-1):
#         for i in range(len(transfer_model_modules) - 1):
#             output = transfer_model_modules[i].double()(this_input)
#             
#             del this_input
#             
#             this_input = output
        
        
        output = get_transfer_model_func(tl, transfer_model, this_input)
        
        del this_input
        
        
        
#         if len(output.shape) > 2 or (not output.shape[1] == in_feature_num):
#             output = torch.flatten(this_input, 1)
#             
#             del this_input
        
        expect_output = last_layer.double()(output)
        
        
        output_cpu = output.to('cpu') 
        
        expect_output_cpu =expect_output.to('cpu')
        
        del output, expect_output
        
        
        return output_cpu,expect_output_cpu 
            
            
        
    
    def compose_train_test_data(self, data_train, resnet):
        
        train_X = []
        
        for i in range(data_train.data.shape[0]):
            curr_train_X = data_train.transforms.transform(data_train.data[i])
            
            curr_transformed_X, _ = self.compute_output_before_last_layer(curr_train_X.view(1, curr_train_X.shape[0], curr_train_X.shape[1], curr_train_X.shape[2]), resnet)
            
#             curr_transformed_X = resnet(curr_train_X.view(1, curr_train_X.shape[0], curr_train_X.shape[1], curr_train_X.shape[2]))
#             
#             print(i)
            
            train_X.append(curr_transformed_X)
            
        return torch.stack(train_X, 0)
            
    
    
    def normalize(self, data):
    
        print('normalization start!!')
        
        x_max,_ = torch.max(data, axis = 0)
        
        x_min,_ = torch.min(data, axis = 0)
        
        range = x_max - x_min
        
        update_data = data[:,range != 0] 
        
        
    #     print(average_value.shape)
    #     
    #     print(data)
    #     
    #     print(average_value)
    #     
    #     print(std_value)
        
        data = (update_data - x_min[range!=0])/range[range!=0]
        
    #     data = data /std_value
        
        return data
    
    # def construct_full_X_Y(self, dataloader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device):
    #
        # full_features = []
        #
        # full_labels = []
        #
        # i = 0
        #
        #
        #
        # get_transfer_model_func = getattr(Transfer_learning, "compute_before_last_layer_" + transfer_model_name)
        #
        # get_last_layer_func = getattr(Transfer_learning, "get_last_layer_" + transfer_model_name)
        #
        #
        # tl = Transfer_learning()
        #
        #
        # last_layer = get_last_layer_func(tl, transfer_model)
        #
        #
        # for features, labels, ids in dataloader:
        #
            # print(i, ids.shape[0])
            #
            #
            # transfered_features,_ = self.compute_output_before_last_layer(features, tl, transfer_model, transfer_model_modules, get_transfer_model_func, last_layer, is_GPU, device)
            #
            # full_features.append(transfered_features)
            #
            # print(transfered_features.shape)
            #
            # full_labels.append(labels)
            #
            # i+=1
            #
            #
        # full_X = torch.cat(full_features, 0)
        #
        # full_Y = torch.cat(full_labels, 0)
        #
        # print(full_X.shape)
        #
        # full_X = self.normalize(full_X)
        #
        # return full_X, full_Y
        #
        
        
        
        
    
        
    
    # def prepare_cifar10_2(self, transfer_model, transfer_model_name, is_GPU, device):
    #
    #
# #         configs = load_config_data(config_file)
# #     
# # #     print(configs)
# #         git_ignore_folder = configs['git_ignore_folder']
# #         
# #         directory_name = configs['directory']
# #         
# #         X_train, y_train = load_svmlight_file(directory_name + "cifar10")
# #         
# #         X_test, y_test = load_svmlight_file(directory_name + "cifar10.t")
# #         
# #         
# #         train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)/255
# #         
# #         train_Y = torch.from_numpy(y_train).type(torch.LongTensor).view(y_train.shape[0], -1)
# #         
# #         test_X = torch.from_numpy(X_test.todense()).type(torch.DoubleTensor)/255
# #         
# #         test_Y = torch.from_numpy(y_test).type(torch.LongTensor).view(y_test.shape[0], -1)
        #
        #
# #         resnet = models.resnet50(pretrained=True)
# #         # freeze all model parameters
# #         for param in resnet.parameters():
# #             param.requires_grad = False
# #         transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# #         transform = transforms.Compose([ToPILImage(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #
        # transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #
        #
        # data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                       # download=True,
                       # transform=transform)
        # data_test = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                          # train=False,
                          # download=True,
                          # transform=transform)
                          #
                          #
                          #
        # if is_GPU:
            # transfer_model.to(device)
            #
        # train_dataset = MyDataset(data_train)
        #
        # test_dataset = MyDataset(data_test)
        #
        # data_train_loader = DataLoader(train_dataset, batch_size=100, num_workers=0)
        # data_test_loader = DataLoader(test_dataset, batch_size=100, num_workers=0)
        #
        #
        # transfer_model_modules = list(transfer_model.children())
        #
        # train_X, train_Y = self.construct_full_X_Y(data_train_loader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device)
        #
        # test_X, test_Y = self.construct_full_X_Y(data_test_loader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device)
        #
        #
        #
# #         train_X = data_train.transforms.transform(data_train.data)
# #         train_X = self.compose_train_test_data(data_train, transfer_model)
# #         
# #         train_Y = data_train.targets
# #         
# #         
# #         test_X = self.compose_train_test_data(data_test, transfer_model)
# #         
# #         test_Y = data_test.targets
#
# #         data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
# #                    download=True,
# #                    transform=transforms.Compose([
# #                        transforms.Resize((32, 32)),
# #                        transforms.RandomHorizontalFlip(),
# #                        transforms.ToTensor(),
# #                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# #                        ]))
# #         data_test = torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
# #                       train=False,
# #                       download=True,
# #                       transform=transforms.Compose([
# #                           transforms.ToTensor(),
# #                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# #                           ]))
#
        #
        # return train_X, train_Y, test_X, test_Y
        #
    
    
    def get_hyperparameters_cifar10_2(self, parameters, init_lr, regularization_rate):
    
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def get_num_class_cifar10_2(self):
        return 10
    
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
#         
# #         regularization_rate = 0.1
#         
#         return criterion, optimizer, lr_scheduler
        