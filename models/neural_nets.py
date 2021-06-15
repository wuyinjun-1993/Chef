'''
Created on Dec 11, 2020

'''
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from collections import OrderedDict

from torchvision.models import *

from collections import namedtuple


import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')

from torchvision import models as md

try:
    from utils.utils import *
    from models.util_func import *
    from models.early_stopping import *
except ImportError:
    from utils import *
    from util_func import *
    from early_stopping import *

class Identity(nn.Module):
    def forward(self, x): return x
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
        
    return m
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y 
    
class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)
    
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target
    
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
        'relu': nn.ReLU(True)
    }
    
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)


def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }



def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'classifier': {
            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)       
    return n


class LeNet5_cifar(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, feature_num, num_class = 10, binary = False):
        super(LeNet5_cifar, self).__init__()
        
        
        
#             model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5)).double()),
            ('relu1', nn.ReLU().double()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5)).double()),
            ('relu3', nn.ReLU().double()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5)).double()),
#             ().
            ('relu5', nn.ReLU().double()),
            ('avg', nn.AdaptiveAvgPool2d((1, 1)).double())
        ]))
        
        self.embed_dim = 84

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84).double()),
            ('relu6', nn.ReLU().double()),
            # ('batch norm', nn.BatchNorm1d(num_features=84).double()),
            ('f7', nn.Linear(84, num_class).double())
#             ('sig7', nn.LogSoftmax(dim=-1).double())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer

    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss

    def soft_loss_function_reduce(self, logit, targets, weight = None):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        if weight is None:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        else:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1).view(-1)* weight.view(-1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val

    def valid_models(self, val_dataset, bz, tag, is_GPU, device, loss_func):
    
        data_val_size = val_dataset.data.shape[0]
        
        avg_loss = 0
        
        pred_acc = 0
        
        y_true = []
        
        y_pred = []
        
        y_out = []
        
        with torch.no_grad():
            for k in range(0, data_val_size, bz):
                end_id = k + bz
                
                if end_id > data_val_size:
                    end_id = data_val_size
                
                
    #             curr_rand_ids = random_ids[k: end_id]
                
                X = val_dataset.data[k: end_id]
                
    #             X = X.view(X.shape[0], -1)
                
                Y = val_dataset.labels[k: end_id]
                
                if (len(Y.shape) == 2 and Y.shape[1] > 1):
                    Y = torch.argmax(Y, 1)
                
                X = X.type(torch.DoubleTensor)
                
#                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                     if args.binary:
#                         Y = Y[:,args.did].type(torch.LongTensor)
#                     else:
#                         Y = Y.type(torch.LongTensor)
#                 
#                 else:
#                     if args.binary:
#                         Y = Y[:,args.did].type(torch.DoubleTensor)
#                     else:
#                         Y = Y.type(torch.DoubleTensor)
                
                
                if is_GPU:
                    X = X.to(device)
                    Y = Y.to(device)
                
                output = self.forward(X)
                
                pred = self.determine_labels(X)
    #             if args.binary:
                    
                y_out.append(output.cpu().view(-1))
                
                pred_acc += torch.sum(pred.cpu() == Y.cpu())
        
                y_true.append(Y.cpu())
                
                y_pred.append(pred.cpu())
        
                if loss_func is None:
                    curr_loss_func = self.get_loss_function('mean')
                    curr_loss = curr_loss_func(self.forward(X), Y.type(torch.long))
                else:
                    curr_loss_func = loss_func
                
#                 curr_loss2 = self.get_loss_function()
#                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                    curr_loss = curr_loss(self.forward(X).view(Y.shape), Y.type(torch.long))
                    
        
#                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                     curr_loss = args.loss(output, Y)
#                 else:
#                     curr_loss = args.loss(output.view(-1), Y)
        
        
    #             curr_loss = args.loss(output.view(-1), Y)
    #             print(Y)
    #             print(curr_loss)
                avg_loss += curr_loss.cpu().detach()*(end_id - k)
                
        avg_loss = avg_loss/data_val_size
        
        pred_acc = pred_acc*1.0/data_val_size
        
        
        y_pred_array = torch.cat(y_pred).numpy()
        
        y_true_array = torch.cat(y_true).numpy()
        
#         y_out_array = torch.cat(y_out).numpy()
#     
#         fpr, tpr, thresholds = metrics.roc_curve(
#                     y_true_array, y_pred_array, pos_label=1)
#         
#         auc = metrics.auc(fpr, tpr)
        
    #     print('auc score::', roc_auc_score(y_true_array, y_pred_array), auc)
#         if not (isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss)):
#             print(tag + ' auc score::', roc_auc_score(y_true_array, y_out_array), auc)
        
        print(tag + ' dataset loss and accuracy::', avg_loss, pred_acc)

    def train_model(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None, val_dataset = None, test_dataset = None):
    
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y = X[i:end_id], Y[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
#                     batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('mean')
                        loss = curr_loss(self.forward(batch_x), batch_y.type(torch.long))
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y)
                    
            
                    loss.backward()
                    
                    avg_loss += loss.cpu().detach()*(end_id - i)
                    
                    optimizer.step()
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                
                
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(origin_X.shape[0]))
                
                del X
                
                del Y
                
                if val_dataset is not None:
                    self.valid_models(val_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                
                if end:
                    break
            print('training with weight loss::', avg_loss/(origin_X.shape[0]))

    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
        
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                curr_r_weights = r_weights[random_ids]
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                        batch_r_weight = batch_r_weight.to(device)
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('none')
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
    #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                    loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    
    #                     if torch.isnan(loss):
    #                         print('here')
                    
    #                     print(i, j, loss)
                    
    #                 if loss < 0:
    #                     print('here')
                    
    #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                    
    #                 print(torch.norm(gap)) 
                    
    #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
    
    #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
            
                    
            
                    loss.backward()
                    
    #                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
    #                         print('here')
                    
                    avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight.cpu().detach())
                    
                    optimizer.step()
    #                 with torch.no_grad():
        
        
                        
    #                     self.lr.theta -= learning_rate * self.lr.theta.grad
                        
    #                     gap = torch.norm(self.lr.theta.grad)
                        
                    
                     
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                    
                    
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(torch.sum(r_weights)))
                
                del X
                
                del Y
                
        #         del random_ids
                
                if end:
                    break
                
                
            print('training with weight loss::', avg_loss/(torch.sum(r_weights)))

class LeNet5_cifar2(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, feature_num, num_class = 10, binary = False):
        super(LeNet5_cifar2, self).__init__()
        
        
        
#             model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5)).double()),
            ('bn1',nn.BatchNorm2d(6).double()),
            ('relu1', nn.ReLU().double()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5)).double()),
            ('bn3',nn.BatchNorm2d(16).double()),
            ('relu3', nn.ReLU().double()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5)).double()),
            ('bn5',nn.BatchNorm2d(120).double()),
#             ().
            ('relu5', nn.ReLU().double()),
            ('avg', nn.AdaptiveAvgPool2d((1, 1)).double())
        ]))
        
        self.embed_dim = 84

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84).double()),
            ('relu6', nn.ReLU().double()),
            # ('batch norm', nn.BatchNorm1d(num_features=84).double()),
            ('f7', nn.Linear(84, num_class).double())
#             ('sig7', nn.LogSoftmax(dim=-1).double())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        # print(output.shape)
        output = self.fc(output)
        return output
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer

    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss

    def soft_loss_function_reduce(self, logit, targets, weight = None):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        if weight is None:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        else:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1).view(-1)* weight.view(-1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val

    def valid_models(self, val_dataset, bz, tag, is_GPU, device, loss_func):
    
        data_val_size = val_dataset.data.shape[0]
        
        avg_loss = 0
        
        pred_acc = 0
        
        y_true = []
        
        y_pred = []
        
        y_out = []
        
        with torch.no_grad():
            for k in range(0, data_val_size, bz):
                end_id = k + bz
                
                if end_id > data_val_size:
                    end_id = data_val_size
                
                
    #             curr_rand_ids = random_ids[k: end_id]
                
                X = val_dataset.data[k: end_id]
                
    #             X = X.view(X.shape[0], -1)
                
                Y = val_dataset.labels[k: end_id]
                
                if (len(Y.shape) == 2 and Y.shape[1] > 1):
                    Y = torch.argmax(Y, 1)
                
                X = X.type(torch.DoubleTensor)
                
#                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                     if args.binary:
#                         Y = Y[:,args.did].type(torch.LongTensor)
#                     else:
#                         Y = Y.type(torch.LongTensor)
#                 
#                 else:
#                     if args.binary:
#                         Y = Y[:,args.did].type(torch.DoubleTensor)
#                     else:
#                         Y = Y.type(torch.DoubleTensor)
                
                
                if is_GPU:
                    X = X.to(device)
                    Y = Y.to(device)
                
                output = self.forward(X)
                
                pred = self.determine_labels(X)
    #             if args.binary:
                    
                y_out.append(output.cpu().view(-1))
                
                pred_acc += torch.sum(pred.cpu() == Y.cpu())
        
                y_true.append(Y.cpu())
                
                y_pred.append(pred.cpu())
        
                if loss_func is None:
                    curr_loss_func = self.get_loss_function('mean')
                    curr_loss = curr_loss_func(self.forward(X), Y.type(torch.long))
                else:
                    curr_loss_func = loss_func
                
#                 curr_loss2 = self.get_loss_function()
#                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                    curr_loss = curr_loss(self.forward(X).view(Y.shape), Y.type(torch.long))
                    
        
#                 if isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss):
#                     curr_loss = args.loss(output, Y)
#                 else:
#                     curr_loss = args.loss(output.view(-1), Y)
        
        
    #             curr_loss = args.loss(output.view(-1), Y)
    #             print(Y)
    #             print(curr_loss)
                avg_loss += curr_loss.cpu().detach()*(end_id - k)
                
        avg_loss = avg_loss/data_val_size
        
        pred_acc = pred_acc*1.0/data_val_size
        
        
        y_pred_array = torch.cat(y_pred).numpy()
        
        y_true_array = torch.cat(y_true).numpy()
        
#         y_out_array = torch.cat(y_out).numpy()
#     
#         fpr, tpr, thresholds = metrics.roc_curve(
#                     y_true_array, y_pred_array, pos_label=1)
#         
#         auc = metrics.auc(fpr, tpr)
        
    #     print('auc score::', roc_auc_score(y_true_array, y_pred_array), auc)
#         if not (isinstance(args.loss, nn.NLLLoss) or isinstance(args.loss, nn.CrossEntropyLoss)):
#             print(tag + ' auc score::', roc_auc_score(y_true_array, y_out_array), auc)
        
        print(tag + ' dataset loss and accuracy::', avg_loss, pred_acc)

    def train_model(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None, val_dataset = None, test_dataset = None):
    
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y = X[i:end_id], Y[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
#                     batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('mean')
                        loss = curr_loss(self.forward(batch_x), batch_y.type(torch.long))
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y)
                    
            
                    loss.backward()
                    
                    avg_loss += loss.cpu().detach()*(end_id - i)
                    
                    optimizer.step()
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                
                
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(origin_X.shape[0]))
                
                del X
                
                del Y
                
                if val_dataset is not None:
                    self.valid_models(val_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                
                if end:
                    break
            print('training with weight loss::', avg_loss/(origin_X.shape[0]))

    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
        
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                curr_r_weights = r_weights[random_ids]
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                        batch_r_weight = batch_r_weight.to(device)
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('none')
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
    #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                    loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    
    #                     if torch.isnan(loss):
    #                         print('here')
                    
    #                     print(i, j, loss)
                    
    #                 if loss < 0:
    #                     print('here')
                    
    #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                    
    #                 print(torch.norm(gap)) 
                    
    #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
    
    #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
            
                    
            
                    loss.backward()
                    
    #                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
    #                         print('here')
                    
                    avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight.cpu().detach())
                    
                    optimizer.step()
    #                 with torch.no_grad():
        
        
                        
    #                     self.lr.theta -= learning_rate * self.lr.theta.grad
                        
    #                     gap = torch.norm(self.lr.theta.grad)
                        
                    
                     
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                    
                    
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(torch.sum(r_weights)))
                
                del X
                
                del Y
                
        #         del random_ids
                
                if end:
                    break
                
                
            print('training with weight loss::', avg_loss/(torch.sum(r_weights)))


class CNN_NLP(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, feature_num, num_class = 10, binary = False, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100],dropout=0.1):
        super(CNN_NLP, self).__init__()
        
        
        
#             model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
#     model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

        
        
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=feature_num[-1],
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i]).double()
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_class).double()
        self.dropout = nn.Dropout(p=dropout)

        # self.fc = nn.Sequential(OrderedDict([
            # ('f6', nn.Linear(120, 84).double()),
            # ('relu6', nn.ReLU().double()),
            # ('f7', nn.Linear(84, num_class).double())
# #             ('sig7', nn.LogSoftmax(dim=-1).double())
        # ]))

    def forward(self, img):
        # output = self.convnet(img)
        # output = output.view(img.size(0), -1)
        # output = self.fc(output)
        # return output
        x_reshaped = img.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))
        
        return logits
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer

    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss

    def soft_loss_function_reduce(self, logit, targets, weight = None):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        if weight is None:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        else:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1).view(-1)* weight.view(-1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val

class ResNet18(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, feature_num, num_class = 10):
        super(ResNet18, self).__init__()
#         print("INIT MODEL")
        self.resnet = md.resnet18(pretrained = True).double()
        
        num_ftrs = self.resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.resnet.fc = nn.Linear(num_ftrs, num_class).double()
        
        

    def forward(self, img):
        # output = self.convnet(img)
        # output = output.view(img.size(0), -1)
        # output = self.fc(output)
        # return output
        
        return self.resnet(img)
        
        # x_reshaped = img.permute(0, 2, 1)
        # x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        #
        # # Max pooling. Output shape: (b, num_filters[i], 1)
        # x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #     for x_conv in x_conv_list]
        #
        # # Concatenate x_pool_list to feed the fully connected layer.
        # # Output shape: (b, sum(num_filters))
        # x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
        #                  dim=1)
        #
        # # Compute logits. Output shape: (b, n_classes)
        # logits = self.fc(self.dropout(x_fc))
        #
        # return logits
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer

    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss

    def soft_loss_function_reduce(self, logit, targets, weight = None):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        if weight is None:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        else:
            loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1).view(-1)* weight.view(-1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val

class CBR_tiny(torch.nn.Module):          
    def __init__(self, num_class = 10, binary = False):
        super(CBR_tiny, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 16, kernel_size=(7, 7)).double()),
            ('relu1', nn.ReLU().double()),
            ('s2', nn.MaxPool2d(kernel_size=(7, 7), stride=2).double()),
            ('c3', nn.Conv2d(16, 32, kernel_size=(7, 7)).double()),
            ('relu3', nn.ReLU().double()),
            ('s4', nn.MaxPool2d(kernel_size=(7, 7), stride=2).double()),
            ('c5', nn.Conv2d(32, 128, kernel_size=(7, 7)).double()),
            ('relu5', nn.ReLU().double()),
            ('s5', nn.MaxPool2d(kernel_size=(7, 7), stride=2).double()),
#             ('c6', nn.Conv2d(128, 256, kernel_size=(7, 7)).double()),
#             ('relu6', nn.ReLU().double()),
#             ('s6', nn.MaxPool2d(kernel_size=(7, 7), stride=2).double()),
            ('avg', nn.AdaptiveAvgPool2d((1, 1)).double())
        ]))

#         nn.init.kaiming_uniform_(self.convnet.c1.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.convnet.c3.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.convnet.c5.weight, mode='fan_in', nonlinearity='relu')

        self.embed_dim = 128
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(128, num_class).double()),
#             ('relu6', nn.ReLU().double()),
            
#             ,('sig7', nn.LogSoftmax(dim=-1).double())
        ]))
#         self.f7 = nn.Linear(84, num_class).double()
#         nn.init.kaiming_uniform_(self.fc.f6.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.fc.f7.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
#         output = self.f7(output)
        return output

    def get_embedding_dim(self):
        return self.embed_dim

    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer

    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    
class LeNet5(torch.nn.Module):          
    def __init__(self, num_class = 10, binary = False):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(4, 4)).double()),
            ('relu1', nn.ReLU().double()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5)).double()),
            ('relu3', nn.ReLU().double()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2).double()),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4)).double()),
            ('relu5', nn.ReLU().double())
        ]))

#         nn.init.kaiming_uniform_(self.convnet.c1.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.convnet.c3.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.convnet.c5.weight, mode='fan_in', nonlinearity='relu')


        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84).double()),
            ('relu6', nn.ReLU().double()),
            
#             ,('sig7', nn.LogSoftmax(dim=-1).double())
        ]))
        
        self.embed_dim = 84
        
        self.f7 = nn.Linear(84, num_class).double()
#         nn.init.kaiming_uniform_(self.fc.f6.weight, mode='fan_in', nonlinearity='relu')
#         
#         nn.init.kaiming_uniform_(self.fc.f7.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        output = self.f7(output)
        return output
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def forward_before_last_layer(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
    
    def forward_from_before_last_layer(self, X):
        return F.softmax(self.f7(X), dim = 1)
    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    def train_model(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None):
    
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y = X[i:end_id], Y[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
#                     batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('mean')
                        loss = curr_loss(self.forward(batch_x), batch_y.type(torch.long))
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y)
                    
#                     if torch.isnan(loss):
#                         print('here')
                    
#                     print(i, j, loss)
                    
    #                 if loss < 0:
    #                     print('here')
                    
    #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                    
    #                 print(torch.norm(gap)) 
                    
    #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
    
    #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
            
                    
            
                    loss.backward()
                    
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                    
                    avg_loss += loss.cpu().detach()*(end_id - i)
                    
                    optimizer.step()
    #                 with torch.no_grad():
        
        
                        
    #                     self.lr.theta -= learning_rate * self.lr.theta.grad
                        
    #                     gap = torch.norm(self.lr.theta.grad)
                        
                    
                     
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                
                
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(origin_X.shape[0]))
                
                del X
                
                del Y
                
        #         del random_ids
                
                if end:
                    break
            print('training with weight loss::', avg_loss/(origin_X.shape[0]))


    
#     def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta):
# 
#         epoch = 0
#         
#         
#         mini_batch_epoch = 0
#         
#     #     if is_GPU:
#     #         lr.theta = lr.theta.to(device)
#         
#         
#         theta_list = []
#         
#         grad_list = []
#         
#         for j in range(epochs):
#             
# #             print('epoch::', j)
#             
#             end = False
#             
#             random_ids = random_ids_multi_super_iterations[j]
#         
#     #         learning_rate = lrs[j]
#             
#             X = origin_X[random_ids]
#             
#             Y = origin_Y[random_ids]
#             
#             
#             curr_r_weights = r_weights[random_ids]
#             
#             avg_loss = 0
#             
#             for i in range(0,X.shape[0], batch_size):
#                 
#                 end_id = i + batch_size
#                 
#                 if end_id >= X.shape[0]:
#                     end_id = X.shape[0]
#         
#                 batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
#                 
#                 batch_y = batch_y.type(torch.double)
#                 
#                 if is_GPU:
#                     batch_x = batch_x.to(device)
#                     
#                     batch_y = batch_y.to(device)
#                     
#                     batch_r_weight = batch_r_weight.to(device)
#                 
# #                 if self.lr.theta.grad is not None:
# #                     self.lr.theta.grad.zero_()
#         
#                 optimizer.zero_grad()
#                 
#                 curr_loss = self.get_loss_function('none')
#                 
# #                 curr_loss2 = self.get_loss_function()
# #                 print(torch.max())
#                 loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
#                 
# #                 if loss < 0:
# #                     print('here')
#                 
# #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
#                 
# #                 print(torch.norm(gap)) 
#                 
# #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
# 
# #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
#         
#                 
#         
#                 loss.backward()
#                 
#                 avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
#                 
#                 optimizer.step()
# #                 with torch.no_grad():
#     
#     
#                     
# #                     self.lr.theta -= learning_rate * self.lr.theta.grad
#                     
# #                     gap = torch.norm(self.lr.theta.grad)
#                     
#                 
#                  
#                 
#                 epoch = epoch + 1
#                 
#                 mini_batch_epoch += 1
#             
#             
# #             avg_loss = avg_loss
#             
#             print('loss::', j, avg_loss/(torch.sum(r_weights)))
#             
#             del X
#             
#             del Y
#             
#     #         del random_ids
#             
#             if end:
#                 break
#         print('training with weight loss::', avg_loss)
# #         return self.lr.theta
#     
    
    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
    
        epoch = 0
        
        
        mini_batch_epoch = 0
        
    #     if is_GPU:
    #         lr.theta = lr.theta.to(device)
        
        
        theta_list = []
        
        grad_list = []
        
        for j in range(epochs):
            
#             print('epoch::', j)
            
            end = False
            
            accuracy = 0
            
            if not random_ids_multi_super_iterations is None:
                random_ids = random_ids_multi_super_iterations[j]
            else:
                random_ids = torch.randperm(origin_X.shape[0])
        
    #         learning_rate = lrs[j]
            
            X = origin_X[random_ids]
            
            Y = origin_Y[random_ids]
            
            
            curr_r_weights = r_weights[random_ids]
            
            avg_loss = 0
            
            for i in range(0,X.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= X.shape[0]:
                    end_id = X.shape[0]
        
                batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                
                batch_y = batch_y.type(torch.double)
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    
                    batch_y = batch_y.to(device)
                    
                    batch_r_weight = batch_r_weight.to(device)
                
#                 if self.lr.theta.grad is not None:
#                     self.lr.theta.grad.zero_()
        
                optimizer.zero_grad()
                
                if loss_func is None:
                    curr_loss = self.get_loss_function('none')
                else:
                    curr_loss = loss_func
                
#                 curr_loss2 = self.get_loss_function()
#                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                
#                     if torch.isnan(loss):
#                         print('here')
                
#                     print(i, j, loss)
                
#                 if loss < 0:
#                     print('here')
                
#                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                
#                 print(torch.norm(gap)) 
                
#                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)

#                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
        
                
        
                loss.backward()
                
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                
                avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight.cpu().detach())
                
                optimizer.step()
#                 with torch.no_grad():
    
    
                    
#                     self.lr.theta -= learning_rate * self.lr.theta.grad
                    
#                     gap = torch.norm(self.lr.theta.grad)
                    
                
                 
                
                epoch = epoch + 1
                
                mini_batch_epoch += 1
                
                
#             avg_loss = avg_loss
            
            print('loss::', j, avg_loss/(torch.sum(r_weights)))
            
            del X
            
            del Y
            
    #         del random_ids
            
            if end:
                break
            
            
        print('training with weight loss::', avg_loss/(torch.sum(r_weights)))
    
class ResNet9(nn.Module):
    def __init__(self, num_class = 1, binary = False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.model = self.model.double()
#         if not binary:
        self.embed_dim = 512
        self.model.fc = nn.Linear(512, num_class).double()
#         else:
#             self.model.fc = nn.Linear(512, 1).double()
            
        self.binary = binary
        
    def forward(self, X):
        return self.model(X)
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
#         else:
#             return nn.BCEWithLogitsLoss(reduction=reduction)
    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
#         if not self.binary:
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
#         else:
#             if soft:
#                 labels = F.sigmoid(model_out)
#             else:
#                 labels = (model_out > 0).type(torch.LongTensor)
        
        return labels    
    
    
# class ResNet18(nn.Module):
#     def __init__(self, num_class = 1, binary = False):
#         super(ResNet18, self).__init__()
#         self.model = resnet18(pretrained = False)
#         self.model = self.model.double()
# #         if not binary:
#         self.model.fc = nn.Linear(512, num_class).double()
#
#         self.embed_dim = 512
# #         else:
# #             self.model.fc = nn.Linear(512, 1).double()
#
#         self.binary = binary
#
#     def forward(self, X):
#         return self.model(X)
#
#
#     def get_embedding_dim(self):
#         return self.embed_dim
#
#     def get_loss_function(self, reduction = 'mean', f1 = False):
#
# #         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
#
# #         optimizer = Adam(self.parameters(), lr=init_lr)
#         self.use_f1_loss = f1
#         if not f1:
#             return nn.CrossEntropyLoss(reduction = reduction)
#         else:
#             return f1_loss
# #         else:
# #             return nn.BCEWithLogitsLoss(reduction=reduction)
#
#     def soft_loss_function(self, logit, targets):
#         loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
#
#         return loss_val
#
#
#     def get_optimizer(self, init_lr, regularization_rate):
#
# #         optimizer = optim.SGD(self.parameters(), lr=init_lr,momentum=0.9, weight_decay=regularization_rate)
#
#         optimizer = optim.Adam(self.parameters(), lr=init_lr)
#
#         return optimizer
#
#
#     def determine_labels(self, X, soft=False):
#         model_out = self.forward(X)
# #         if not self.binary:
#         if soft:
#             labels = F.softmax(model_out, dim = 1)
#         else:
#             labels = torch.argmax(model_out, 1)
# #         else:
# #             if soft:
# #                 labels = F.sigmoid(model_out)
# #             else:
# #                 labels = (model_out > 0).type(torch.LongTensor)
#
#         return labels
#
#     def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
#
#         epoch = 0
#
#
#         mini_batch_epoch = 0
#
#     #     if is_GPU:
#     #         lr.theta = lr.theta.to(device)
#
#
#         theta_list = []
#
#         grad_list = []
#
#         for j in range(epochs):
#
# #             print('epoch::', j)
#
#             end = False
#
#             accuracy = 0
#
#             if not random_ids_multi_super_iterations is None:
#                 random_ids = random_ids_multi_super_iterations[j]
#             else:
#                 random_ids = torch.randperm(origin_X.shape[0])
#
#     #         learning_rate = lrs[j]
#
#             X = origin_X[random_ids]
#
#             Y = origin_Y[random_ids]
#
#
#             curr_r_weights = r_weights[random_ids]
#
#             avg_loss = 0
#
#             for i in range(0,X.shape[0], batch_size):
#
#                 print(j, i)
#
#                 end_id = i + batch_size
#
#                 if end_id >= X.shape[0]:
#                     end_id = X.shape[0]
#
#                 batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
#
#                 batch_y = batch_y.type(torch.double)
#
#                 if is_GPU:
#                     batch_x = batch_x.to(device)
#
#                     batch_y = batch_y.to(device)
#
#                     batch_r_weight = batch_r_weight.to(device)
#
# #                 if self.lr.theta.grad is not None:
# #                     self.lr.theta.grad.zero_()
#
#                 optimizer.zero_grad()
#
#                 if loss_func is None:
#                     curr_loss = self.get_loss_function('none')
#                 else:
#                     curr_loss = loss_func
#
# #                 curr_loss2 = self.get_loss_function()
# #                 print(torch.max())
# #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
#                 loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
#
# #                     if torch.isnan(loss):
# #                         print('here')
#
# #                     print(i, j, loss)
#
# #                 if loss < 0:
# #                     print('here')
#
# #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
#
# #                 print(torch.norm(gap)) 
#
# #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
#
# #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
#
#
#
#                 loss.backward()
#
# #                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
# #                         print('here')
#
#                 avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
#
#                 optimizer.step()
# #                 with torch.no_grad():
#
#
#
# #                     self.lr.theta -= learning_rate * self.lr.theta.grad
#
# #                     gap = torch.norm(self.lr.theta.grad)
#
#
#
#
#                 epoch = epoch + 1
#
#                 mini_batch_epoch += 1
#
#
# #             avg_loss = avg_loss
#
#             print('loss::', j, avg_loss/(torch.sum(r_weights)))
#
#             del X
#
#             del Y
#
#     #         del random_ids
#
#             if end:
#                 break
#         print('training with weight loss::', avg_loss/(torch.sum(r_weights)))
#
#     def train_model(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None):
#
#             epoch = 0
#
#
#             mini_batch_epoch = 0
#
#         #     if is_GPU:
#         #         lr.theta = lr.theta.to(device)
#
#
#             theta_list = []
#
#             grad_list = []
#
#             for j in range(epochs):
#
#     #             print('epoch::', j)
#
#                 end = False
#
#                 accuracy = 0
#
#                 if not random_ids_multi_super_iterations is None:
#                     random_ids = random_ids_multi_super_iterations[j]
#                 else:
#                     random_ids = torch.randperm(origin_X.shape[0])
#
#         #         learning_rate = lrs[j]
#
#                 X = origin_X[random_ids]
#
#                 Y = origin_Y[random_ids]
#
#
#                 avg_loss = 0
#
#                 for i in range(0,X.shape[0], batch_size):
#
#                     end_id = i + batch_size
#
#                     if end_id >= X.shape[0]:
#                         end_id = X.shape[0]
#
#                     batch_x, batch_y = X[i:end_id], Y[i:end_id]
#
#                     batch_y = batch_y.type(torch.double)
#
# #                     batch_x = batch_x.view(batch_x.shape[0],-1)
#
#                     if is_GPU:
#                         batch_x = batch_x.to(device)
#
#                         batch_y = batch_y.to(device)
#
#
#     #                 if self.lr.theta.grad is not None:
#     #                     self.lr.theta.grad.zero_()
#
#                     optimizer.zero_grad()
#
#                     if loss_func is None:
#                         curr_loss = self.get_loss_function('mean')
#                         loss = curr_loss(self.forward(batch_x), batch_y.type(torch.long))
#                     else:
#                         curr_loss = loss_func
#
#     #                 curr_loss2 = self.get_loss_function()
#     #                 print(torch.max())
# #                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
#                         loss = curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y)
#
# #                     if torch.isnan(loss):
# #                         print('here')
#
# #                     print(i, j, loss)
#
#     #                 if loss < 0:
#     #                     print('here')
#
#     #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
#
#     #                 print(torch.norm(gap)) 
#
#     #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
#
#     #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
#
#
#
#                     loss.backward()
#
# #                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
# #                         print('here')
#
#                     avg_loss += loss.cpu().detach()*(end_id - i)
#
#                     optimizer.step()
#     #                 with torch.no_grad():
#
#
#
#     #                     self.lr.theta -= learning_rate * self.lr.theta.grad
#
#     #                     gap = torch.norm(self.lr.theta.grad)
#
#
#
#
#                     epoch = epoch + 1
#
#                     mini_batch_epoch += 1
#
#
#     #             avg_loss = avg_loss
#
#                 print('loss::', j, avg_loss/(origin_X.shape[0]))
#
#                 del X
#
#                 del Y
#
#         #         del random_ids
#
#                 if end:
#                     break
#             print('training with weight loss::', avg_loss/(origin_X.shape[0]))


    
#     def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta):
# 
#         epoch = 0
#         
#         
#         mini_batch_epoch = 0
#         
#     #     if is_GPU:
#     #         lr.theta = lr.theta.to(device)
#         
#         
#         theta_list = []
#         
#         grad_list = []
#         
#         for j in range(epochs):
#             
# #             print('epoch::', j)
#             
#             end = False
#             
#             random_ids = random_ids_multi_super_iterations[j]
#         
#     #         learning_rate = lrs[j]
#             
#             X = origin_X[random_ids]
#             
#             Y = origin_Y[random_ids]
#             
#             
#             curr_r_weights = r_weights[random_ids]
#             
#             avg_loss = 0
#             
#             for i in range(0,X.shape[0], batch_size):
#                 
#                 end_id = i + batch_size
#                 
#                 if end_id >= X.shape[0]:
#                     end_id = X.shape[0]
#         
#                 batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
#                 
#                 batch_y = batch_y.type(torch.double)
#                 
#                 if is_GPU:
#                     batch_x = batch_x.to(device)
#                     
#                     batch_y = batch_y.to(device)
#                     
#                     batch_r_weight = batch_r_weight.to(device)
#                 
# #                 if self.lr.theta.grad is not None:
# #                     self.lr.theta.grad.zero_()
#         
#                 optimizer.zero_grad()
#                 
#                 curr_loss = self.get_loss_function('none')
#                 
# #                 curr_loss2 = self.get_loss_function()
# #                 print(torch.max())
#                 loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
#                 
# #                 if loss < 0:
# #                     print('here')
#                 
# #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
#                 
# #                 print(torch.norm(gap)) 
#                 
# #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
# 
# #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
#         
#                 
#         
#                 loss.backward()
#                 
#                 avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
#                 
#                 optimizer.step()
# #                 with torch.no_grad():
#     
#     
#                     
# #                     self.lr.theta -= learning_rate * self.lr.theta.grad
#                     
# #                     gap = torch.norm(self.lr.theta.grad)
#                     
#                 
#                  
#                 
#                 epoch = epoch + 1
#                 
#                 mini_batch_epoch += 1
#             
#             
# #             avg_loss = avg_loss
#             
#             print('loss::', j, avg_loss/(torch.sum(r_weights)))
#             
#             del X
#             
#             del Y
#             
#     #         del random_ids
#             
#             if end:
#                 break
#         print('training with weight loss::', avg_loss)
# #         return self.lr.theta
#     


class ResNet50(nn.Module):
    def __init__(self, num_class = 1):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained = True)
        self.model = self.model.double()
        self.model.fc = nn.Linear(2048, num_class).double()
        self.embed_dim = 2048
        
    def forward(self, X):
        return self.model(X)
    
    
    
    def forward_before_last_layer(self, x):
        
#         input_x = x.clone()
#         self.model.conv1.weight
        x = self.model.conv1.double()(x)
#         print(torch.norm(x - output_val1[-3]))
        x1 = self.model.bn1.double()(x)
#         print(torch.norm(x1 - output_val1[-2]))
        x = self.model.relu(x1)
#         print(torch.norm(x - output_val1[-1]))
#         x = self.resnet.maxpool(x)

        x = self.model.layer1.double()(x)
#         print(torch.norm(x - output_val2[-1]))
        x = self.model.layer2.double()(x)
        x = self.model.layer3.double()(x)
        x = self.model.layer4.double()(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
#         self.check_output(x, input_x)
        
        return x
    
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
    
        epoch = 0
        
        
        mini_batch_epoch = 0
        
    #     if is_GPU:
    #         lr.theta = lr.theta.to(device)
        
        
        theta_list = []
        
        grad_list = []
        
        for j in range(epochs):
            
#             print('epoch::', j)
            
            end = False
            
            accuracy = 0
            
            if not random_ids_multi_super_iterations is None:
                random_ids = random_ids_multi_super_iterations[j]
            else:
                random_ids = torch.randperm(origin_X.shape[0])
        
    #         learning_rate = lrs[j]
            
            X = origin_X[random_ids]
            
            Y = origin_Y[random_ids]
            
            
            curr_r_weights = r_weights[random_ids]
            
            avg_loss = 0
            
            for i in range(0,X.shape[0], batch_size):
                
                print(j, i)
                
                end_id = i + batch_size
                
                if end_id >= X.shape[0]:
                    end_id = X.shape[0]
        
                batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                
                batch_y = batch_y.type(torch.double)
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    
                    batch_y = batch_y.to(device)
                    
                    batch_r_weight = batch_r_weight.to(device)
                
#                 if self.lr.theta.grad is not None:
#                     self.lr.theta.grad.zero_()
        
                optimizer.zero_grad()
                
                if loss_func is None:
                    curr_loss = self.get_loss_function('none')
                else:
                    curr_loss = loss_func
                
#                 curr_loss2 = self.get_loss_function()
#                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                
#                     if torch.isnan(loss):
#                         print('here')
                
#                     print(i, j, loss)
                
#                 if loss < 0:
#                     print('here')
                
#                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                
#                 print(torch.norm(gap)) 
                
#                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)

#                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
        
                
        
                loss.backward()
                
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                
                avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
                
                optimizer.step()
#                 with torch.no_grad():
    
    
                    
#                     self.lr.theta -= learning_rate * self.lr.theta.grad
                    
#                     gap = torch.norm(self.lr.theta.grad)
                    
                
                 
                
                epoch = epoch + 1
                
                mini_batch_epoch += 1
            
            
#             avg_loss = avg_loss
            
            print('loss::', j, avg_loss/(torch.sum(r_weights)))
            
            del X
            
            del Y
            
    #         del random_ids
            
            if end:
                break
        print('training with weight loss::', avg_loss/(torch.sum(r_weights)))
        
    def train_model(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, loss_func = None):
    
            epoch = 0
            
            
            mini_batch_epoch = 0
            
        #     if is_GPU:
        #         lr.theta = lr.theta.to(device)
            
            
            theta_list = []
            
            grad_list = []
            
            for j in range(epochs):
                
    #             print('epoch::', j)
                
                end = False
                
                accuracy = 0
                
                if not random_ids_multi_super_iterations is None:
                    random_ids = random_ids_multi_super_iterations[j]
                else:
                    random_ids = torch.randperm(origin_X.shape[0])
            
        #         learning_rate = lrs[j]
                
                X = origin_X[random_ids]
                
                Y = origin_Y[random_ids]
                
                
                avg_loss = 0
                
                for i in range(0,X.shape[0], batch_size):
                    
                    end_id = i + batch_size
                    
                    if end_id >= X.shape[0]:
                        end_id = X.shape[0]
            
                    batch_x, batch_y = X[i:end_id], Y[i:end_id]
                    
                    batch_y = batch_y.type(torch.double)
                    
#                     batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('mean')
                        loss = curr_loss(self.forward(batch_x), batch_y.type(torch.long))
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y)
                    
#                     if torch.isnan(loss):
#                         print('here')
                    
#                     print(i, j, loss)
                    
    #                 if loss < 0:
    #                     print('here')
                    
    #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                    
    #                 print(torch.norm(gap)) 
                    
    #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
    
    #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
            
                    
            
                    loss.backward()
                    
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                    
                    avg_loss += loss.cpu().detach()*(end_id - i)
                    
                    optimizer.step()
    #                 with torch.no_grad():
        
        
                        
    #                     self.lr.theta -= learning_rate * self.lr.theta.grad
                        
    #                     gap = torch.norm(self.lr.theta.grad)
                        
                    
                     
                    
                    epoch = epoch + 1
                    
                    mini_batch_epoch += 1
                
                
    #             avg_loss = avg_loss
                
                print('loss::', j, avg_loss/(origin_X.shape[0]))
                
                del X
                
                del Y
                
        #         del random_ids
                
                if end:
                    break
            print('training with weight loss::', avg_loss/(origin_X.shape[0]))


    
#     def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta):
# 
#         epoch = 0
#         
#         
#         mini_batch_epoch = 0
#         
#     #     if is_GPU:
#     #         lr.theta = lr.theta.to(device)
#         
#         
#         theta_list = []
#         
#         grad_list = []
#         
#         for j in range(epochs):
#             
# #             print('epoch::', j)
#             
#             end = False
#             
#             random_ids = random_ids_multi_super_iterations[j]
#         
#     #         learning_rate = lrs[j]
#             
#             X = origin_X[random_ids]
#             
#             Y = origin_Y[random_ids]
#             
#             
#             curr_r_weights = r_weights[random_ids]
#             
#             avg_loss = 0
#             
#             for i in range(0,X.shape[0], batch_size):
#                 
#                 end_id = i + batch_size
#                 
#                 if end_id >= X.shape[0]:
#                     end_id = X.shape[0]
#         
#                 batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
#                 
#                 batch_y = batch_y.type(torch.double)
#                 
#                 if is_GPU:
#                     batch_x = batch_x.to(device)
#                     
#                     batch_y = batch_y.to(device)
#                     
#                     batch_r_weight = batch_r_weight.to(device)
#                 
# #                 if self.lr.theta.grad is not None:
# #                     self.lr.theta.grad.zero_()
#         
#                 optimizer.zero_grad()
#                 
#                 curr_loss = self.get_loss_function('none')
#                 
# #                 curr_loss2 = self.get_loss_function()
# #                 print(torch.max())
#                 loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
#                 
# #                 if loss < 0:
# #                     print('here')
#                 
# #                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
#                 
# #                 print(torch.norm(gap)) 
#                 
# #                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)
# 
# #                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
#         
#                 
#         
#                 loss.backward()
#                 
#                 avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
#                 
#                 optimizer.step()
# #                 with torch.no_grad():
#     
#     
#                     
# #                     self.lr.theta -= learning_rate * self.lr.theta.grad
#                     
# #                     gap = torch.norm(self.lr.theta.grad)
#                     
#                 
#                  
#                 
#                 epoch = epoch + 1
#                 
#                 mini_batch_epoch += 1
#             
#             
# #             avg_loss = avg_loss
#             
#             print('loss::', j, avg_loss/(torch.sum(r_weights)))
#             
#             del X
#             
#             del Y
#             
#     #         del random_ids
#             
#             if end:
#                 break
#         print('training with weight loss::', avg_loss)
# #         return self.lr.theta
#     
          
        
class ConvNet(nn.Module):
    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1)).double()
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1)).double()
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1)).double()
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1)).double()
        self.pool = nn.MaxPool2d(2,2).double()
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512).double()
        self.fc2 = nn.Linear(in_features=512, out_features=64).double()
        self.Dropout = nn.Dropout(0.25).double()
        self.embed_dim = 64
        self.fc3 = nn.Linear(in_features=64, out_features=num_class).double()

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
    
    
    def get_embedding_dim(self):
        return self.embed_dim
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    def soft_loss_function(self, logit, targets):
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
        return loss_val
    
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    
    def determine_labels(self, X, soft=False):
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    
    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False):
    
        epoch = 0
        
        
        mini_batch_epoch = 0
        
    #     if is_GPU:
    #         lr.theta = lr.theta.to(device)
        
        
        theta_list = []
        
        grad_list = []
        
        for j in range(epochs):
            
#             print('epoch::', j)
            
            end = False
            
            accuracy = 0
            
            if not random_ids_multi_super_iterations is None:
                random_ids = random_ids_multi_super_iterations[j]
            else:
                random_ids = torch.randperm(origin_X.shape[0])
        
    #         learning_rate = lrs[j]
            
            X = origin_X[random_ids]
            
            Y = origin_Y[random_ids]
            
            
            curr_r_weights = r_weights[random_ids]
            
            avg_loss = 0
            
            for i in range(0,X.shape[0], batch_size):
                
                end_id = i + batch_size
                
                if end_id >= X.shape[0]:
                    end_id = X.shape[0]
        
                batch_x, batch_y, batch_r_weight = X[i:end_id], Y[i:end_id], curr_r_weights[i:end_id]
                
                batch_y = batch_y.type(torch.double)
                
                if is_GPU:
                    batch_x = batch_x.to(device)
                    
                    batch_y = batch_y.to(device)
                    
                    batch_r_weight = batch_r_weight.to(device)
                
#                 if self.lr.theta.grad is not None:
#                     self.lr.theta.grad.zero_()
        
                optimizer.zero_grad()
                
                if loss_func is None:
                    curr_loss = self.get_loss_function('none')
                else:
                    curr_loss = loss_func
                
#                 curr_loss2 = self.get_loss_function()
#                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                loss = torch.sum(curr_loss(self.forward(batch_x).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                
#                     if torch.isnan(loss):
#                         print('here')
                
#                     print(i, j, loss)
                
#                 if loss < 0:
#                     print('here')
                
#                 gap = torch.mean(curr_loss(self.forward(batch_x), batch_y)) - curr_loss2(self.forward(batch_x), batch_y)
                
#                 print(torch.norm(gap)) 
                
#                 loss2 = self.loss_function1(batch_x*batch_r_weight.view(-1,1), batch_y, theta, torch.sum(batch_r_weight), 0)

#                 loss2 = self.loss_function1(batch_x, batch_y, self.lr.theta, batch_x.shape[0], regularization_coeff)
        
                
        
                loss.backward()
                
#                     if (self.fc1.weight.grad != self.fc1.weight.grad).any() or (self.fc1.bias.grad != self.fc1.bias.grad).any():
#                         print('here')
                
                avg_loss += loss.cpu().detach()*torch.sum(batch_r_weight)
                
                optimizer.step()
#                 with torch.no_grad():
    
    
                    
#                     self.lr.theta -= learning_rate * self.lr.theta.grad
                    
#                     gap = torch.norm(self.lr.theta.grad)
                    
                
                 
                
                epoch = epoch + 1
                
                mini_batch_epoch += 1
            
            
#             avg_loss = avg_loss
            
            print('loss::', j, avg_loss/(torch.sum(r_weights)))
            
            del X
            
            del Y
            
    #         del random_ids
            
            if end:
                break
        print('training with weight loss::', avg_loss/(torch.sum(r_weights)))
    