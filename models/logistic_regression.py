'''
Created on Nov 18, 2020

'''

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/utils')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models')

try:
    from utils.utils import *
    from models.util_func import *
    from models.early_stopping import *
except ImportError:
    from utils import *
    from util_func import *
    from early_stopping import *

import time


def get_selected_data(r_weight, X, Y, removed_count):
    
        _,sorted_ids = torch.sort(r_weight, descending=True)
        
        removed_ids = sorted_ids.view(-1)[X.shape[0]-removed_count:]
        
        return X[removed_ids], Y[removed_ids]
        
#         remaining_ids = sorted_ids.view(-1)[0:X.shape[0]-removed_count] 
#         
#         integer_r_weight = torch.zeros_like(r_weight)
#         
#         integer_r_weight[remaining_ids] = 1
        
        
        


def get_derivative_norm(x, norm):
    
    x = x.view(-1)
    
    if norm == '2':
        return 2*x
    else:
        if norm == 'inf':
            
            abs_X = torch.abs(x)
            
            sgns = torch.sign(x)
            
            max_x = torch.max(abs_X)
            
            res = torch.zeros_like(x)
            
            res_ids = torch.nonzero((abs_X - max_x)==0)
            
            res[res_ids] = 1
            
            res = res*sgns
            
            return res



class DNN(nn.Module):

    def __init__(self, input_dim, output_dim=1, bias = True, hidden = 200):
        super(DNN, self).__init__()
        self.input_dim = input_dim
#         self.bnorm = nn.BatchNorm1d(input_dim).double()
        self.fc1 = nn.Linear(input_dim, hidden, bias = bias).double()
        
        self.fc2 = nn.Linear(hidden, output_dim, bias = bias).double()
        
#         self.fc2 = nn.LogSoftmax().double()


    def forward(self, x, layer_idx=None):
#         x = self.bnorm(x)
        if layer_idx is None:
            x = x.view(x.shape[0], -1)
            out1 = self.fc2(F.relu(self.fc1(x)))
            
            return out1
        
        if layer_idx == 0:
            return x#.view(x.shape[0], -1)
        else:
            if layer_idx >= 1:
                x = x.view(x.shape[0], -1)
                out1 = self.fc1(x)
                return out1
#                 return 
#         else:
#             curr_features = nn.ModuleList(features).eval()
#             
#             for i, model in enumerate(self._features):
#                 x = model(x)
#     
#                 if i == layer_idx:
#                     return x
        
#         out1 = self.fc2(out1)

#         return out1
    
    def forward_embedding(self, x):

        x = x.view(x.shape[0], -1)
        out1 = self.fc1(x)
        
        return out1, x
    
    def get_embedding_dim(self):
        return self.input_dim
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            # return nn.CrossEntropyLoss(reduction = reduction)
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    def forward_before_last_layer(self, X):
        return X
    
    def forward_from_before_last_layer(self, X):
        return F.softmax(self.fc1(X), dim = 1)
        
    
    '''targets:: soft'''
    '''-torch.mean(torch.sum(torch.nn.functional.log_softmax(y, dim=1) * onehot(t, 2), dim=1))'''
    def soft_loss_function(self, logit, targets):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val
    
    
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
    
    # def soft_loss_function_reduce(self, logit, targets):
    #
# #         hard_labels = torch.argmax(targets, dim = 1)
# #         
# #         time1 = time.time()
# #         
# #         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
# #         
# #         time2 = time.time()
        #
        #
        # loss_val = -torch.mean(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        #
# #         time3 = time.time()
# #         
# #         print('t1::', time2 - time1)
# #         
# #         print('t2::', time3 - time2)
        #
        # return loss_val
    
    
    def determine_labels(self, X, soft=False):
        
        X = X.view(X.shape[0], -1)
        
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    
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
        
        return avg_loss
        
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
                    
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                    
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
                
                if val_dataset is not None:
                    self.valid_models(val_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'valid', is_GPU, device, loss_func)
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

    
    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False, valid_dataset = None, test_dataset = None):
    
            epoch = 0
            
            early_stopping = models.EarlyStopping(patience=7, verbose=True)

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
                    
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                        batch_r_weight = batch_r_weight.to(device)
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('none')
                        loss = torch.sum(curr_loss(self.forward(batch_x), batch_y.type(torch.long)).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = torch.sum(curr_loss(self.forward(batch_x.view(batch_x.shape[0], -1)).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    
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
                
#                 if valid_dataset is not None:
                valid_loss = self.valid_models(valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                
                early_stopping(valid_loss, self)
                
                
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                    
                if early_stopping.early_stop:
                    set_model_parameters(self, early_stopping.model_param)
                    print("Early stopping")
                    self.valid_models(valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                    if test_dataset is not None:
                        self.valid_models(test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                    break
        #         del random_ids
                
            print('training with weight loss::', avg_loss/(torch.sum(r_weights)))


class Logistic_regression(nn.Module):

    def __init__(self, input_dim, output_dim=1, bias = True):
        super(Logistic_regression, self).__init__()
        self.input_dim = input_dim
#         self.bnorm = nn.BatchNorm1d(input_dim).double()
        self.fc1 = nn.Linear(input_dim, output_dim, bias = bias).double()
        
#         self.fc2 = nn.LogSoftmax().double()


    def forward(self, x, layer_idx=None):
#         x = self.bnorm(x)
        if layer_idx is None:
            x = x.view(x.shape[0], -1)
            out1 = self.fc1(x)
            return out1
        
        if layer_idx == 0:
            return x#.view(x.shape[0], -1)
        else:
            if layer_idx >= 1:
                x = x.view(x.shape[0], -1)
                out1 = self.fc1(x)
                return out1
#                 return 
#         else:
#             curr_features = nn.ModuleList(features).eval()
#             
#             for i, model in enumerate(self._features):
#                 x = model(x)
#     
#                 if i == layer_idx:
#                     return x
        
#         out1 = self.fc2(out1)

#         return out1
    
    def forward_embedding(self, x):

        x = x.view(x.shape[0], -1)
        out1 = self.fc1(x)
        
        return out1, x
    
    def get_embedding_dim(self):
        return self.input_dim
    
    def get_loss_function(self, reduction = 'mean', f1 = False):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay = regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        self.use_f1_loss = f1
        if not f1:
            # return nn.CrossEntropyLoss(reduction = reduction)
            return nn.CrossEntropyLoss(reduction = reduction)
        else:
            return f1_loss
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    def forward_before_last_layer(self, X):
        return X
    
    def forward_from_before_last_layer(self, X):
        return F.softmax(self.fc1(X), dim = 1)
        
    
    '''targets:: soft'''
    '''-torch.mean(torch.sum(torch.nn.functional.log_softmax(y, dim=1) * onehot(t, 2), dim=1))'''
    def soft_loss_function(self, logit, targets):
        
#         hard_labels = torch.argmax(targets, dim = 1)
#         
#         time1 = time.time()
#         
#         loss_val2 = F.cross_entropy(logit, hard_labels, reduce = 'none')
#         
#         time2 = time.time()
        
        
        loss_val = -(torch.sum(F.log_softmax(logit, dim=1) * targets, dim=1))
        
#         time3 = time.time()
#         
#         print('t1::', time2 - time1)
#         
#         print('t2::', time3 - time2)
        
        return loss_val
    
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
    
    
    def determine_labels(self, X, soft=False):
        
        X = X.view(X.shape[0], -1)
        
        model_out = self.forward(X)
        if soft:
            labels = F.softmax(model_out, dim = 1)
        else:
            labels = torch.argmax(model_out, 1)
        
        return labels
    
    
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
        
        return avg_loss
        
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
                    
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                    
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
                
                if val_dataset is not None:
                    self.valid_models(val_dataset, batch_size, 'valid', is_GPU, device, loss_func)
        #         del random_ids
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'valid', is_GPU, device, loss_func)
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

    
    def train_model_with_weight(self, optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count, theta, loss_func = None, integer_label = False, valid_dataset = None, test_dataset = None):
    
            epoch = 0
            
            early_stopping = models.EarlyStopping(patience=7, verbose=True)

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
                    
                    batch_x = batch_x.view(batch_x.shape[0],-1)
                    
                    if is_GPU:
                        batch_x = batch_x.to(device)
                        
                        batch_y = batch_y.to(device)
                        
                        batch_r_weight = batch_r_weight.to(device)
                    
    #                 if self.lr.theta.grad is not None:
    #                     self.lr.theta.grad.zero_()
            
                    optimizer.zero_grad()
                    
                    if loss_func is None:
                        curr_loss = self.get_loss_function('none')
                        loss = torch.sum(curr_loss(self.forward(batch_x), batch_y.type(torch.long)).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    else:
                        curr_loss = loss_func
                    
    #                 curr_loss2 = self.get_loss_function()
    #                 print(torch.max())
#                     print(i, j, curr_loss, batch_x, batch_y, batch_r_weight)
                        loss = torch.sum(curr_loss(self.forward(batch_x.view(batch_x.shape[0], -1)).view(batch_y.shape), batch_y).view(-1)*batch_r_weight.view(-1))/torch.sum(batch_r_weight)
                    
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
                
#                 if valid_dataset is not None:
                valid_loss = self.valid_models(valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                
                early_stopping(valid_loss, self)
                
                
                if test_dataset is not None:
                    self.valid_models(test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                    
                if early_stopping.early_stop:
                    set_model_parameters(self, early_stopping.model_param)
                    print("Early stopping")
                    self.valid_models(valid_dataset, batch_size, 'valid', is_GPU, device, loss_func=None)
                    if test_dataset is not None:
                        self.valid_models(test_dataset, batch_size, 'test', is_GPU, device, loss_func=None)
                    break
        #         del random_ids
                
            print('training with weight loss::', avg_loss/(torch.sum(r_weights)))


    
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

class Binary_Logistic_regression(nn.Module):

    def __init__(self, input_dim, bias=True):
        super(Binary_Logistic_regression, self).__init__()
        
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, 1, bias=bias).double()
        
        self.fc2 = nn.Sigmoid()


    def forward(self, x):

        out1 = self.fc1(x)
        
        out = self.fc2(out1)

        return out

    def forward_embedding(self, x):

        out1 = self.fc1(x)
        
        out = self.fc2(out1)

        return out, out1

    
    def get_embedding_dim(self):
        return self.input_dim
    
    def get_loss_function(self, reduction = 'mean'):
        
#         optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
#         optimizer = Adam(self.parameters(), lr=init_lr)
        
        return nn.BCELoss(reduction=reduction)
    
    
    def get_optimizer(self, init_lr, regularization_rate):
        
        optimizer = optim.SGD(self.parameters(), lr=init_lr, weight_decay=regularization_rate)
        
        return optimizer
    
    
    def determine_labels(self, X, soft = False):
        model_out = self.forward(X)
        
        if soft:
            labels = model_out
        else:
            labels = (model_out > 0.5).type(torch.LongTensor).view(-1)
        
        return labels
        
    
    '''targets:: soft'''
    def soft_loss_function(self, logit, targets):
        
#         origin_logit = logit.clone()
#         
#         logit[logit < 1e-7] = 1e-7
#         
#         logit[logit > 1- 1e-7] = 1-1e-7
#         
#         print(torch.norm(origin_logit - logit))
        
        loss_val = -(torch.log(logit + 1e-7).view(-1) * targets.view(-1) + torch.log(1 - logit.view(-1) + 1e-7) * (1 - targets.view(-1)))
        
        return loss_val
    
    def soft_loss_function_reduce(self, logit, targets):
        
#         origin_logit = logit.clone()
#         
#         logit[logit < 1e-7] = 1e-7
#         
#         logit[logit > 1- 1e-7] = 1-1e-7
#         
#         print(torch.norm(origin_logit - logit))
        
        loss_val = -torch.mean((torch.log(logit + 1e-7).view(-1) * targets.view(-1) + torch.log(1 - logit.view(-1) + 1e-7) * (1 - targets.view(-1))), dim = 0)
        
        return loss_val
    
    def loss_function_hessian_with_weight(self, X, Y, sigmoid_res, beta, r_weight, removed_count, model):
    
        '''n*1, r_weight:n*1'''
#         X_Y_theta_prod = Y.view(-1)*torch.mm(X, theta).view(-1)
        
#         X_Y_theta_prod = Y.view(-1)*model(X).view(-1)
#         
# #         eps_vec = ((X>0).type(torch.float)*2 - 1)*epsilon
# #         
# #         perturbed_X = X - eps_vec
#         
#         hessian1 = torch.mm(torch.t(r_weight.view(-1,1)*X), X*(F.sigmoid(X_Y_theta_prod)*(1-F.sigmoid(X_Y_theta_prod))).view(-1,1))/(X.shape[0] - removed_count)
#         
#         print(hessian1.shape)

#         print(sigmoid_res, r_weight)



        hessian1 = torch.mm(torch.t(X), X*sigmoid_res.view(-1,1)*(1-sigmoid_res.view(-1,1))*(r_weight.cpu().view(-1,1)))

#         sigmoid_res2 = model(X)
# 
#         model.fc1.zero_grad()
# 
#         first_order_grad = torch.sum(-(Y.view(-1,1) - sigmoid_res2.view(-1,1))*X*r_weight.view(-1,1), dim=0)
#         
#         first_order_grad[0].backward()
#         
#         second_order_grad = model.fc1.weight.grad.clone()
#         
#         model.fc1.zero_grad()
# 
#         print('diff::', torch.norm(second_order_grad - hessian1[0]))

        res = beta*torch.eye(X.shape[1], dtype = torch.double, device = X.device) + hessian1/(X.shape[0] - removed_count)
        
        
        
        
        return res, hessian1
    
#     def compute_partial_gradient_x_delete_perturb(self, X, update_X, Y, theta, origin_theta, regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec, lower_bound = None, upper_bound = None, sample_id=0):
# #      = X + delta_X
#     
#         hessian_res, _ = self.loss_function_hessian_with_weight(update_X, Y, theta, regularization_coeff, r_weight, removed_count)
#         
#         X_Y_theta_prod = Y.view(-1,1)*torch.mm(update_X, theta)
#         
#         hessian_inv = torch.inverse(hessian_res)
# 
#         '''n*1'''
#         sigmoid_res = F.sigmoid(X_Y_theta_prod)
#         
#         
#         term1 = regularization_coeff*theta.view(1,-1) - Y.view(-1,1)*update_X*(1-sigmoid_res) 
#         
#         '''n*m'''
#         term1_X = update_X*(sigmoid_res*(sigmoid_res - 1))
#         
#         '''n*1'''
#         
#         term2_X = (Y.view(-1,1)*(1-sigmoid_res))
#         
# #         res_term1 = term1[0:X.shape[0]-1] - term1[X.shape[0]-1:] 
#         
# #         '''n*m'''
# #         term1 = X*(sigmoid_res*(sigmoid_res - 1))
# #         
# #         '''n*1'''
# #         
# #         term2 = (Y.view(-1,1)*(1-sigmoid_res))
#         
#         full_grad = torch.zeros_like(r_weight)
#         
#         
#         full_grad_X = torch.zeros_like(X)
# #         full_grad2 = torch.zeros_like(X)
#         
#     #     term2 = 
#         
#         
#         total_count = X.shape[0]
#         
# #             total_count = global_count
# #         
# #         
# #         for k in range(total_count):
# #             
# #             print('compute gradient::', k)
# #             
# #             curr_term1 = torch.mm(sub_hessian_inv.view(1,-1), torch.mm(term1[k].view(-1,1), theta.view(1,-1)))/X.shape[0]
# #              
# #     #         curr_term1_extra = torch.mm(sub_hessian_inv, torch.abs(torch.mm(term1_extra[i].view(-1,1), theta.view(1,-1))))
# #              
# #     #         print(curr_term1.shape)
# #              
# #             curr_term2 = sub_hessian_inv.view(1,-1)*term2[k]/X.shape[0]
# #     
# #     #         print(curr_term1.shape, curr_term2.shape)
# #             curr_grad = curr_term1 + curr_term2
# #             
# #             full_grad2[k] = curr_grad
#         
#     
# #         gap = 1000
#     
#         
# #             theta_copy = theta.view(1, 1, -1).repeat(gap, 1, 1)
#         
#         '''gap, 1, m'''
# #             if is_l2_norm:
# #             sub_hessian_inv_copy = sub_hessian_inv.view(1,1,-1).repeat(gap, 1, 1)
# #             else:
#         sub_hessian_inv_copy = hessian_inv.view(1,hessian_inv.shape[0],hessian_inv.shape[1]).repeat(gap, 1, 1)
#         
#         if (not norm == 'pos') and (not norm == 'neg'):
#             theta_grad = get_derivative_norm(theta.view(-1) - origin_theta.view(-1), norm)
#             
#     #             if norm == 'l2':
#             theta_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
#         else:
#             theta.requires_grad = True
#             
#             origin_theta_copy = origin_theta.detach().clone() 
#             
#             origin_theta_copy.requires_grad = False
#             
#             if norm == 'pos':
#                 loss = torch.dot(origin_theta_copy.view(-1), theta.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta.view(-1)))
#             else:
#                 loss = -torch.dot(origin_theta_copy.view(-1), theta.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta.view(-1)))
#             
#             print('theta difference::', loss)
# #             loss = torch.sum(torch.argmax(torch.mm(update_X, theta), 1) == torch.argmax(torch.mm(X, origin_theta_copy), 1))
#             
# #             loss = torch.sum(F.gumbel_softmax(torch.mm(update_X, theta), hard = True) == F.gumbel_softmax(torch.mm(X, origin_theta_copy), hard = True))
#             
# #             loss = -torch.sum(torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy)))
#             
# #             loss = -torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy))
#             
#             
#             if theta.grad is not None:
#                 theta.grad.zero_()
#             
#             loss.backward()
#             
#             theta_grad = theta.grad.clone()
#             
#             theta_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
#             
#             theta.grad.zero_()
#             
#             theta.requires_grad = False
# #             else:
# #                 if norm == ''
#         
#         
#         sub_hessian_inv_theta_prod = -torch.bmm(theta_copy, sub_hessian_inv_copy)
#         with torch.no_grad():
#             for k in range(0, total_count, gap):
#                 
#                 end_id = k+gap
#                 if end_id >= total_count:
#                     end_id = total_count
#     
#                 batch_term1 = term1[k:end_id]
#                 
#                 batch_term1_X = term1_X[k:end_id]
#                 
#                 batch_term2_X = term2_X[k:end_id]
#                 
#                 print('compute gradient::', k, end_id)
#                 
#                 '''gap, 1'''
#                 
# #                 batch_term2 = term2[k: end_id]
#                 '''gap*m*1'''
# #                 print(sub_hessian_inv_copy[0: end_id - k].shape, batch_term1.shape, end_id)
#                 curr_res_term1 = -torch.bmm(sub_hessian_inv_theta_prod[0: end_id - k], batch_term1.view(end_id - k, -1,1))/(X.shape[0]-removed_count)
#                 
# #                 curr_term2 = sub_hessian_inv_copy[0: end_id - k]*batch_term2.view(end_id - k, 1, 1)/X.shape[0]
#                 
# #                 curr_grad = (curr_term1 + curr_term2).view(end_id - k, -1)
#                 '''- from the max problem'''
# #                 curr_res_term1 = -torch.bmm(theta_copy[0: end_id - k], curr_res_term1)
#                 
# #                 print(res_term1.shape, full_grad[k:end_id].shape)
#                 
#                 full_grad[k:end_id] = curr_res_term1.view(-1) + mu_vec[k:end_id].view(-1) - lambda_vec[k:end_id].view(-1) - gamma_vec
#         
#                 curr_term1_X = torch.bmm(sub_hessian_inv_theta_prod[0: end_id - k], torch.bmm(batch_term1_X.view(end_id - k, -1,1), theta_copy[0: end_id - k]))/X.shape[0]
#                 
#                 curr_term2_X = sub_hessian_inv_theta_prod[0: end_id - k]*batch_term2_X.view(end_id - k, 1, 1)/X.shape[0]
#                 
#                 curr_grad = (curr_term1_X + curr_term2_X).view(end_id - k, -1)
#                 
#                 full_grad_X[k:end_id] = curr_grad
#     #         print(torch.norm(full_grad2 - full_grad))
#         
#         update_r_weight = r_weight - learning_rate*full_grad
#         
#         if lower_bound is None:
#             
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
#             
#             update_X = X+update_delta_X
#         else:
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_X[update_X > upper_bound] = upper_bound[update_X > upper_bound]
#             
#             update_X[update_X < lower_bound] = lower_bound[update_X < lower_bound]
#             
# #             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
#             
# #             update_X = X+update_delta_X
# #         update_r_weight[-1] = X.shape[0] - removed_count - torch.sum(update_r_weight[0:-1])
# #         update_delta_r_weight = torch.clamp(r_weight - update_r_weight, 0, 1)
# #         
# #         update_r_weight = r_weight-update_delta_r_weight
# 
#         
#         
#         
#         _,sorted_ids = torch.sort(update_r_weight, descending=True)
#         
#         removed_ids = sorted_ids.view(-1)[X.shape[0]-removed_count:]
#         
#         remaining_ids = sorted_ids.view(-1)[0:X.shape[0]-removed_count] 
#         
#         integer_r_weight = torch.zeros_like(r_weight)
#         
#         integer_r_weight[remaining_ids] = 1
#         
# #         removed_ids = torch.nonzero(update_r_weight < ((X.shape[0] - removed_count)/X.shape[0])) 
#         
#         
#         
#         
#         return integer_r_weight, removed_ids, update_r_weight, mu_vec, lambda_vec, gamma_vec, update_X
#     
    
    def is_small_model(self):
        return self.fc1.weight.shape[1] < 1000
#         return False
    
    def compute_partial_gradient_x_delete(self, X, Y, model, origin_model, theta, origin_theta, regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec, is_GPU, device, optimizer, lower_bound = None, upper_bound = None, sample_id=0, hessian_lr = 1000, hessian_eps = 1e-8, hv_random_sampling = False, hv_bz = 1):
#      = X + delta_X

        frozen_model_para(origin_model)
    
        total_count = X.shape[0]
    
        update_X = X.clone()
    
        sigmoid_res = model(update_X)
        
        full_loss = self.get_loss_function()(sigmoid_res.view(Y.shape), Y.type(torch.DoubleTensor))
        
        model.fc1.zero_grad()
        
        full_loss.backward()
    
        expected_loss = model.fc1.weight.grad.clone()
    
        model.fc1.zero_grad()
        
        if model.fc1.bias is not None:
            update_X = torch.cat([update_X, torch.ones([update_X.shape[0], 1], dtype = update_X.dtype, device = update_X.device)], dim = 1)
    
#         '''n*1'''
#         sigmoid_res = F.sigmoid(X_Y_theta_prod)
        
        
#         term1 = regularization_coeff*theta.view(1,-1) - Y.view(-1,1)*update_X*(1-sigmoid_res) 
        '''cpu'''
        term1 = regularization_coeff*theta.view(1,-1) - (Y.view(-1,1) - sigmoid_res.view(-1,1))*update_X
            
            
        full_grad = torch.zeros_like(r_weight)
    #         diff = expected_loss.view(1,-1) + regularization_coeff*theta.view(1,-1) - torch.sum(term1, dim=0)/X.shape[0]
    #         
    #         print('diff::', torch.norm(diff))
        if self.is_small_model():
            '''cpu'''
            hessian_res, _ = self.loss_function_hessian_with_weight(update_X, Y, sigmoid_res, regularization_coeff, r_weight, removed_count, model)
            
    #         X_Y_theta_prod = Y.view(-1,1)*torch.mm(update_X, theta)
            
    #         X_Y_theta_prod = Y.view(-1,1)*model(update_X)
    #
            '''cpu'''
            hessian_inv = torch.inverse(hessian_res)
            
            
    #         
    #         print(diff)
            
            
    #         '''n*m'''
    #         term1_X = update_X*(sigmoid_res*(sigmoid_res - 1))
    #         
    #         '''n*1'''
    #         
    #         term2_X = (Y.view(-1,1)*(1-sigmoid_res))
            
    #         res_term1 = term1[0:X.shape[0]-1] - term1[X.shape[0]-1:] 
            
    #         '''n*m'''
    #         term1 = X*(sigmoid_res*(sigmoid_res - 1))
    #         
    #         '''n*1'''
    #         
    #         term2 = (Y.view(-1,1)*(1-sigmoid_res))
            
            
    #         full_grad_X = torch.zeros_like(X)
    #         full_grad2 = torch.zeros_like(X)
            
        #     term2 = 
            
            
    #             total_count = global_count
    #         
    #         
    #         for k in range(total_count):
    #             
    #             print('compute gradient::', k)
    #             
    #             curr_term1 = torch.mm(sub_hessian_inv.view(1,-1), torch.mm(term1[k].view(-1,1), theta.view(1,-1)))/X.shape[0]
    #              
    #     #         curr_term1_extra = torch.mm(sub_hessian_inv, torch.abs(torch.mm(term1_extra[i].view(-1,1), theta.view(1,-1))))
    #              
    #     #         print(curr_term1.shape)
    #              
    #             curr_term2 = sub_hessian_inv.view(1,-1)*term2[k]/X.shape[0]
    #     
    #     #         print(curr_term1.shape, curr_term2.shape)
    #             curr_grad = curr_term1 + curr_term2
    #             
    #             full_grad2[k] = curr_grad
            
        
    #         gap = 1000
        
            
    #             theta_copy = theta.view(1, 1, -1).repeat(gap, 1, 1)
            
            '''gap, 1, m'''
    #             if is_l2_norm:
    #             sub_hessian_inv_copy = sub_hessian_inv.view(1,1,-1).repeat(gap, 1, 1)
    #             else:
            '''cpu'''
            sub_hessian_inv_copy = hessian_inv.view(1,hessian_inv.shape[0],hessian_inv.shape[1]).repeat(gap, 1, 1)
        
        if (not norm == 'pos') and (not norm == 'neg') and (not norm == 'loss'):
            theta_grad = get_derivative_norm(theta.view(-1) - origin_theta.view(-1), norm)
            
    #             if norm == 'l2':
            theta_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
        else:
            
            theta_copy = theta.detach().clone()
            
            theta_copy.requires_grad = True
            
            origin_theta_copy = origin_theta.detach().clone() 
            
            origin_theta_copy.requires_grad = False
            
            if norm == 'pos':
                loss = torch.dot(origin_theta_copy.view(-1), theta_copy.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta_copy.view(-1)))
            else:
                if norm == 'neg':
                    loss = -torch.dot(origin_theta_copy.view(-1), theta_copy.view(-1))/(torch.norm(origin_theta_copy.view(-1))*torch.norm(theta_copy.view(-1)))
                else:
                    loss_func = self.get_loss_function(reduction='none')

                    if model.fc1.weight.grad is not None:
                        model.fc1.weight.grad.zero_()
                    
                    if model.fc1.bias is not None and model.fc1.bias.grad is not None:
                        model.fc1.bias.grad.zero_()

#                     origin_model.fc1.requires_grad = False
                    
                    individual_loss_terms = loss_func(model(X).view(Y.shape), Y.type(torch.DoubleTensor)).clone()
                    
                    origin_individual_loss_terms = loss_func(origin_model(X).view(Y.shape), Y.type(torch.DoubleTensor)).clone()
                    
                    r_weight_cpu = r_weight.cpu()
                    
                    print('model_parameter::', get_vectorized_params(model), get_vectorized_params(origin_model))
                    
                    if norm == 'loss':
                        
                        loss1 = torch.sum(individual_loss_terms.view(-1)*r_weight_cpu.view(-1))/(X.shape[0] - removed_count)
                        
                        print('origin model parameters:', origin_model.fc1.weight, origin_model.fc1.bias)
                        
                        loss2 = torch.mean(loss_func(origin_model(X).view(Y.shape), Y.type(torch.DoubleTensor))*torch.ones_like(r_weight_cpu))
                    else:
                        loss1 = torch.sum(individual_loss_terms.view(-1)*(1-r_weight_cpu).view(-1))/torch.sum(1-r_weight_cpu)
                         
                        loss2 = torch.sum(origin_individual_loss_terms.view(-1)*(1-r_weight_cpu).view(-1))/torch.sum(1-r_weight_cpu)
                    
#                     selected_X, selected_Y = get_selected_data(r_weight, X, Y, removed_count)
#                     
#                     loss2 = torch.mean(loss_func(model(selected_X), selected_Y.type(torch.DoubleTensor)).view(-1))
#                     
#                     loss1 = torch.mean(loss_func(origin_model(selected_X), selected_Y.type(torch.DoubleTensor)))
                    
                    
                    print('loss1::', loss1)
                    
                    print('loss2::', loss2)
                    
                    loss = loss1# - loss2
                    
                    
                    
                    
#                     loss = 
            
            print('loss::', loss)
#             loss = torch.sum(torch.argmax(torch.mm(update_X, theta), 1) == torch.argmax(torch.mm(X, origin_theta_copy), 1))
            
#             loss = torch.sum(F.gumbel_softmax(torch.mm(update_X, theta), hard = True) == F.gumbel_softmax(torch.mm(X, origin_theta_copy), hard = True))
            
#             loss = -torch.sum(torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy)))
            
#             loss = -torch.abs(torch.mm(update_X[sample_id].view(1,-1), theta) - torch.mm(X[sample_id].view(1,-1), origin_theta_copy))
            
            if not norm == 'loss':
                if theta.grad is not None:
                    theta.grad.zero_()
                
                loss.backward()
                
                theta_grad = theta_copy.grad.clone()
                
                theta_grad_copy = -theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
                
                
                del theta_copy
            else:
#                 model.fc1.weight.grad.zero_()
                
                loss.backward()
                
                theta_grad = self.get_vectorized_grads(model.fc1.weight, model.fc1.bias)
                
                theta_grad_copy = theta_grad.view(1, 1, -1).repeat(gap, 1, 1)
                
#             theta_copy.grad.zero_()
#             
#             theta_copy.requires_grad = False
#             else:
#                 if norm == ''
        
        '''cpu'''
                
#         if self.is_small_model():
        sub_hessian_inv_theta_prod2 = torch.bmm(theta_grad_copy, sub_hessian_inv_copy)
    
#         else:
        hessian_inv_theta_prod2 = compute_conjugate_grad(model, theta_grad, X, Y, loss_func, optimizer, r_weight, removed_count, is_GPU, device, learning_rate = hessian_lr, eps = hessian_eps, random_sampling = hv_random_sampling, bz = hv_bz)
        
        sub_hessian_inv_theta_prod = hessian_inv_theta_prod2.view(1, 1, -1).repeat(gap, 1,1)
        
        print(torch.norm(sub_hessian_inv_theta_prod[0] - sub_hessian_inv_theta_prod2[0])) 
        
        
        
        with torch.no_grad():
            for k in range(0, total_count, gap):
                
                end_id = k+gap
                if end_id >= total_count:
                    end_id = total_count
    
                batch_term1 = term1[k:end_id]
                
                if is_GPU:
                    batch_term1 = batch_term1.to(device)
                
#                 batch_term1_X = term1_X[k:end_id]
#                 
#                 batch_term2_X = term2_X[k:end_id]
                
                print('compute gradient::', k, end_id)
                
                '''gap, 1'''
                
#                 batch_term2 = term2[k: end_id]
                '''gap*m*1'''
#                 print(sub_hessian_inv_copy[0: end_id - k].shape, batch_term1.shape, end_id)
                '''cpu'''
                curr_hessian_inv_theta_prod = sub_hessian_inv_theta_prod[0: end_id - k]
                
                if is_GPU:
                    curr_hessian_inv_theta_prod = curr_hessian_inv_theta_prod.to(device)
                    
#                 print(curr_hessian_inv_theta_prod.device, batch_term1.device)
                curr_res_term1 = -torch.bmm(curr_hessian_inv_theta_prod, batch_term1.view(end_id - k, -1,1))/(X.shape[0]-removed_count)
                
#                 curr_term2 = sub_hessian_inv_copy[0: end_id - k]*batch_term2.view(end_id - k, 1, 1)/X.shape[0]
                
#                 curr_grad = (curr_term1 + curr_term2).view(end_id - k, -1)
                '''- from the max problem'''
#                 curr_res_term1 = -torch.bmm(theta_copy[0: end_id - k], curr_res_term1)
                
#                 print(res_term1.shape, full_grad[k:end_id].shape)
#                 print(full_grad.device, curr_res_term1.device, mu_vec.device, lambda_vec.device, gamma_vec.device)
                full_grad[k:end_id] = curr_res_term1.view(-1) + (mu_vec[k:end_id].view(-1) - lambda_vec[k:end_id].view(-1) - gamma_vec)/(X.shape[0] - removed_count)
        
#                 curr_term1_X = torch.bmm(sub_hessian_inv_theta_prod[0: end_id - k], torch.bmm(batch_term1_X.view(end_id - k, -1,1), theta_copy[0: end_id - k]))/X.shape[0]
#                 
#                 curr_term2_X = sub_hessian_inv_theta_prod[0: end_id - k]*batch_term2_X.view(end_id - k, 1, 1)/X.shape[0]
#                 
#                 curr_grad = (curr_term1_X + curr_term2_X).view(end_id - k, -1)
                del batch_term1, curr_hessian_inv_theta_prod
#                 full_grad_X[k:end_id] = curr_grad
    #         print(torch.norm(full_grad2 - full_grad))
        if not (norm == 'loss' or norm == 'rloss'):
            update_r_weight = r_weight - learning_rate*full_grad
        else:
            
            if norm == 'rloss':
                update_r_weight = r_weight - learning_rate*(full_grad - (individual_loss_terms + origin_individual_loss_terms).detach().view(full_grad.shape))
            
            else:
                if is_GPU:
                    individual_loss_terms = individual_loss_terms.to(device)
                
                update_r_weight = r_weight - learning_rate*(full_grad + ((individual_loss_terms)/((X.shape[0] - removed_count))).detach().view(full_grad.shape))
        
                del individual_loss_terms
        
#         if lower_bound is None:
#             
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
#             
#             update_X = X+update_delta_X
#         else:
#             update_X = update_X - learning_rate*full_grad_X
#             
#             update_X[update_X > upper_bound] = upper_bound[update_X > upper_bound]
#             
#             update_X[update_X < lower_bound] = lower_bound[update_X < lower_bound]
            
#             update_delta_X = torch.clamp(update_X - X, -epsilon, epsilon)
            
#             update_X = X+update_delta_X
#         update_r_weight[-1] = X.shape[0] - removed_count - torch.sum(update_r_weight[0:-1])
#         update_delta_r_weight = torch.clamp(r_weight - update_r_weight, 0, 1)
#         
#         update_r_weight = r_weight-update_delta_r_weight

        
        
        
        sorted_weight,sorted_ids = torch.sort(update_r_weight, descending=True)
        
        removed_ids = sorted_ids.view(-1)[X.shape[0]-removed_count:]
        
        remaining_ids = sorted_ids.view(-1)[0:X.shape[0]-removed_count] 
        
        integer_r_weight = torch.zeros_like(r_weight)
        
        integer_r_weight[remaining_ids] = 1
        
#         removed_ids = torch.nonzero(update_r_weight < ((X.shape[0] - removed_count)/X.shape[0])) 
        
        
        
        
        return integer_r_weight, removed_ids, update_r_weight, mu_vec, lambda_vec, gamma_vec, update_X, sorted_weight



    def loss_function1(self, X, Y, theta, dim, beta):
    
        res = torch.sum(-F.logsigmoid(Y.view(-1)*torch.mm(X, theta).view(-1)))/dim
        
        return res + beta/2*torch.sum(theta*theta)


#     def evaluate_test_set(self, X_test, Y_test, batch_size, is_GPU, device):
#         
#         for i in range(0,X_test.shape[0], batch_size):
#                     
#             end_id = i + batch_size
#             
#             if end_id >= X_test.shape[0]:
#                 end_id = X_test.shape[0]
#     
#             batch_x, batch_y = X_test[i:end_id], Y_test[i:end_id]
#             
#             batch_y = batch_y.type(torch.double)
#         
#             if is_GPU:
#                 batch_x = batch_x.to(device)
#                 
#                 batch_y = batch_y.to(device)
#         
#             self.forward(batch_x)


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
    def set_parameters(self, weight, bias):
        self.fc1.weight.data.copy_(weight.detach().clone())
        
        self.fc1.weight.requires_grad = True
        
        if bias is not None:
            self.fc1.bias.data.copy_(bias.detach().clone())
            
            self.fc1.bias.requires_grad = True
    
    
    def get_vectorized_paras(self, weight, bias):
        if bias is not None:
            theta = torch.cat([weight.view(-1), bias.view(-1)])
        else:
            theta = weight.view(-1)
            
        return theta

    def get_vectorized_grads(self, weight, bias):
        if bias is not None:
            theta = torch.cat([weight.grad.view(-1), bias.grad.view(-1)])
        else:
            theta = weight.grad.view(-1)
            
        return theta
    
    
    def compute_model_para_diff(self, norm,origin_weight, origin_bias):
        
        if self.fc1.bias is not None:
            theta = torch.cat([self.fc1.weight.view(-1), self.fc1.bias.view(-1)])
            
            origin_theta = torch.cat([origin_weight.view(-1), origin_bias.view(-1)], dim = 0)
        
        else:
            theta = self.fc1.weight.view(-1)
            
            origin_theta = origin_weight.view(-1)
        
        if norm == '2':
            
            delta_w = torch.norm(theta - origin_theta)
            print('theta difference::', torch.norm(theta - origin_theta))
        else:
            if norm == 'inf':
                delta_w = torch.norm(theta - origin_theta, float('inf'))
                
                print('theta difference::', torch.norm(theta - origin_theta, float('inf')))
            else:
                if norm == 'pos':
                
                    delta_w = torch.dot(theta, origin_theta)/(torch.norm(theta) * torch.norm(origin_theta))
                else:
                    delta_w = -torch.dot(theta, origin_theta)/(torch.norm(theta) * torch.norm(origin_theta))
    
                print('theta difference::', delta_w)
    
    def optimize_two_steps_delete_perturb(self, dataset_train, origin_model, regularization_coeff, out_epoch_count, inner_epoch_count, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, learning_rate, batch_size, removed_count, training_learning_rate, wd, is_GPU, device, gap, norm, lower_bound = None, upper_bound = None, epsilon=0, loss_func = None, hessian_lr = 1000, hessian_eps = 1e-8, hv_random_sampling = False, hv_bz = 1):
#     delta_X = torch.zeros_like(X)
    
        optimizer = self.get_optimizer(training_learning_rate, wd)
    
        if random_ids_multi_super_iterations is None:
            random_ids_multi_super_iterations = create_random_id_multi_super_iters(dataset_train.data.shape[0], inner_epoch_count)
    
        origin_weight = origin_model.fc1.weight.detach()
        
        origin_bias = None
        
        if origin_model.fc1.bias is not None:
            origin_bias = origin_model.fc1.bias.detach()
        print('origin model parameter::', origin_weight, origin_bias)
    
        X = dataset_train.data
        
        Y = dataset_train.labels
    
        decay_threshold = 0.01
    
#         if is_GPU:
#             X = X.to(device)
#              
#             Y = Y.to(device)
            
#             origin_weight = origin_weight.to(device)
#             
#             origin_bias = 
        
        
        removed_ids = torch.randperm(X.shape[0])[0:X.shape[0] - removed_count]
        
        '''GPU'''
        r_weight = torch.rand(X.shape[0], device= origin_weight.device, dtype = X.dtype)
        
#         r_weight[:] = 0.5#(X.shape[0] - removed_count)/X.shape[0]
#         
#         
#         r_weight[:]=1
#         r_weight[1000:1060]=0
#         self.set_parameters(origin_weight, origin_bias)
#         self.train_model_with_weight(optimizer, random_ids_multi_super_iterations, X.type(torch.DoubleTensor), Y.type(torch.DoubleTensor), batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, r_weight, removed_count, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias))
        
        
        print('r_weight count::', torch.sum(r_weight))
#         update_X = X + delta_X
        self.set_parameters(origin_weight, origin_bias)
        
        prev_r_weight = r_weight.clone()
        
        '''GPU'''
        mu_vec = torch.rand(X.shape[0],device = origin_weight.device, dtype = X.dtype)
        
        '''GPU'''
        lambda_vec = torch.rand(X.shape[0],device = origin_weight.device, dtype = X.dtype)
        
        '''GPU'''
        gamma_vec = torch.rand(1,device = origin_weight.device, dtype = X.dtype)
        
        d_epoch_count = 1
        
        learning_rate2 = 0.0001
        
#         learning_rate3 = 0.0005
        
        warmup_epochs = 100
        
        upper_bound_tensor = None
        
        lower_bound_tensor = None
        
        delta_X = torch.rand((X.shape[0], X.shape[1]), device = X.device, dtype = X.dtype)
        
        if lower_bound is None:
            delta_X = (delta_X*2 - 1)*epsilon
        else:
            
            upper_bound_tensor = dataset_train.upper_bound
            
            lower_bound_tensor = dataset_train.lower_bound
            
            if is_GPU:
                upper_bound_tensor = upper_bound_tensor.to(device)
            
                lower_bound_tensor = lower_bound_tensor.to(device)
                
            upper_eps = (upper_bound_tensor - X)*delta_X

            lower_eps = (X - lower_bound_tensor)*delta_X
            
            delta_X[delta_X > 0] = upper_eps[delta_X > 0]
            
            delta_X[delta_X <= 0] = lower_eps[delta_X <= 0] 
        
        
        delta_w = 0
        
#         update_X = X + delta_X
        
        X = X.type(torch.DoubleTensor)
        
#         update_X = update_X.type(torch.DoubleTensor)
        
        for o_epoch in range(out_epoch_count):
            
            print('outer loop epoch::', o_epoch)
            for d_epoch in range(d_epoch_count):
                
            
                print('dual update epoch::', d_epoch)
        #         lr.theta.requires_grad = False
#                 with torch.no_grad():
                    
                '''X, update_X, Y, theta, regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec'''
                
                if o_epoch >= 100:
                    print('here')
                
                
                if is_GPU:
                    self = self.to('cpu')
                    origin_model = origin_model.to('cpu')
                
                updated_r_weight, removed_ids, origin_update_r_weight, mu_vec, lambda_vec, gamma_vec, new_update_X, sorted_weight = self.compute_partial_gradient_x_delete(X, Y, self, origin_model, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias), self.get_vectorized_paras(origin_weight, origin_bias), regularization_coeff, learning_rate, gap, r_weight, removed_count, epsilon, norm, mu_vec, lambda_vec, gamma_vec, is_GPU, device, optimizer, lower_bound_tensor, upper_bound_tensor, hessian_lr = hessian_lr, hessian_eps = hessian_eps, hv_random_sampling = hv_random_sampling, hv_bz = hv_bz)
            
                if is_GPU:
                    self = self.to(device)
                    origin_model = origin_model.to(device)
                
                print('removed_ids count::', removed_ids.view(-1)[0:100])
                
                print('removed_ids weight::', sorted_weight.view(-1)[0:100])
                
#                 print('X difference', torch.max(new_update_X - update_X), torch.min(new_update_X - update_X), torch.max(new_update_X - X), torch.min(new_update_X - X))
                
                print(torch.max(origin_update_r_weight), torch.min(origin_update_r_weight))
                
                print('r weight update::', torch.norm(origin_update_r_weight.type(torch.double) - r_weight), torch.norm(updated_r_weight.type(torch.double) - prev_r_weight.type(torch.double)))

#                 update_X = new_update_X
    #             print('X difference', torch.max(new_update_X - update_X), torch.min(new_update_X - update_X), torch.max(new_update_X - X), torch.min(new_update_X - X))
        
        #         
            
#                 self.set_parameters(origin_weight, origin_bias)
                
#                 self.lr.theta.requires_grad = True
    #             update_theta = self.model_update_standard_lib_logistic_regression(inner_epoch_count, dataset_train, random_ids_multi_super_iterations, sorted_ids_multi_super_iterations, removed_ids.view(-1), batch_size, training_learning_rate, is_GPU, device, regularization_coeff)
#                 '''random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, learning_rate, optimizer, criterion, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count'''
#                 self.train_model_with_weight(random_ids_multi_super_iterations, update_X, Y, batch_size, training_learning_rate, None, None, inner_epoch_count, is_GPU, device, regularization_coeff, origin_update_r_weight, removed_count)
            
                '''optimizer, random_ids_multi_super_iterations, origin_X, origin_Y, batch_size, epochs, is_GPU, device, regularization_coeff, r_weights, removed_count'''
                self.set_parameters(origin_weight, origin_bias)
 
                update_optimizer(optimizer, training_learning_rate)
                 
                origin_update_r_weight_new = origin_update_r_weight.clamp(min=0, max=1)
#                 origin_update_r_weight_new = origin_update_r_weight
                 
                self.train_model_with_weight(optimizer, random_ids_multi_super_iterations, X, Y, batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, origin_update_r_weight_new, removed_count, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias), loss_func = loss_func)
             
                print('model parameter::', self.fc1.weight, self.fc1.bias)
             
                self.compute_model_para_diff(norm,origin_weight, origin_bias)
#                 update_theta = update_theta/torch.norm(update_theta)
    #             update_X = new_update_X
    
                prev_r_weight = updated_r_weight
    
                r_weight = origin_update_r_weight_new
                
#                 self.set_parameters(update_theta.detach().clone())
#                 self.compute_model_para_diff(norm,origin_weight, origin_bias)

            
            mu_vec = mu_vec + learning_rate*(origin_update_r_weight.view(-1) - 1)/(X.shape[0] - removed_count)
        
            lambda_vec = lambda_vec + learning_rate*(-origin_update_r_weight.view(-1))/(X.shape[0] - removed_count)
            
            gamma_vec = gamma_vec + learning_rate*(X.shape[0] - removed_count - torch.sum(origin_update_r_weight))/(X.shape[0] - removed_count)
            
            mu_vec[mu_vec < 0] = 0
            
            lambda_vec[lambda_vec < 0] = 0
            
            print('learning rate::', learning_rate)
            
            print('mu vec::', torch.max(mu_vec), torch.min(mu_vec))
                
            print('lambda vec::', torch.max(lambda_vec), torch.min(lambda_vec))
            
            print('gamma::', gamma_vec)
            
            print('r_sum_diff::', torch.sum(origin_update_r_weight) - (X.shape[0] - removed_count))
            
            print('r_weight::', torch.max(origin_update_r_weight), torch.min(origin_update_r_weight))
            
            if torch.abs(X.shape[0] - removed_count - torch.sum(origin_update_r_weight)).item() < stop_number and torch.abs(torch.max(origin_update_r_weight) - 1) <= 0.001 and torch.abs(torch.min(origin_update_r_weight)) <= 0.001:
                break
            if torch.abs(X.shape[0] - removed_count - torch.sum(origin_update_r_weight)).item() < stop_number and torch.abs(torch.max(origin_update_r_weight) - 1) <= decay_threshold and torch.abs(torch.min(origin_update_r_weight)) <= decay_threshold:
                learning_rate = learning_rate/2
                decay_threshold = decay_threshold/2
            
            
#             if torch.abs(X.shape[0] - removed_count - torch.sum(r_weight)).item() < 5:
#                 learning_rate = learning_rate/3*2
#                 if learning_rate < 0.2:
#                     learning_rate = 0.2
            
#             self.set_parameters(origin_weight, origin_bias)
#                 
# #             self.lr.theta.requires_grad = True
#             
#             
#             self.train_model_with_weight(optimizer, random_ids_multi_super_iterations, update_X, Y, batch_size,  inner_epoch_count, is_GPU, device,regularization_coeff, updated_r_weight, removed_count, self.get_vectorized_paras(self.fc1.weight, self.fc1.bias))
# 
#             print('model parameter::', self.fc1.weight, self.fc1.bias)
#             
# #             final_update_theta = self.train_model_with_weight(random_ids_multi_super_iterations, update_X, Y, batch_size, training_learning_rate, None, None, inner_epoch_count, is_GPU, device, regularization_coeff, updated_r_weight, removed_count)
#             
# #             final_update_theta = final_update_theta/torch.norm(final_update_theta)
#             
#             self.compute_model_para_diff(norm, origin_weight, origin_bias)
            
            
#             if norm == '2':
#                 print('theta difference::', torch.norm(final_update_theta - origin_weight))
#             else:
#                 if norm == 'inf':
#                     print('theta difference::', torch.norm(final_update_theta - origin_weight, float('inf')))
#             
#             
#                 else:
#                     if norm == 'pos':
#                     
#                         curr_delta_w = torch.dot(final_update_theta.view(-1), origin_weight.view(-1))/(torch.norm(final_update_theta.view(-1)) * torch.norm(origin_theta.view(-1)))
#                     else:
#                         curr_delta_w = -torch.dot(final_update_theta.view(-1), origin_weight.view(-1))/(torch.norm(final_update_theta.view(-1)) * torch.norm(origin_theta.view(-1)))
#         
#                     print('theta difference::', curr_delta_w)
            
            print('here')
            
        return delta_w, removed_ids

    
    
    
    
    