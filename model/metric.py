import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1) #.cpu().numpy()
        #target = target.cpu().numpy()
        assert pred.shape[0] == len(target)
        correct = 0
        new_pred = pred[target != -100]
        new_target = target[target != -100]
        correct += torch.sum(new_pred == new_target).item()
        #print('pred:', torch.unique(new_pred, return_counts=True))
        #print('target:', torch.unique(new_target, return_counts=True))
    if len(new_target) > 0:
        out = correct / len(new_target)
    else:
        out = 0
        
    return out

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        new_pred = pred[target != -100]
        new_target = target[target != -100]
        precision = precision_score(new_pred.cpu().numpy(), new_target.cpu().numpy(), average='macro', zero_division=0)
        #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    return precision

def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        new_pred = pred[target != -100]
        new_target = target[target != -100]
        recall = recall_score(new_pred.cpu().numpy(), new_target.cpu().numpy(), average='macro', zero_division=0)
        #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    return recall

def conf_matrix(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        new_pred = pred[target != -100]
        new_target = target[target != -100]
        cm = confusion_matrix(new_target.cpu().numpy(), new_pred.cpu().numpy())
        #tn, fp, fn, tp = cm.ravel()
        #print('Confusion matrix:', cm)
        #out_msg = ('tn: %d, fp: %d, fn: %d, tp: %d' % (tn, fp, tn, tp))
    return cm

def mse(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        new_output = target[target != -100]
        new_target = target[target != -100]
        mse = mean_squared_error(new_target.cpu().numpy(), new_output.cpu().numpy())
        #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    return mse    


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        new_pred = pred[target != -100]
        new_target = target[target != -100]
        correct = 0
        for i in range(k):
            correct += torch.sum(new_pred[:, i] == new_target).item()
    return correct / len(new_target)
