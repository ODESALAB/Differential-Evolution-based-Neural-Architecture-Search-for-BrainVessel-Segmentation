import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        batch_size = len(inputs)
        scores = []
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        for i in range(batch_size):

            #flatten label and prediction tensors
            inputs_f = inputs[i].view(-1)
            targets_f = targets[i].view(-1)
            
            #intersection is equivalent to True Positive count
            #union is the mutually inclusive area of all labels & predictions 
            intersection = (inputs_f * targets_f).sum()
            total = (inputs_f + targets_f).sum()
            union = total - intersection 
            
            IoU = (intersection + smooth)/(union + smooth)
            scores.append(1. - IoU)
                
        return sum(scores) / len(scores)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        batch_size = len(inputs)
        scores = []
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)  
        for i in range(batch_size):
            
            #flatten label and prediction tensors
            input = inputs[i].view(-1)
            target = targets[i].view(-1)
            
            intersection = (input * target).sum()                            
            dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
            scores.append(1. - dice)
        return sum(scores) / len(scores)