import torch
import torch.nn as nn
import torch.nn.functional as F

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class DiceCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoef, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        scores = []
        batch_size = len(inputs)
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)  
        for i in range(batch_size):
                     
            #flatten label and prediction tensors
            input = inputs[i].view(-1)
            target = targets[i].view(-1)
            
            intersection = (input * target).sum()                            
            dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)
            scores.append(dice)
        
        return sum(scores) / len(scores)