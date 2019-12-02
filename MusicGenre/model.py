from __future__ import print_function, division
import torch
import torch.nn as nn
import time
from torchvision import models
from efficientnet_pytorch import EfficientNet

class dense201(nn.Module):
    def __init__(self,n_classes = 8):
        super(dense201, self).__init__()
        model = models.densenet201(pretrained = True)
        model = model.features
        for child in model.children():
          for layer in child.modules():
            layer.requires_grad = False
            if(isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d)):
              layer.requires_grad = True
        self.model = model
        self.linear = nn.Linear(3840, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, n_classes)
        self.bn1 = nn.BatchNorm1d(3840)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res
    
class effnet(nn.Module):
    def __init__(self,n_classes = 8):
        super(effnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.linear = nn.Linear(1000, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, n_classes)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(out))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res
class dense161(nn.Module):
    def __init__(self,n_classes = 8):
        super(dense161, self).__init__()
        model = models.densenet161(pretrained = True)
        model = model.features
        for child in model.children():
          for layer in child.modules():
            layer.requires_grad = False
            if(isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d)):
              layer.requires_grad = True
        self.model = model
        self.linear = nn.Linear(4416, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, n_classes)
        self.bn1 = nn.BatchNorm1d(4416)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res
class dense121(nn.Module):
    def __init__(self,n_classes = 8):
        super(dense121, self).__init__()
        model = models.densenet121(pretrained = True)
        model = model.features
        for child in model.children():
          for layer in child.modules():
            layer.requires_grad = False
            if(isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d)):
              layer.requires_grad = True
        self.model = model
        self.linear = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, n_classes)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res
