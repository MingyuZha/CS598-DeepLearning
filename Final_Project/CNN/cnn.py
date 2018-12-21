import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from tqdm import tqdm

class CNN_model(nn.Module):
    def __init__(self, embedding_feature = 1000, pre_trained = True, model_use = 'resnet'):
        super(CNN_model, self).__init__()
        self.model = models.resnet50(pretrained = pre_trained)
        self.model_use = model_use
        if (model_use == 'inception'):
            self.model = models.inception_v3(pretrained = pre_trained)
        if (model_use == 'densenet'):
            self.model = models.densenet161(pretrained = pre_trained)
        if model_use != 'densenet':
            fc_features = self.model.fc.in_features
        else:
            fc_features = self.model.classifier.in_features
        
        self.feature_number = embedding_feature
        # https://blog.csdn.net/whut_ldz/article/details/78845947 for how to customize the CNN model
        if (embedding_feature != 1000):
            if model_use != 'densenet':
                self.model.fc = nn.Linear(fc_features, embedding_feature)
            else:
                self.model.classifier = nn.Linear(fc_features, embedding_feature)
        
    def forward(self, x):
        x = self.model(x)
        if isinstance(x,tuple):
            x = x[0]
        x = x.view(-1, self.feature_number)
        return x


