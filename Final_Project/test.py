from flickrloader import Flickr30K, Flickr8K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import os.path
from flickrloader import Flickr30K
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.models as models
import os
import io
import json
import nltk
from PIL import Image
from mscocoloader import MSCOCO

from pycocotools.coco import COCO
# https://zhuanlan.zhihu.com/p/30385675 for customizing collate_fn
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  
# 0 for unknown
# 1 for start
# 2 for end
# 3 for pad
imdb_dictionary = np.load('./word_to_id.npy')
actual_dic = {}
for i in range(imdb_dictionary.shape[0]):
    actual_dic[imdb_dictionary[i]] = i
def tokenize_sentence(line):
    line = line.split('\t')[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = line.replace('\n','')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    return line

def my_collate(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    image, caption = zip(*batch)
    
    max_len = len(caption[0])
    
    caption_new = []
    sentence_length = [] #prepared for the packed_pad_sequence
    for i in range(len(batch)):
        diff = max_len - len(caption[i])
        caption_curr = caption[i]
        sentence_length.append(len(caption[i]) + 2)
        caption_curr = [1] + [actual_dic[token] if token in actual_dic else 0 for token in caption[i]] + [2] + diff * [3]
        assert len(caption_curr) == max_len + 2
        caption_new.append(caption_curr)
    caption_new = torch.LongTensor(caption_new)
    image = torch.stack(image, 0)
    return image, caption_new, sentence_length



train_data = MSCOCO(train=True, transform=data_transform, target_transform=None)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=32, shuffle=True,num_workers=32, collate_fn=my_collate)

filename = './flickr30k_images/flickr30k_images/16653104.jpg'
img = (Image.open(filename).convert('RGB'))
img = data_transform(img)
