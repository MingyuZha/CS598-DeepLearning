import numpy as np
import torch
import torch.nn as nn
import os.path
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import collections
import torchvision.models as models
import os
import json
import io
from PIL import Image
import nltk
nltk.download('punkt')
from pycocotools.coco import COCO

def tokenize_sentence(line):
    line = line.split('\t')[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = line.replace('\n','')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    return line
    
####################################################################
#MSCOCO dataset #############################Loading in loader phase
####################################################################
# http://cocodataset.org/#format-data for data format
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb for dataset Usage
class MSCOCO(Dataset):
    def __init__(self, root = './MSCOCO',train=True, transform=None, target_transform=None, download=False):
        self.train = train
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        dataType = 'train2014'
        annFile='{}/annotations/captions_{}.json'.format(root,dataType)
        self.coco_train = COCO(annFile)
        self.train_image_list = json.load(io.open(annFile))['annotations']

        dataType = 'val2014'
        annFile='{}/annotations/captions_{}.json'.format(root,dataType)
        self.coco_test = COCO(annFile)     
        self.test_image_list = json.load(io.open(annFile))['annotations']
    def __getitem__(self, index):
        image_id = self.coco_train.loadImgs(self.train_image_list[index]['image_id'])
        if not self.train:
            image_id = self.coco_test.loadImgs(self.test_image_list[index]['image_id'])
        folder_name = self.root +"/" +'train2014/'
        if not self.train:
            folder_name = self.root +"/" +'val2014/'
        filename = folder_name +   image_id[0]['file_name']
        caption = self.train_image_list[index]['caption']
        if not self.train:
            caption = self.test_image_list[index]['caption']
        img = (Image.open(filename).convert('RGB'))
        if self.transform is not None:
            img = self.transform(img)
        return img, tokenize_sentence(caption)

    def __len__(self):
        if self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)
        
        

        
        
        
        
        
class MSCOCOFORSCORE(Dataset):
    def __init__(self, root = './MSCOCO', transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        dataType = 'val2014'
        annFile='{}/annotations/captions_{}.json'.format(root,dataType)
        self.coco_test = COCO(annFile)     
        self.test_image_list = json.load(io.open(annFile))['annotations']
        self.test_image_to_caption = collections.defaultdict(list)
        for anno in self.test_image_list:
            line = anno['caption']
            line = line.replace('<br />',' ')
            line = line.replace('\x96',' ')
            line = line.replace('\n','')
            line = line.replace('.','')
            self.test_image_to_caption[anno['image_id']].append(line)
    def __getitem__(self, index):
        image_id = self.coco_test.loadImgs(self.test_image_list[index]['image_id'])
        folder_name = self.root +"/" +'val2014/'
        filename = folder_name + image_id[0]['file_name']
        img = (Image.open(filename).convert('RGB'))
        if self.transform is not None:
            img = self.transform(img)
        return image_id[0]['file_name'], img, self.test_image_to_caption[self.test_image_list[index]['image_id']]

    def __len__(self):
        return len(self.test_image_list)