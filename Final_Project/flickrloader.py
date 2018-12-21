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
import io
from PIL import Image
import nltk
nltk.download('punkt')

def tokenize_sentence(line):
    line = line.split('\t')[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = line.replace('\n','')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    return line


train_num = 28000



########################################################################
#flickr8K, 30K dataset for score evaluation#############################
########################################################################



class Flickr30KFORSCORE(Dataset):
    def __init__(self, root = './flickr30k_images/flickr30k_images/', transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.root = root
        f = io.open('./flickr30k_images/results_20130124.token', 'r', encoding='utf-8') 
        lines=f.readlines()
        # training image filename list
        self.train_image_list = []
        self.test_image_list = []
        self.image2caption = collections.defaultdict(list)
        for x in lines:  
            description = x.split('\t')[1]
            imgname = x.split('\t')[0].split('#')[0]     
            line = description.replace('<br />',' ')
            line = line.replace('\x96',' ')
            line = line.replace('\n','')   
            line = line.replace('.','')     
            self.image2caption[imgname].append(line)
        all_images = list(self.image2caption.keys())
        self.train_image_list = all_images[0: train_num]
        self.test_image_list = all_images[train_num:]
        # training image file list(actual numpy array)
        self.train_image = []
        self.test_image = []
        # 500 for testing, could be removed after testing is done
        # count = 0
        # for name in self.train_image_list:
        #     img = (Image.open(root + name).convert('RGB'))
        #     count += 1
        #     if (count > 500):
        #         break;
        #     if self.transform is not None:
        #         self.train_image.append(self.transform(img))
        # count = 0
        
        # for name in self.test_image_list:
        #     img = (Image.open(root + name).convert('RGB'))
        #     if (count > 500):
        #         break;
        #     if self.transform is not None:
        #         self.test_image.append(self.transform(img))        
    def __getitem__(self, index):
        img_name = self.test_image_list[index]
        img = (Image.open(self.root + img_name).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img_name ,img, self.image2caption[img_name]
    def __len__(self):
        return len(self.test_image_list)







class Flickr8KFORSCORE(Dataset):
    def __init__(self, root = './flickr8k_images/Flickr8k_Dataset/', transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.train_image_list = []
        self.test_image_list = []
        f = io.open('./flickr8k_images/Flickr8k_text/Flickr_8k.testImages.txt', 'r', encoding='utf-8') 
        lines=f.readlines()
        self.test_image_list = []
        for x in lines:
            self.test_image_list.append(x.strip())

        self.root = root
        f = io.open('./flickr8k_images/Flickr8k_text/Flickr8k.lemma.token.txt', 'r', encoding='utf-8')
        lines=f.readlines()
        self.image2caption = collections.defaultdict(list)
        for x in lines:  
            description = x.split('\t')[1]
            line = description.replace('<br />',' ')
            line = line.replace('\x96',' ')
            line = line.replace('\n','')   
            line = line.replace('.','')    
            imgname = x.split('\t')[0].split('#')[0]               
            self.image2caption[imgname].append(line)
    def __getitem__(self, index):
        img_name = self.test_image_list[index]
        img = (Image.open(self.root + img_name).convert('RGB'))
        if self.transform:
            img = self.transform(img)

        return img_name , img, self.image2caption[img_name]

    def __len__(self):
        return len(self.test_image_list)




class Flickr30K(Dataset):
    def __init__(self, root = './flickr30k_images/flickr30k_images/', train=True, transform=None, target_transform=None, download=False):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.root = root
        f = io.open('./flickr30k_images/results_20130124.token', 'r', encoding='utf-8') 
        lines=f.readlines()
        # training image filename list
        self.train_image_list = []
        self.test_image_list = []
        self.image2caption = collections.defaultdict(list)
        for x in lines:  
            description = x.split('\t')[1]
            imgname = x.split('\t')[0].split('#')[0]               
            self.image2caption[imgname].append(description)
        all_images = list(self.image2caption.keys())
        self.train_image_list = all_images[0: train_num]
        self.test_image_list = all_images[train_num:]
        # training image file list(actual numpy array)
        self.train_image = []
        self.test_image = []
        # 500 for testing, could be removed after testing is done
        # count = 0
        # for name in self.train_image_list:
        #     img = (Image.open(root + name).convert('RGB'))
        #     count += 1
        #     if (count > 500):
        #         break;
        #     if self.transform is not None:
        #         self.train_image.append(self.transform(img))
        # count = 0
        
        # for name in self.test_image_list:
        #     img = (Image.open(root + name).convert('RGB'))
        #     if (count > 500):
        #         break;
        #     if self.transform is not None:
        #         self.test_image.append(self.transform(img))        
    def __getitem__(self, index):
        img_name = self.train_image_list[index]
        if not self.train:
            img_name = self.test_image_list[index]
        img = (Image.open(self.root + img_name).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img, tokenize_sentence(self.image2caption[img_name][0])
    def __len__(self):
        if self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)




###############################################
#flickr8K dataset #############################
###############################################

class Flickr8K(Dataset):
    def __init__(self, root = './flickr8k_images/Flickr8k_Dataset/',train=True, transform=None, target_transform=None, download=False):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.train_image_list = []
        self.test_image_list = []
        if self.train:
            f = io.open('./flickr8k_images/Flickr8k_text/Flickr_8k.trainImages.txt', 'r', encoding='utf-8') 
            lines=f.readlines()
            for x in lines:
                self.train_image_list.append(x.strip())
        else:
            f = io.open('./flickr8k_images/Flickr8k_text/Flickr_8k.testImages.txt', 'r', encoding='utf-8') 
            lines=f.readlines()
            self.test_image_list = []
            for x in lines:
                self.test_image_list.append(x.strip())

        self.root = root
        f = io.open('./flickr8k_images/Flickr8k_text/Flickr8k.lemma.token.txt', 'r', encoding='utf-8')
        lines=f.readlines()
        self.image2caption = collections.defaultdict(list)
        for x in lines:  
            description = x.split('\t')[1]
            imgname = x.split('\t')[0].split('#')[0]               
            self.image2caption[imgname].append(description)
    def __getitem__(self, index):
        if self.train:
            img_name = self.train_image_list[index]
            img = (Image.open(self.root + img_name).convert('RGB'))
        else:
            img_name = self.test_image_list[index]
            img = (Image.open(self.root + img_name).convert('RGB'))
        if self.transform:
            img = self.transform(img)

        return img, tokenize_sentence(self.image2caption[img_name][0])

    def __len__(self):
        if self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)


