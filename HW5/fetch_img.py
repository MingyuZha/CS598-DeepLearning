import numpy as np
import matplotlib.pyplot as plt
import os
from time import *
from torch.utils.data import Dataset
class Train_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    def __getitem__(self, index):
        img_classes = [f for f in os.listdir(self.root_dir) if not f.startswith('.')]
        class_index = int(index/500)
        label = img_classes[class_index]
        img_names = [f for f in os.listdir(self.root_dir+'/'+label+'/images') if f.endswith('.JPEG')]
        img_index = index - 500*class_index
        img_path = self.root_dir+'/'+label+'/images/'+img_names[img_index]
        with PIL.Image.open(img_path) as img:
            img = img.convert('RGB')
        if (self.transform is not None):
            img = self.transform(img)
        return img, label
    def __len__(self):
        return 500*200

dataset = Train_Dataset('./tiny-imagenet-200/train')
top = []
top_dist = []
bottom = []
bottom_dist = []
row_counter = 0
flag = True
with open('./Ranking_Top_and_Bottom_10_2.txt', 'r') as file:
    for line in file:
        row_counter += 1
        if row_counter < 8: continue
        if (line.strip('\n') == 'Bottom 10: '): 
            flag = False
            continue
        if (line == '\n'): continue
        if flag:
            content = line.split()
#             print (content)
            top.append(int(content[0]))
            top_dist.append(float(content[1]))
        else:
            content = line.split()
            bottom.append(int(content[0]))
            bottom_dist.append(float(content[1]))

print (dataset[top[0]][1])








