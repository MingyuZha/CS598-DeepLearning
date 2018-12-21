import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
from torch.utils import model_zoo
import numpy as np
#pretrained model resnet18


def organize_images(txt_path, image_path):
	with open(txt_path) as f:
		content = f.readlines()
	for ele in content:
		ele = ele.strip('\n')
		desp = ele.split('\t')
		category = desp[1]
		filename = desp[0]
		file_dir = os.path.join(image_path, category)
		if not os.path.exists(file_dir):
			os.mkdir(file_dir)
		src_file_path = os.path.join(image_path, filename)
		dst_file_path = os.path.join(file_dir, filename)
		os.rename(src_file_path, dst_file_path)

if __name__ == '__main__':
    txt_path = './tiny-imagenet-200/val/val_annotations.txt'
    image_path = './tiny-imagenet-200/val/images'
    organize_images(txt_path, image_path)


    
