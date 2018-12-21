import os
import torch 
import random
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
from time import *
import torchvision.transforms as T
import PIL
# import pickle
# from progress.bar import ChargingBar
from heapq import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def RandomSample(model, k=5):
    print ("Preprocessing..")
    all_imgs = []
    val_labels = []
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as file:
        for line in file:
            content = line.split()
            all_imgs.append(content[0])
            val_labels.append(content[1])
    print ("Done!")
    print ("Randomly sampling 5 images from validation set")
    cur_classes = []
    cur_image = []
    while True:
        if (len(cur_classes) == 5): break
        rand_idx = random.randint(0, len(all_imgs)-1)
        if (val_labels[rand_idx] in cur_classes): continue
        cur_classes.append(val_labels[rand_idx])
        cur_image.append(all_imgs[rand_idx])
    print ("Done!")
    #---------------------------------------------------------
    #                   Show Top 10 
    #---------------------------------------------------------
    print ("Ranking top 10")
    start = time()
    results_idx = []
    results_dist = []
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    neigh = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', n_jobs=32).fit(X_train, y_train)
    print ("Finished initializing knn model")
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    model.eval()
    with torch.no_grad():
        for idx, img in enumerate(cur_image):
            with PIL.Image.open('./tiny-imagenet-200/val/images/'+img) as image:
                image = image.convert('RGB')
            image = (transform(image).view(1,3,224,224)).cuda()
            f = model(image).cpu().numpy()
            distance, neigh_indices = neigh.kneighbors(f, n_neighbors=10, return_distance=True)
            results_idx.append(neigh_indices[0])
            results_dist.append(distance[0])
            print ("[%d/%d]: %.4fs"%(idx+1, 5, time()-start))
    # rand_idx = [random.randint(0, len(all_classes)-1) for i in range(k)]
    # selected_class = [all_classes[i] for i in rand_idx]
    print ("Done! Used: %.5fs"%(time()-start))
    #---------------------------------------------------------
    #                   Show Bottom 10
    #---------------------------------------------------------
    print ("Ranking bottom 10")
    start = time()
    bottom_idx = []
    bottom_dist = []
    with torch.no_grad():
        for idx, img in enumerate(cur_image):
            with PIL.Image.open('./tiny-imagenet-200/val/images/'+img) as image:
                image = image.convert('RGB')
            image = (transform(image).view(1,3,224,224)).cuda()
            f = model(image).cpu().numpy()
            distance, neigh_indices = neigh.kneighbors(f, n_neighbors=100000, return_distance=True)
            bottom_idx.append(neigh_indices[0][-10:])
            bottom_dist.append(distance[0][-10:])
            print ("[%d/%d]: %.4fs"%(idx+1, 5, time()-start))
    print ("Done!")
    return cur_classes, cur_image, results_idx, results_dist, bottom_idx, bottom_dist
    
    # train_dataset = Train_Dataset(root_dir='./tiny-imagenet-200/train', transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=4)

    # model.eval()
    
if __name__ == '__main__':
    model = torchvision.models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4096, bias=True)
    model.load_state_dict(torch.load("model_trained_14_epochs.ckpt"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    cur_classes, cur_image, results_idx, results_dist, bottom_idx, bottom_dist = RandomSample(model)
    print ("Start writing output")
    with open('Ranking_Top_and_Bottom_10.txt', 'w') as file:
        file.write("Sampled images from validation set:\n")
        for i in range(5):
            file.write(cur_classes[i]+" "+cur_image[i]+"\n")
        file.write("Top 10: \n")
        for i in range(5):
            for j in range(10):
                file.write(results_idx[i][j]+" "+results_dist[i][j]+"\n")
            file.write("\n")
        file.write("Bottom 10: \n")
        for i in range(5):
            for j in range(10):
                file.write(bottom_idx[i][j]+" "+bottom_dist[i][j]+"\n")
            file.write("\n")
    print ("Done!")









