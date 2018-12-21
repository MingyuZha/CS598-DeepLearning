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

class Test_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = []
        self.labels = []
        with open(os.path.join(self.root_dir, 'val_annotations.txt'), 'r') as file:
            for line in file:
                content = line.split()
                self.img_names.append(content[0])
                self.labels.append(content[1])
    def __getitem__(self, index):
        with PIL.Image.open(os.path.join(self.root_dir+'/images', self.img_names[index])) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return 10000

class Train_Dataset(Dataset):
    def __init__(self, root_dir, transform):
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



def Testing_Stage(network, root_dir, k=30):
    """
    Feed one query image to the network and get the feature embedding of the query image. Compare the feature embedding of the query image to 
    all the feature embeddings in the whole training dataset. Rank the result and output the top k results.
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    test_dataset = Test_Dataset(root_dir=root_dir+'/val', transform=transform)
    train_dataset = Train_Dataset(root_dir=root_dir+'/train', transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # ---------------------------------------------------------------------
    #     Calculate feature embeddings of all images in train folder
    # ---------------------------------------------------------------------
    train_feature_embeddings = []
    X = []      #Stores feature embeddings of all images in training dataset
    y = []      #Stores corresponding class labels
    print ("Start calculating feature embeddings of training data")
    for index, data in enumerate(train_data_loader):
        image, label = data
        image = image.to(device)
        stamp = time()
        f = network(image).cpu().numpy()
        # print ("forward costs: %.2f seconds"%(time()-stamp))
        # temp = {'Image': image.cpu(), 'Class': label.cpu(), 'Embedding': f}
        stamp = time()
        for i, l in enumerate(label):
            # temp = {'Class': l, 'Embedding': f[i]}
            # train_feature_embeddings.append(temp)
            X.append(f[i])
            y.append(l)
        # print ("saving costs: %.6f seconds"%(time()-stamp)) 
        if (index % 100 == 99):
            print ("Finished %.2f percent"%((index+1)*100/(100000./BATCH_SIZE)))

    print ("Finished Calculating feature embeddings of images in training data")
    X, y = np.array(X), np.array(y)
    np.save("Feature_Embedding_X.npy"X, X)
    np.save("Labels_X.npy", y)
    # ---------------------------------------------------------------------
    #                    Calculate Overall Accuracy
    # ---------------------------------------------------------------------
    l_precision = []
    # f_q = network(query_image).numpy()
    print ("Start calculating overall accuracy")
    neigh = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', n_jobs=32)
    neigh.fit(X, y)             #Shape of X is (100000, 4096), Shape of y is (1000000,)
    for index, data in enumerate(test_data_loader):
        image, ground_truth = data
        image = image.to(device)

        f = network(image).cpu().numpy()  #compute the feature embedding of the query image
        print ('the shape of f: ', f.shape)
        neighbors_index = neigh.kneighbors(f, return_distance=False)
        labels_neigh = [[y[m] for m in neighbors_index[n]] for n in range(len(f))]
        # for t in train_feature_embeddings:
        #     distance = np.linalg.norm(f-t['Embedding'], ord=2)
        #     heappush(pq, (distance, t))
        # correct = 0.
        for n in range(len(f)):
            correct = np.sum(labels_neigh[n] == ground_truth[n])        #labels_neigh[n] is a list of labels of all k nearest neighbors, ground_truth[n] is a scalar
            l_precision.append(correct/k)
        # for i in range(k):
        #     # current = heappop(pq)
        #     if (ground_truth == current['Class']):
        #         correct += 1
        # precision = correct/k
        # l_precision.append(precision)
        if (index % 100 == 99):
            print ("Finished calculating accuracy of %d mini-batches images in testing dataset"%(index+1))
    avg_precision = np.mean(l_precision)        #Calculate the overall accuracy

    return avg_precision


if __name__ == '__main__':
    model = torch.load('temp_model_trained_3_epochs.ckpt', map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    BATCH_SIZE = 32
    precision = Testing_Stage(model, './tiny-imagenet-200')
    print (precision)



