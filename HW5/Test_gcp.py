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
    network.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    test_dataset = Test_Dataset(root_dir=root_dir+'/val', transform=transform)
    train_dataset = Train_Dataset(root_dir=root_dir+'/train', transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # ---------------------------------------------------------------------
    #     Calculate feature embeddings of all images in train folder
    # ---------------------------------------------------------------------
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")


    # train_feature_embeddings = []
    # X_train = []      #Stores feature embeddings of all images in training dataset
    # X_test = []
    # y_train = []      #Stores corresponding class labels
    # y_test = []
    
    
    # with torch.no_grad():
    #     print ("Start calculating feature embeddings of training data")
    #     stamp = time()
    #     for index, data in enumerate(train_data_loader):
    #         image, label = data
    #         image = image.to(device)
    #         f = network(image).cpu().numpy().tolist()
    #         X_train += f
    #         y_train += label
    #         if (index % 100 == 99):
    #             print ("Finished %.2f percent, cost: %.5f seconds"%((index+1)*100/(100000./BATCH_SIZE), time()-stamp))
    #     y_train = np.array(y_train)
    #     X_train = np.array(X_train)
    #     print ("Finished Calculating feature embeddings of images in training data")
    #     print ("Start calculating feature embeddings of testing data")
    #     stamp = time()
    #     for index, data in enumerate(test_data_loader):
    #         image, label = data
    #         image = image.to(device)
    #         f = network(image).cpu().numpy().tolist()
    #         X_test += f
    #         y_test += label
    #         if (index % 100 == 99):
    #             print ("Finished %.2f percent, cost: %.5f seconds"%((index+1)*100/(10000./BATCH_SIZE), time()-stamp))
    #     y_test = np.array(y_test)
    #     X_test = np.array(X_test)

    #     print ("Finished Calculating feature embeddings of images in testing data")
    start = time()
    for i in range(len(X_test)):
        tmp = np.array([[X_test[i]] for j in range(len(X_train))])
        all_dist = np.linalg.norm((tmp-X_train), ord=2, axis=1)
        indices = np.argpartition(all_dist, k)[:k]
        tmp_labels = [y_train[j] for j in indices]
        correct = np.sum(np.array(tmp_labels) == y_test[i])
        precision = correct/k
        all_precision.append(precision)

        if (i%100==99):
            print ("[%d/%d]: %.4f%%, cost: %.4fs"%(i+1, len(X_test), np.mean(all_precision), time()-start))

    # print ("Start initializing and fitting the KNN model")
    # start = time()
    # neigh = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', n_jobs=32).fit(X_train, y_train)
    # print ("Finished, cost: %.5fs"%(time()-start))
    # print ("Start calculating the indices of %d nearest neighbors"%(k))
    # start = time()
    # neighbors_index = neigh.kneighbors(X_test, return_distance=False)
    # print ("Finished, cost: %.5fs"%(time()-start))
    # print ("shape of neighbors_index: ", neighbors_index.shape)

    # print ("Start calculating the accuracy")
    # start = time()
    # labels_neigh = [[y_train[m] for m in neighbors_index[n]] for n in range(len(X_test))]
    # all_precision = []
    # for i in range(len(X_test)):
    #     correct = np.sum(np.array(labels_neigh[i]) == y_test[i])
    #     precision = correct/k
    #     all_precision.append(precision)
    # print ("Finished, cost: %.5fs"%(time()-start))
    avg_precision = np.mean(all_precision)
    return avg_precision


    



    #np.save("Feature_Embedding_X.npy", X)
    #np.save("Labels_X.npy", y)
    # print ("Finished Calculating feature embeddings of training data")
    # X = np.load("Feature_Embedding_X.npy")
    # y = np.load("Labels_X.npy")
    # ---------------------------------------------------------------------
    #                    Calculate Overall Accuracy
    # ---------------------------------------------------------------------
    # l_precision = []
    # # f_q = network(query_image).numpy()
    # print ("Start calculating overall accuracy")
    # stamp = time()
    # neigh = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', n_jobs=32)
    # neigh.fit(X, y)             #Shape of X is (100000, 4096), Shape of y is (1000000,)
    # print ("Initialize model costs: %.2f seconds"%(time()-stamp))
    # stamp = time()
    # with torch.no_grad():
    #     for index, data in enumerate(test_data_loader):
    #         image, ground_truth = data
    #         image = image.to(device)
            
    #         f = network(image).cpu().numpy()  #compute the feature embedding of the query image

    #         neighbors_index = neigh.kneighbors(f, return_distance=False)
    #         labels_neigh = [[y[m] for m in neighbors_index[n]] for n in range(len(f))]
         
    #         for n in range(len(f)):
    #             correct = np.sum(np.array(labels_neigh[n]) == ground_truth[n])        #labels_neigh[n] is a list of labels of all k nearest neighbors, ground_truth[n] is a scalar
    #             l_precision.append(correct/k)
    #         print ("precision: ", l_precision)

    #         if (index % 10 == 9):
    #             print ("[%d, %d]: %.5f seconds"%(index+1, int(10000/12), time()-stamp))
    # avg_precision = np.mean(l_precision)*100        #Calculate the overall accuracy

    # return avg_precision


if __name__ == '__main__':
    model = torchvision.models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4096, bias=True)
    model.load_state_dict(torch.load("model_trained_14_epochs.ckpt"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    BATCH_SIZE = 100
    precision = Testing_Stage(model, './tiny-imagenet-200', 30)
    print (precision)



