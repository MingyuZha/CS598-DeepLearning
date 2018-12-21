
# coding: utf-8

# In[19]:


import torch
from random import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from time import *
from torch.nn import *
from numpy import *
from torchvision import *
from torch.utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import numpy
# In[20]:


# In[21]:


class Data(torch.utils.data.Dataset):
    def __init__(self, root_dir, train= None, transform=None):
        super(Data, self).__init__()
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.root_dir, transform = t)
    def __len__(self):
         return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            query, query_cat = self.dataset[idx]
            positive_idx = torch.randint(0, 499, size=(1, ),dtype=torch.int64) + query_cat*500
            negative_cat = torch.randint(0,199, size=(1, ),dtype=torch.int64)
            if negative_cat ==query_cat:
                negative_idx = 199 - query_cat
            negative_idx = torch.randint(0, 499, size=(1, ),dtype=torch.int64) + negative_cat*500
            positive, a = self.dataset[positive_idx]
            negative, b  = self.dataset[negative_idx]
            sample = (query,positive,negative)
            return sample
        query, query_cat = self.dataset[idx]
        return query, query_cat



# In[17]:





# In[4]:


class TripletLossFunc(nn.Module):
    def __init__(self, g):
        super(TripletLossFunc, self).__init__()
        self.g = torch.tensor(g)
        return


# In[ ]:


def train(model, dataset, lr,mtm, num_epochs, margin):
    '''
    
    '''
    #print(model)
    start = time()
    #torchvision.models.resnet101(pretrained = True)
    #for params in net.params():
        #params.requires_grad = False
    #net.fc2 = nn.linear(net.fc1,4096)
    triplet_loss = TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum= mtm)
    #testset = testloader
    loss_rec = [ ] 
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            query, positive, negative = data
            query, positive, negative = query.to(device), positive.to(device), negative.to(device)
            embedding1 = model(query)
            embedding2 = model(positive)
            embedding3 = model(negative)
            optimizer.zero_grad()

            loss = triplet_loss(embedding1, embedding2, embedding3)
            loss.backward()#calculate the gradient
            if (epoch)>3:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if (state['step']>=1024):
                            state['step'] = 1000
            optimizer.step()#update the paramters
            running_loss += loss.item()
            if i % 100  == 99:    # print every 2000 mini-batches
                print('In %dth epoch, loss over batch %d: is %.3f, %2fs used.' %     (epoch + 1, i + 1, running_loss / 100., time()-start))
                loss_rec.append(numpy.around(running_loss,4))
                running_loss = 0.0
        #if epoch % 2==1:
            #print("Epoch ", epoch+1,'finished,', 'the accuracy on trainset is ',acc_cal(model,trainloader2))
        
        print("Epoch ", epoch+1,'finished,', 'the accuracy on testset is ',numpy.around(acc_cal(model,test_loader, train_loader2),4),'%.', time()-start,'s used')
        path = 'SGD_epoch_'+str(epoch+1)+'.ckpt'
        torch.save(model,path)
        if epoch%2 == 1:
            temp_lr = lr / 3
            update_lr(optimizer, temp_lr)
        if epoch %4 ==3 :
            print('Now writing loss to record')
            with open('loss.txt', 'w') as f:
                for item in loss_rec:
                    f.write("%s\n" % item)
    print('checkpoint created, learning rate updated!')
    print('Finished Training')

def update_lr(optimizer,lr):
        for param_groups in optimizer.param_groups:
                    param_groups["lr"] = lr

def acc_cal(model, testset, trainset):
    
    correct = 0
    total = 0
    query_space = [ ]
    train_data = torch.zeros([1,1024])
    train_data = train_data.to(device)
    test_data = torch.zeros([1,1024])
    test_data = test_data.to(device)
    test_labels = torch.zeros([1],dtype = torch.long)
    test_labels = test_labels.to(device)
    model.eval()
    with torch.no_grad():
        print("Fetching training data")
        for i, (images, labels) in enumerate(trainset, 0):
            images = images.to(device)
            outputs = model(images).to(device)
            train_data = torch.cat((train_data, outputs),0)
        train_data = train_data[1:,:]
        print("Fitting knn")
        nbrs = NearestNeighbors(n_neighbors=30, algorithm='kd_tree',n_jobs =32).fit(train_data)
        print("Fetching test data")
        for i, data in enumerate(testset, 0):
            
            #print(type(labels))
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            test_data = torch.cat((test_data,outputs),0)
            test_labels = torch.cat((test_labels, labels),0)

            #total += labels.size(0)
        print("training data, query image been all fetched")
        test_data = test_data[1:,:]
        test_labels = test_labels[1:].reshape(-1,1)            
        _, indices = nbrs.kneighbors(test_data)
        correct = torch.sum(indices//500 == test_labels).item()/30/10000
            #print(type(indices),numpy.shape(indices),type(labels.data.numpy()),numpy.shape(labels.data.numpy()))
            #print(indices, labels.data.numpy())
    return 100*correct

 


# In[ ]:

def resnet18(pretrained = True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='./'))
    return model

def resnet101(pretrained = True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='./'))
        
    #q = 0
    #for param in model.parameters():
     #   q +=1
      #  if q<= 312:
       #     param.requires_grad = False
    model.fc = nn.Linear(2048, 1024)
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
    return model
# In[83]:
if __name__ == '__main__':
    BATCH = 12
    t = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])
    train_dir = './tiny-imagenet-200/train'
    test_dir = './tiny-imagenet-200/val/images'
    # train_dataset = Data(train_dir, train= True, transform=t)
    # train_loader = data.DataLoader(train_dataset, batch_size=BATCH, num_workers=4)
    train_dataset2 = Data(train_dir, transform=t)
    train_loader2 = data.DataLoader(train_dataset2, batch_size=200, num_workers=4)
    test_dataset = Data(test_dir, transform=t)
    test_loader = data.DataLoader(test_dataset, batch_size=200, num_workers=4)
    # model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth','resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'}
    model = torchvision.models.resnet101()
    model.fc = nn.Linear(model.fc.in_features, 4096)
    model.load_state_dict(torch.load('model_trained_20_epochs.ckpt'))
    # net = resnet101()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    acc_cal(model, test_loader, train_loader)
    # train(net, train_loader, 0.0003,0.9, 10, 1.0)


