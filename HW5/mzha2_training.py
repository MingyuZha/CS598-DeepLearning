import os
import torch 
import random
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
from time import *
#from ResNet import ResNet
import torchvision.transforms as T
import PIL
# import pickle
# from progress.bar import ChargingBar
from heapq import *
import matplotlib.pyplot as plt

class Triplets_Dataset(Dataset):
    def __init__(self, sampler, transform=None):
        self.sampler = sampler
        self.transform = transform
    def __getitem__(self, index):
        triplets = {}
        with PIL.Image.open(self.sampler[index]['Query']) as img:
            img = img.convert('RGB')
            triplets['Query'] = img
        with PIL.Image.open(self.sampler[index]['Positive']) as img:
            img = img.convert('RGB')
            triplets['Positive'] = img
        with PIL.Image.open(self.sampler[index]['Negative']) as img:
            img = img.convert('RGB')
            triplets['Negative'] = img

        if self.transform is not None:
            triplets['Query'] = self.transform(triplets['Query'])
            triplets['Positive'] = self.transform(triplets['Positive'])
            triplets['Negative'] = self.transform(triplets['Negative'])
        return triplets
    def __len__(self):
        return len(self.sampler)


def Triplets_PreSampler():
    all_classes = []
    with open('./tiny-imagenet-200/wnids.txt', 'r') as file:
        for line in file:
            all_classes.append(line.strip('\n\r'))
    triplets = []
    for Q_class in all_classes:
        all_imgs = os.listdir('./tiny-imagenet-200/train/'+Q_class+'/images')
        for img_name in all_imgs:
            path = './tiny-imagenet-200/train/'+Q_class+'/images/'+img_name
            p = './tiny-imagenet-200/train/'+Q_class+'/images/'+img_name
            p_plus = './tiny-imagenet-200/train/'+Q_class+'/images/'+random.sample(all_imgs, k=1)[0]
            while p == p_plus:
                p_plus = './tiny-imagenet-200/train/'+Q_class+'/images/'+random.sample(all_imgs, k=1)[0]
            while True:
                N_class = random.sample(all_classes, k=1)[0]
                if (N_class != Q_class): break
            N_class_imgs = os.listdir('./tiny-imagenet-200/train/'+N_class+'/images')
            p_minus = './tiny-imagenet-200/train/'+N_class+'/images/'+random.sample(N_class_imgs, k=1)[0]
            triplets.append({'Query': p, 'Positive': p_plus, 'Negative': p_minus})
    return triplets

  


def Testing_Stage(query_image, k):
    """
    Feed one query image to the network and get the feature embedding of the query image. Compare the feature embedding of the query image to 
    all the feature embeddings in the whole training dataset. Rank the result and output the top k results.
    """
    results = []
    pq = []
    f_q = network(query_image).numpy()
    for index, data in enumerate(train_loader):
        compare_image = data['Query'].to(decive)
        f_c = network(compare_image).numpy()
        distance = np.linalg.norm(f_q-f_c, ord=2)
        heappush(pq, (distance, compare_image))
    for i in range(k):
        results.append(heappop(pq))
    return results

def Calc_Precision(query_class, ranked_images, k):
    """
    We calculated the categorical level similarity. The precision is calculated by number of correctly ranked images divided by k
    Parameters:
        -query_class: the class which the query image belongs to
        -ranked_images: list, the classes which all ranked images belong to
        -k: scalar, the number of ranked images
    Return:
        -precision: scalar
    """
    all_images = os.listdir('./train/'+query_class+'/images')
    correct = 0.0
    for idx, img in enumerate(ranked_images):
        if (img in all_images): correct += 1
    precision = correct/k
    return precision

def Plot_training_loss(loss):
    """
    We need to keep record of the training loss throughout the whole training process.
    Parameters:
        -loss: list, the training loss of each training epoch
    """
    plt.plot(range(EPOCH), loss)
    plt.show()



if __name__ == "__main__":
    # -----------------------------------------
    #             Load dataset
    # -----------------------------------------
    #sampler = Triplets_PreSampler()
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
    #train_dataset = Triplets_Dataset(sampler, transform)
    BATCH_SIZE = 32
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 0.001
    g = 1
    EPOCH = 20
    D = 4096
    # network = ResNet(D).to(device)
    network = torchvision.models.resnet101(pretrained=True)

    # child_counter = 1
    # for child in network.children():
    #     if (child_counter < 8):
    #         for param in child.parameters():
    #             param.requires_grad = False
    #     child_counter += 1

    # for param in network.parameters():
    #     param.requires_grad = False
    network.fc = nn.Linear(network.fc.in_features, D, bias=True) ## Modify the last fully-connected layer's output dimension to be 4096
    network = network.to(device)
    #print (next(network.parameters()).is_cuda)
    #print ("device: ", device)
    optimizer = torch.optim.SGD(network.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
    running_loss = 0.0
    training_loss = []
    loss_fn = nn.TripletMarginLoss(margin=g, p=2)

    for epoch in range(EPOCH):
        epoch_loss = 0.0
        sampler = Triplets_PreSampler()
        train_dataset = Triplets_Dataset(sampler, transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        start = time()
        for index, data in enumerate(train_loader):
            # print ("load batch sample cost: ", time()-stamp)
            optimizer.zero_grad()
            query_image = data["Query"].to(device)
            positive_image = data["Positive"].to(device)
            negative_image = data["Negative"].to(device)

            stamp = time()

            f_p = network(query_image)
            f_p_plus = network(positive_image)
            f_p_minus = network(negative_image)

            #print ("forward step cost: ", time()-stamp)
            loss = loss_fn(f_p, f_p_plus, f_p_minus)
            #print ("loss: ", loss)
            loss.backward()

            # print ("calculate loss cost: ", time()-stamp)
            # if(epoch>3):
            #     for group in optimizer.param_groups:
            #         for p in group['params']:
            #             state = optimizer.state[p]
            #             if(state['step']>=1024):
            #                 state['step'] = 1000

            optimizer.step()
            # print ("optimization cost: ", time()-stamp)


            running_loss += loss.item()
            epoch_loss += loss.item()
            #print (type(running_loss))
            if (index%100==99):
                print("Epoch %d, %d mini-batches, Loss: %.3f, time consumiong: %.2f seconds"%(epoch+1, index+1, running_loss/100., time()-start))
                running_loss = 0.

        avg_epoch_loss = epoch_loss/(index+1)
        # training_loss.append(avg_epoch_loss)
        print ("Epoch %d finished, average loss is: %.5f, cost: %.2f seconds"%(epoch+1, avg_epoch_loss, time()-start))
        torch.save(network.state_dict(), 'temp_model_trained_%d_epochs.ckpt'%(epoch+1))


    print ("Training Completed!")
    torch.save(network, 'Deep_Ranking_model.ckpt')














    
