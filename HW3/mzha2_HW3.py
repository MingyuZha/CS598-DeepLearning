import torch
import torchvision
import torchvision.transforms as transforms
# from torch.autograd import Variable
import numpy as np
from time import *




class Deep_Conv_NN(torch.nn.Module):
    def __init__(self):
        super(Deep_Conv_NN, self).__init__()

        #Define the model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2),  #Convolution layer 1
            torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU()                                         #(33,33)
            
            )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=2),  #Convolution layer 2
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)         #(17,17)
            # torch.nn.Dropout()
            )                 

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2),  #Convolution layer 3
            torch.nn.BatchNorm2d(num_features=128),                 #Shape: (18,18))
            torch.nn.ReLU()
            
            )
            
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),                          #Convolution layer 4
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),           #Shape: (9,9)
            # torch.nn.Dropout()
            )
            
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=2),                          #Convolution layer 5
            torch.nn.BatchNorm2d(num_features=256),                  #Shape: (10,10)
            torch.nn.ReLU()
            
            )
            
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),                          #Convolution layer 6
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU(),
            # torch.nn.Dropout()                             #Shape: (8,8)
            )
            
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),                          
            torch.nn.BatchNorm2d(num_features=512),                  #Shape: (6,6)
            torch.nn.ReLU()
            
            )
            
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),                          #Convolution layer 8
            torch.nn.BatchNorm2d(num_features=512),                  #Shape: (4,4)
            torch.nn.ReLU(),
            
            # torch.nn.Dropout()
            )
            
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(512*4*4, 500),
            torch.nn.BatchNorm1d(num_features=500),
            torch.nn.ReLU(),
            torch.nn.Dropout()
            )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(500, 500),
            torch.nn.BatchNorm1d(num_features=500),
            torch.nn.ReLU(),
            torch.nn.Dropout()
            )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(500, 10),
            torch.nn.BatchNorm1d(num_features=10),
            torch.nn.Softmax(dim=1)
            )
            
            

    #Forward pass
    def forward(self, x):
        x = self.conv1(x)
        # print (x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1, 512*4*4)
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.out(x)
        return x




    #Calculate accuracy



def train(model, dataset, LR, num_epochs):
    """
    parameters:
    model: object, defined neural network model
    dataset: torch.dataloader
    LR: scalar, learning rate
    num_epochs: scalar, the training iterations you need 
    """
    #Define the Loss function
    start_time = time()
    loss_fn = torch.nn.CrossEntropyLoss()
    test_set = test_loader
    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # #Pre-parameter option
    # optimizer = torch.optim.Adam([{'params':model.conv1.parameters(), 'lr': 0.01},
    #                             {'params':model.conv2.parameters(), 'lr': 0.01},
    #                             {'params':model.conv3.parameters(), 'lr': 0.01},
    #                             {'params':model.conv4.parameters(), 'lr': 0.01},
    #                             {'params':model.conv5.parameters(), 'lr': 0.01},
    #                             {'params':model.conv6.parameters()},
    #                             {'params':model.conv7.parameters()},
    #                             {'params':model.conv8.parameters()},
    #                             {'params':model.fc1.parameters()},
    #                             {'params':model.fc2.parameters()},
    #                             {'params':model.out.parameters()}], lr = LR)
    #Set different learning rate for different steps
    # schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30,100], gamma = 0.1)
    for epoch in range(num_epochs):

        # schedular.step()
        running_loss = 0
        
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            inputs, labels = inputs.to(device), labels.to(device)

            #zero the parameter gradients
            
            optimizer.zero_grad()

            #Forward+backward+optimize
            f = model(inputs)
            loss = loss_fn(f, labels)

            

            loss.backward()

            if(epoch>10):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if(state['step']>=1024):
                            state['step'] = 1000

            optimizer.step()

            #print statistics
            running_loss += loss.item()
            if (i%100 == 99):   #print every 2000 mini-batches

                print ("Epoch: ", epoch+1, ", ", i+1, " mini-batches, loss: %.3f"%(running_loss/100.), ", Using %.3f s"%(time()-start_time))
                running_loss = 0.
            # if (i%100 == 99):
            #     acc = calc_acc(model, test_set)
            #     print ("accuracy: %.2f "%acc)
        acc = calc_acc(model, test_set)
        print ("Echo %i finished, Test accuracy is: %.3f"%(epoch+1, acc))
    print ('Finished Training!')
    return model

def Load_Data(batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)
    return train_loader, test_loader

def calc_acc(model, test_dataset):
    correct = 0.
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100*correct/total

if __name__ == "__main__":
    #Load Dataset with data augmentation
    # transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomRotation(10), transforms.RandomCrop(size=32,padding=4)], p=0.6), 
    #                                 transforms.RandomHorizontalFlip(p=.5), 
    #                                 transforms.RandomVerticalFlip(p=0.3), 
    #                                 transforms.ToTensor(), 
    #                                 transforms.Normalize((.5,.5,.5), (.5,.5,.5))]
    #                                 )
    transform = transforms.Compose([transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    BATCH_SIZE = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # dataset, test = Load_Data(batch_size=4)


    
    # device = torch.device("cuda" if use_cuda else "cpu")
    # print ("Currently using: ", device)

    # model = Deep_Conv_NN().cuda()   
    device = torch.device("cuda")
    # model = Deep_Conv_NN() 
    model = Deep_Conv_NN().to(device)

    # device = torch.device("cuda")
    # model = model.to(device)

    model = train(model, train_loader, 0.0001, 200)
    print("Final model accuracy is: %.5f"%(calc_acc(model, test_loader)/100.))

    # f = model.forward(inputs)
    # print (f)



    


