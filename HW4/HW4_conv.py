import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from time import *
from copy import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.weight_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
            )
        self.short_cut = nn.Sequential()
        if (stride != 1 or in_channel != out_channel):
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
                nn.BatchNorm2d(out_channel)
                )
    def forward(self, x):
        out = self.weight_layers(x)
        out += self.short_cut(x)
        out = nn.functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        #define basic block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )

        self.blocks1 = self.make_layer(block=ResidualBlock, in_channel=32, out_channel=32, num_basic_blocks=2, stride=1)
        self.blocks2 = self.make_layer(ResidualBlock, 32, 64, 4, 2)
        self.blocks3 = self.make_layer(ResidualBlock, 64, 128, 4, 2)
        self.blocks4 = self.make_layer(ResidualBlock, 128, 256, 2, 2)

        self.MaxPool = nn.MaxPool2d(3,1)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(256*2*2, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout()
        #     )
        self.fc2 = nn.Sequential(
            nn.Linear(256*2*2, 100),
            # nn.BatchNorm1d(100), 
            # nn.Softmax(dim=1)
            )
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.MaxPool(x)
        x = x.view(-1, 256*2*2)
        # x = self.fc1(x)
        x = self.fc2(x)
        return x


    def make_layer(self, block, in_channel, out_channel, num_basic_blocks, stride):
        layers = []
        for i in range(num_basic_blocks):
            if (i == 0): 
                s = stride
                exact_inchannel = in_channel
            else: 
                s = 1
                exact_inchannel = out_channel
            layers.append(block(exact_inchannel, out_channel, s))
        return nn.Sequential(*layers)



def CalcAccuracy(model):
    correct = 0.
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100*correct/total

if __name__ == "__main__":
    #Load Dataset
    transform_train = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    BATCH_SIZE = 200
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Intialize the model
    model = ResNet().to(device)

    #Hyper-parameters setting
    EPOCHS = 50
    LR = 0.001

    #Loss function
    loss_fn = nn.CrossEntropyLoss()
    #Opitimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,100], 0.1)
    #Training
    start = time()
    for i in range(EPOCHS):
        running_loss = 0
        # scheduler.step()
        for index, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            f = model(data)
            loss = loss_fn(f, labels)
            # print ("Loss: %.3f"%loss)
            loss.backward()

            if(epoch>15):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if(state['step']>=1024):
                            state['step'] = 1000

            optimizer.step()

            running_loss += loss.item()

            if (index%100 == 99):
                print ("Epoch: %d, %d minibatches, Loss: %.4f, Time usage: %.2f seconds"%(i+1, index+1, running_loss/(100.0), time()-start))
                running_loss = 0.
        #Calculate the test accuracy
        test_acc = CalcAccuracy(model)
        print ("Epoch %d Finished, Current Test Accuracy: %.2f"%(i+1, test_acc)+"%")

    print ("Finished training!")

    #Calculate the Final Test Accuracy
    final_acc = CalcAccuracy(model)
    print("Final model accuracy is: %.3f"%(final_acc)+"%")



