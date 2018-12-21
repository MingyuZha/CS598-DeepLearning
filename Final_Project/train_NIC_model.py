from CNN.cnn import CNN_model
from LSTM.rnn import RNN_model
import numpy as np
import torch
import nltk
import torch.nn as nn
import os.path
from flickrloader import Flickr30K, Flickr8K
import random
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.models as models
import os
import os.path
import nltk
from mscocoloader import MSCOCO
from PIL import Image
nltk.download('punkt')
# https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
# 0 for unknown
# 1 for start
# 2 for end
# 3 for pad

dataset_trained = 'MSCOCO' # Choose your dataset between flickr30K, flickr8K and MSCOCO

model_save_name = 'MSCOCO_resnet2000'

model_use = 'resnet' # Choose from inception, resnet, densenet

CNN_pretrained = True #True for not training all, False for train all

embedding_feature = 2000

def tokenize_sentence(line):
    line = line.replace('\n','')
    line = line.split('\t')[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    assert '\n' not in line
    return line

#customizing loss function to filter padding tag
# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e for customizing loss function
def my_loss(y_prob, y):
    y = y[:,1:]
    y_prob = F.log_softmax(y_prob, dim=2)
    y = y.contiguous()
    y = y.view(-1)
    y_prob = y_prob.view(-1, vocab_size)
    mask = ((y != 3) * (y != 1)).float()
    
    count = int(torch.sum(mask).data.item())
    total = mask * y_prob[range(y.shape[0]), y]
    return -torch.sum(total) / count

def my_accurate_count(y_prob, y):
    y = y[:,1:]
    y_prob = F.log_softmax(y_prob, dim=2)
    y = y.contiguous()
    y = y.view(-1)
    y_prob = y_prob.view(-1, vocab_size)
    maxidx = y_prob.max(1)[1]

    mask = ((y != 3) * (y != 1)).float()
    count = int(torch.sum(mask).data.item())
    right = (y == maxidx).float() * mask
    right_count = int(torch.sum(right).data.item())
    return count, right_count



#reference https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
# and # https://zhuanlan.zhihu.com/p/30385675 for customizing collate_fn for padding different length of the caption token sequence
def my_collate(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    image, caption = zip(*batch)
    
    max_len = len(caption[0])
    
    caption_new = []
    sentence_length = [] #prepared for the packed_pad_sequence
    for i in range(len(batch)):
        diff = max_len - len(caption[i])
        caption_curr = caption[i]
        sentence_length.append(len(caption[i]) + 1)
        caption_curr = [1] + [actual_dic[token] if token in actual_dic and actual_dic[token] < vocab_size else 0 for token in caption[i]] + [2] + diff * [3]
        assert len(caption_curr) == max_len + 2
        caption_new.append(caption_curr)
    caption_new = torch.LongTensor(caption_new)
    image = torch.stack(image, 0)
    return image, caption_new, sentence_length

imdb_dictionary = np.load('./word_to_id.npy')
actual_dic = {}
for i in range(imdb_dictionary.shape[0]):
    actual_dic[imdb_dictionary[i]] = i
vocab_size = 8000
#8000 over 5

num_epochs = 200
num_classes = 10
batch_size = 128
num_of_hidden_units = 500
num_of_layer = 2
root = "."
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if model_use == 'inception':
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(350),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# reference https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# reference https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978
if dataset_trained == 'flickr30K':
    train_data = Flickr30K(train=True, transform=data_transform, target_transform=None)
    test_data = Flickr30K(train=False, transform=val_data_transform, target_transform=None)
elif dataset_trained == 'MSCOCO':
    train_data = MSCOCO(root = './MSCOCO', train=True, transform=data_transform, target_transform=None)
    test_data = MSCOCO(root = './MSCOCO', train=False, transform=val_data_transform, target_transform=None)
else:
    train_data = Flickr8K(train=True, transform=data_transform, target_transform=None)
    test_data = Flickr8K(train=False, transform=val_data_transform, target_transform=None)
    
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True,num_workers=32, collate_fn=my_collate)

test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,num_workers=32, collate_fn=my_collate)




criterion = nn.CrossEntropyLoss()
cnn_model = CNN_model(embedding_feature, CNN_pretrained, model_use=model_use)

# Trained the last layer of CNN and all of the LSTM
param_needs_train = []
if CNN_pretrained:
    for param in cnn_model.parameters():
        param.requires_grad_(False)
    #param_needs_train.append(param)
    if model_use != 'densenet':
        for param in cnn_model.model.fc.parameters():
            param.requires_grad_(True)
            param_needs_train.append(param)
    else:
        for param in cnn_model.model.classifier.parameters():
            param.requires_grad_(True)
            param_needs_train.append(param)
else:
    for param in cnn_model.parameters():
        param.requires_grad_(True)
        param_needs_train.append(param)


#TODO change the parameter for RNN_model
lstm_model = RNN_model(vocab_size = vocab_size, no_of_hidden_units=num_of_hidden_units, num_of_layers=num_of_layer, embedding_feature=embedding_feature)
for param in lstm_model.parameters():
    param.requires_grad_(True)
    param_needs_train.append(param)
optimizer = optim.Adam(param_needs_train, lr = 1e-3)

if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()
    lstm_model = lstm_model.cuda()


def evalNIC():
    cnn_model.eval()
    lstm_model.eval()
    loss_total = 0.0
    count = 0.0
    for idx, (image, caption, sentence_length) in enumerate(tqdm(test_loader), 0):
        if torch.cuda.is_available():
            image = image.cuda()
            caption = caption.cuda()
        with torch.no_grad():
            image_feature = cnn_model(image)
            predicted_caption_prob = lstm_model(image_feature, caption, sentence_length)
        loss = my_loss(predicted_caption_prob, caption)
        count += 1
        loss_total += loss
    return loss_total / count




def trainNIC():
    count = 0
    running_loss = 0.0
    loss_chart = [] 
    eval_loss_chart = []
    if os.path.isfile(model_save_name + 'params1.pth'):
        if os.path.isfile(model_save_name +'loss.npy'):
            loss_chart = list(np.load(model_save_name + 'loss.npy'))
            if os.path.isfile(model_save_name + 'losscount.npy'):
                running_loss = list(np.load(model_save_name + 'losscount.npy'))[1].clone()
                count = int(list(np.load(model_save_name + 'losscount.npy'))[0])
        if os.path.isfile(model_save_name +'eval_loss.npy'):
            eval_loss_chart = list(np.load(model_save_name + 'eval_loss.npy'))
        if torch.cuda.is_available():
            cnn_model.load_state_dict(torch.load(model_save_name + 'params1.pth'))
            lstm_model.load_state_dict(torch.load(model_save_name + 'param2.pth'))
        else:
            cnn_model.load_state_dict(torch.load(model_save_name + 'params1.pth',map_location='cpu'))
            lstm_model.load_state_dict(torch.load(model_save_name + 'param2.pth',map_location='cpu'))
    
    for epoch in range(num_epochs):
        if len(loss_chart) > 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        if len(loss_chart) > 250:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        if len(loss_chart) > 330:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
        if len(loss_chart) > 420:
            return
        cnn_model.train()
        lstm_model.train()
        for idx, (image, caption, sentence_length) in enumerate(tqdm(train_loader), 0):
            if torch.cuda.is_available():
                image = image.cuda()
                caption = caption.cuda()        
            optimizer.zero_grad()
            count+= 1        
            image_feature = cnn_model(image)
            predicted_caption_prob = lstm_model(image_feature, caption, sentence_length)
            loss = my_loss(predicted_caption_prob, caption)
            running_loss += loss
            if count == 500:
                print(running_loss / 500)
                loss_chart.append(running_loss / 500)
                running_loss = 0.0
                count = 0
            loss.backward()
            optimizer.step()
        # TODO place holder for forward pass, may need to fine tune or fix some bugs or typos
        torch.save(cnn_model.state_dict(), model_save_name + 'params1.pth')  
        torch.save(lstm_model.state_dict(), model_save_name  + 'param2.pth')
        np.save(model_save_name + 'loss.npy', np.asarray(loss_chart))
        np.save(model_save_name + 'losscount.npy', np.asarray([count, running_loss]))
        if (epoch % 5 == 0):
            eval_loss_chart.append(evalNIC())
            np.save(model_save_name + 'eval_loss.npy', np.asarray(eval_loss_chart))

trainNIC()







        
            
        
        













