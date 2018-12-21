from CNN.cnn import CNN_model
from LSTM.rnn import RNN_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import nltk
import torch.nn as nn
import os.path
from mscocoloader import MSCOCOFORSCORE
from flickrloader import Flickr30K, Flickr8K, Flickr30KFORSCORE, Flickr8KFORSCORE
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import io
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.models as models
import os
import nltk
from PIL import Image
nltk.download('punkt')

dataset_tested = 'MSCOCO'   # The dataset you want to test on
model_used = 'MSCOCO_resnet'  # The model you want to load.

model_use = 'resnet'
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html on val transform
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




if model_use == 'inception':
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(350),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

sequence_length = 20
vocab_size = 8000 
num_of_hidden_units = 500
num_of_layer = 2
imdb_dictionary = np.load('./word_to_id.npy')

actual_dic = {}
for i in range(imdb_dictionary.shape[0]):
    actual_dic[imdb_dictionary[i]] = i

root = "."
test_data = []
if dataset_tested == 'flickr30K':
    test_data = Flickr30KFORSCORE(transform=data_transform, target_transform=None)
elif dataset_tested == 'MSCOCO':
    test_data = MSCOCOFORSCORE(transform=data_transform, target_transform=None)
else:
    test_data = Flickr8KFORSCORE(transform=data_transform, target_transform=None)

imdb_dictionary = np.load('./word_to_id.npy')
actual_dic = {}
for i in range(imdb_dictionary.shape[0]):
    actual_dic[imdb_dictionary[i]] = i
id_to_word = {}
for i in actual_dic.keys():
    id_to_word[actual_dic[i]] = i

# Generate sentences with sampling(greedy), the idea is same with the paper, from the image features and start symbol. Every time the output of the current time stamp 
# will be used as the input of the next timestamp together with the hidden state
def generate_sentences(image, sequence_length):
    #like language models
    cnn_model.eval()
    lstm_model.eval()
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        image_features = cnn_model(image)
    image_features = image_features.reshape(image_features.shape[0], 1, image_features.shape[1])
    # https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html for sampling
    output_sentence = ""
    hidden = None
    
    
    # start from the start symbol
    output, hidden = lstm_model.lstm(image_features, hidden)
    image_features = torch.Tensor(1).long()
    if torch.cuda.is_available():
        image_features = image_features.cuda()
    image_features = lstm_model.embedding(image_features)
    image_features = image_features.reshape(1, 1, image_features.shape[1])
    with torch.no_grad():
        for i in range(sequence_length):
            output, hidden = lstm_model.lstm(image_features, hidden) #update hidden
            output_prob = lstm_model.decoder(output)
            output_prob = output_prob.view(-1)
            value, idx = output_prob.topk(1) #torch.topk: return the k largest elements of the given input tensor along a given dimension
            if (idx.item() == 2):   # end
                break
            output_sentence += id_to_word[idx.item()] + " "
            image_features = lstm_model.embedding(idx)
            image_features = image_features.reshape(image_features.shape[0], 1, image_features.shape[1])
        return output_sentence.strip().replace('.','')


def generate_sentences_beam_search(image, beam_size, sentence_length):
    cnn_model.eval()
    lstm_model.eval()
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        image_features = cnn_model(image)
        image_features = image_features.reshape(image_features.shape[0], 1, image_features.shape[1])

        hidden = None
        output, hidden = lstm_model.lstm(image_features, hidden)
        image_features = torch.Tensor(1).long()
        if torch.cuda.is_available():
            image_features = image_features.cuda()
        image_features = lstm_model.embedding(image_features)
        image_features = image_features.reshape(1, 1, image_features.shape[1])


    
        all_candidate = [('',image_features, hidden, 0)]
        res = [] 
        for i in range(sentence_length):
            new_candidate = []
            for sentence_pair, input_lstm, hidden_state, probs in all_candidate:
                output, hidden_output = lstm_model.lstm(input_lstm, hidden_state)
                output_prob = lstm_model.decoder(output)
                output_prob = output_prob.view(-1)
                output_prob = F.log_softmax(output_prob, dim=0)
                value, idx = output_prob.topk(beam_size)
                for j in range(beam_size):
                    new_features = lstm_model.embedding(idx[j])
                    if (idx[j].item() == 2):
                        res.append(sentence_pair.strip())
                        if (len(res) == beam_size):
                            return res
                        continue
                    new_candidate.append((sentence_pair + id_to_word[idx[j].item()] + ' ', new_features.reshape(1, 1, new_features.shape[0]), hidden_output,  probs + value[j]))
                new_candidate.sort(key=lambda x: -x[-1])
                all_candidate = new_candidate[0: beam_size]
    return res

cnn_model = CNN_model(1000, model_use = model_use)
lstm_model = RNN_model(vocab_size = vocab_size, no_of_hidden_units=num_of_hidden_units, num_of_layers=num_of_layer, embedding_feature=1000)
if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()
    lstm_model = lstm_model.cuda()
if torch.cuda.is_available():
    cnn_model.load_state_dict(torch.load(model_used + 'params1.pth'))
    lstm_model.load_state_dict(torch.load(model_used + 'param2.pth'))
else:
    cnn_model.load_state_dict(torch.load(model_used + 'params1.pth', map_location='cpu'))
    lstm_model.load_state_dict(torch.load(model_used + 'param2.pth', map_location='cpu'))
list_predicted = []
list1_right = []
list2_right = []
list3_right = []
list4_right = []
list5_right = []
list_beam_search = []
count = 0
filename_list = []
for filename, image, caption_list in test_data:
    if (len(caption_list) != 5):
        continue

    list1_right.append(caption_list[0].lower())
    list2_right.append(caption_list[1].lower())
    list3_right.append(caption_list[2].lower())
    list4_right.append(caption_list[3].lower())
    list5_right.append(caption_list[4].lower())
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    list_predicted.append(generate_sentences(image, sequence_length))
    filename_list.append(filename)
    list_beam_search.append(generate_sentences_beam_search(image, 5, 20))
    count += 1


folder = './Metrics/' + dataset_tested + '/'
# https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python for writing list
with io.open(folder + 'predict.txt', 'w') as f:
    for item in list_predicted:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'ground_truth1.txt', 'w') as f:
    for item in list1_right:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'ground_truth2.txt', 'w') as f:
    for item in list2_right:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'ground_truth3.txt', 'w') as f:
    for item in list3_right:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'ground_truth4.txt', 'w') as f:
    for item in list4_right:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'ground_truth5.txt', 'w') as f:
    for item in list5_right:
        f.write("%s.\n" % item.strip())
with io.open(folder + 'filename.txt', 'w') as f:
    for item in range(len(filename_list)):
        f.write("%s.\n" % (filename_list[item].strip() + "||" + list_beam_search[item][0] + "||" + list_beam_search[item][1] + "||" +list_beam_search[item][2] + "||" + list_beam_search[item][3] + "||" + list_beam_search[item][4] + "||"))





