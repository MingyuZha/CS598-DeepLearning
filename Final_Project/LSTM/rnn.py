import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import time
import os
import sys
import io
# https://zhuanlan.zhihu.com/p/34418001 and https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099
# reference https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.htmll and HW7 on defining RNN
class RNN_model(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units, num_of_layers = 1, dropout = 0.5, embedding_feature = 1000):
        super(RNN_model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_feature) #embedding to image feature number

        self.lstm = nn.LSTM(input_size = embedding_feature, hidden_size = no_of_hidden_units, num_layers = num_of_layers, dropout = dropout, batch_first=True)

        self.decoder = nn.Linear(no_of_hidden_units, vocab_size)

        self.vocab_size = vocab_size

    def forward(self, image_feature, caption, sentence_length, train=True):
        embed = self.embedding(caption) 
        # embed -- batch * caption_length * emdedding_feature(same with the size of image_feature)
        # n * length of longest sequence in this batch * 1000
        

        
        # https://zhuanlan.zhihu.com/p/34418001 for padding
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e for customizing loss function
        

        image_feature = image_feature.reshape(image_feature.shape[0], 1, image_feature.shape[1])
        # image -  n * 1000 -> n * 1 * 1000 as the input at t[0]

        lstm_output_first, hidden_after_image = self.lstm(image_feature,None)  
        #feed image to the lstm first to get the hidden state for next input    
        
        # lstm_output_first is P[0]
        #After feeding the image, feed the hidden state with the padded embedding caption
            
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embed, sentence_length, batch_first=True)
        encoder_outputs_packed, _ = self.lstm(lstm_input, hidden_after_image)
        
        # output here is from P[1] to P[N] which is exactly we need to compute the loss
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        # (n - 1) * sentence_length * embedding
        outputs = self.decoder(lstm_output)
        # (n - 1) * sentence_length * vocab_size
        return outputs
    




        

