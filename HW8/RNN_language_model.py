import torch.nn as nn
import torch
from torch.autograd import Variable

class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()
        
        self.lstm = nn.LSTMCell(in_size,out_size)
        self.out_size = out_size
        
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self,x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


class RNN_language_model(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units):
        super(RNN_language_model, self).__init__()

        
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.lstm3 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm3= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout3 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.decoder = nn.Linear(no_of_hidden_units, vocab_size)

        self.loss = nn.CrossEntropyLoss()#ignore_index=0)

        self.vocab_size = vocab_size

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        self.lstm2.reset_state()
        self.dropout2.reset_state()
        self.lstm3.reset_state()
        self.dropout3.reset_state()

    def forward(self, x, train=True):


        embed = self.embedding(x) # batch_size, time_steps, features

        no_of_timesteps = embed.shape[1]-1

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.3,train=train)

            h = self.lstm2(h)
            h = self.bn_lstm2(h)
            h = self.dropout2(h,dropout=0.3,train=train)

            h = self.lstm3(h)
            h = self.bn_lstm3(h)
            h = self.dropout3(h,dropout=0.3,train=train)

            h = self.decoder(h)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,vocab_size)
        target_prediction = outputs.permute(1,0,2) # batch, time, vocab
        outputs = outputs.permute(1,2,0) # (batch_size,vocab_size,time_steps)

        if(train==True):

            target_prediction = target_prediction.contiguous().view(-1,self.vocab_size)
            target = x[:,1:].contiguous().view(-1)
            loss = self.loss(target_prediction,target)

            return loss, outputs
        else:
            return outputs



