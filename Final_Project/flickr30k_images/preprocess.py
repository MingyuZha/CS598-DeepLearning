import numpy as np
import os
import nltk
import itertools
import io
nltk.download('punkt')
count = 0
x_train = []
filename = './flickr30k_images/results_20130124.token' 
# https://github.com/yunjey/seq2seq-dataloader/blob/master/build_vocab.py
with io.open(filename,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')[1]
        line = line.replace('<br />',' ')
        line = line.replace('\x96',' ')
        line = nltk.word_tokenize(line)
        line = [w.lower() for w in line]
        x_train.append(line)
        count += 1
no_of_tokens = []
for tokens in x_train:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
#print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
total_count = len(word_to_id)

# word_to_id['<pad>'] = total_count
# total_count += 1

# word_to_id['<start>'] = total_count
# total_count += 1

# word_to_id['<end>'] = total_count
# total_count += 1

# word_to_id['<unknown>'] = total_count
# total_count += 1




all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
# id_to_word.append('<pad>')
# id_to_word.append('<start>')
# id_to_word.append('<end>')
# id_to_word.append('<unknown>')
id_to_word = np.asarray(id_to_word)

## let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]
#print(np.sum(count[0:3000]))
#vocab size = 3000
hist = np.histogram(count,bins=[1,10,100,1000,10000])
#print(hist)
for i in range(10):
    print(id_to_word[i],count[i])
## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[0 if token not in word_to_id else word_to_id[token] + 4 for token in x] for x in x_train]
id_to_word = np.asarray(['<unknown>', '<start>','<end>', '<pad>'] + list(id_to_word))
print(id_to_word[4],id_to_word[5],id_to_word[6],id_to_word[7])
np.save('./word_to_id.npy',np.asarray(id_to_word))

## save training data to single text file
with io.open('./preprocessed_flickr30K.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            print(id_to_word[token])
            f.write("%i " % token)
        f.write("\n")