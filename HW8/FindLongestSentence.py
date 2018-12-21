import numpy as np
import os

def FindLongestSentence(root):
    file_path = os.path.join(root, '/preprocessed_flickr30K.txt')
    if (not os.path.isfile(file_path)):
        print ("Check your file path, there's no existed file")
        return
    max_len = 0
    with open(file_path, 'r') as file:
        line = file.readline().strip('\n\r').split()
        max_len = max(max_len, len(line))
    return max_len