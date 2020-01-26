import scipy.io as sio
import numpy as np
import json
import heapq

def generate_axis(wordcnt_path, n, word_tag=None):
    with open(wordcnt_path) as f:
       wordcnt = f.readlines()
    indices = []
    values = []
    for i in range(len(wordcnt)):
        line =wordcnt[i].split('\n')[0]
        tokens = line.split(' ')
        if(word_tag):
            if(tokens[2] in word_tag):
                indices.append(i)
                values.append(float(tokens[1]))
        else:
            indices.append(i)
            values.append(float(tokens[1]))
    if(n > len(indices)):
        exit(0)

    indices = np.array(indices).reshape(-1, )
    print(indices.shape)
    values = np.array(values).reshape(-1, )
    n_largest = heapq.nlargest(n, range(len(indices)), values.take)
    return indices[n_largest]
    

if(__name__ == '__main__'):
    res = generate_axis('/data/zj/local_pyproject/TF-IDF/TF_IDF_TAG.txt', 1000)
    print(res)
    



