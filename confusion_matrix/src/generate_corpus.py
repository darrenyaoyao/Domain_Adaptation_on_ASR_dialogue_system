from collections import Counter
from copy import deepcopy
import pandas as pd
import numpy as np
import pickle
#import enchant

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    return origin_data

def saveFile(outfileName, content, utterence, label):
    text = np.asarray([content, utterence, label])
    text = np.transpose(text)
    df = pd.DataFrame(text,columns=['Context','Utterence','Label'])
    df.to_csv(outfileName,index=False)
    return True

def del_oov(container):
    #this function can delete the word out of voca.
    del container['_']  # delete "_"
    d = enchant.Dict("en_US") # english dictionary
    copy = deepcopy(container)
    
    for element in copy:
        if (not d.check(element)) or (element.isdigit()):
            del container[element]
    del copy # release memory

def choose_topK(container,K):
    '''
    this function choose the top K condidate of the substituent
    container is a counter count in counter
    K is the top K number
    '''
    #for counter in container.values():
    
    for element in container:
        counter = container[element]
        temp = counter.most_common(K) # temp is a list e.g [(a,4),(b,3)]
        newCounter = Counter(dict(temp))
        container[element]=newCounter
        del counter

def trans_to_prob(container):
    '''
    this function will transfer the data in container from
    statistic info to prob. info.
    '''
    for counter in container.values():
        temp = 0
        summation = sum(counter.values())
        for ele in counter:
            counter[ele] = float(counter[ele])/float(summation)

def trans_prob_to_list(container):
    '''
    this function will construct another container with list
    In order to sample using numpy
    '''
    new_container = Counter()
    for element in container:
        counter = container[element]
        l_counter = np.empty([2,0])
        for key, value in counter.items():
            temp = np.array([key,value]).reshape(2,1)
            l_counter = np.append(l_counter, temp, axis=1)
        new_container[element]=l_counter
    return new_container

def random_pick(l_counter,num_sample=1):
    '''
    input counter (l_counter) is expected an 2d-nparray,
    the first dim is the key, and the second dim is the value
    '''
    return np.random.choice(l_counter[0],num_sample,p=l_counter[1].astype(float))

def transform(content,l_container,container):
    for i,sentence in enumerate(content):
        temp = [ ]
        for words in sentence.split(" "):
            if container[words] !=0:
                subst = random_pick(l_container[words])
                if subst[0] != "_":
                    temp.append(subst[0])
            else:
                temp.append(words)
        content[i]=" ".join(temp)

if __name__=="__main__":
    rfile = open(r'container.p','rb')
    container = pickle.load(rfile)
    rfile.close()
    #del_oov(container)
    choose_topK(container,20)
    trans_to_prob(container)
    list_container = trans_prob_to_list(container)
    original_data = readFromData('train_org.csv')
    content = original_data[:,0]
    utter = original_data[:,1]
    label = original_data[:,2]
    transform(content, list_container, container)
    saveFile("train_top20_dicOOV_3.csv",content,utter,label)

