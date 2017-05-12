import pandas as pd
import sys
import re
import numpy as np

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    return origin_data[:,0]

def split(content,split_string):
    temp = []
    for a in content:
        for b in a.split(split_string):
            if b != "" and b!= " ":
                temp.append(b)
    content=temp
    return content

def saveFile(filename,content):
    ofile=open(filename,'w',encoding='utf-8')
    for i in range(len(content)):
        ofile.write(str(content[i])+'\n')
    ofile.close()

def clean_corpus(corpus):
    clean_space = re.compile('[\n\/,\.\?()_]')
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|\\+begin.*?\\+end|[^a-zA-Z\' ]')
    corp = []
    for sentence in corpus:
        word = sentence.split(' ')
        words = [w for w in word if '@' not in w and 'http' not in w and '&' not in w]
        word = []
        for w in words:
            if ('\\' not in w):
                word.append(w)
            elif ((w=='\\begin') or (w=='\\end') or (w=='\\\\begin') or (w=='\\\\end')):
                word.append(w)
            elif ('$' in w):
                word.append(w)
            elif (w=='\\x08'):
                word.append('\\begin')
        corp.append(" ".join(word))
    corpus = corp
    corpus = [clean_space.sub(' ', sentence) for sentence in corpus]
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]
    return corpus

if __name__=='__main__':
    origin_data = readFromData(sys.argv[1])
    origin_data = split(origin_data,"__eou__")
    origin_data = split(origin_data,"__eot__")
    org = origin_data
    origin_data = split(origin_data,"? ")
    origin_data = split(origin_data,". ")
    origin_data = split(origin_data,"! ")
    corpus = []
    for sentence in origin_data:
        word = sentence.split(' ')
        word = [w for w in word if len(w)<13]
        words = []
        for w in word:
            if w.isupper() and w != "I": #whole chars are capital, then don't save it
                pass    
            else:
                temp = w.replace("n't"," not")
                temp = temp.replace("'ve"," have")
                temp = temp.replace("$","$ ")
                temp = temp.replace("@","@ ")
                temp = temp.replace("#","# ")
                temp = temp.replace("&","& ")
                temp = temp.replace("*","* ")
                temp = temp.replace("%","% ")
                temp = temp.replace("~","~ ")
                temp = temp.replace("<","< ")
                temp = temp.replace(">","> ")
                temp = temp.lower()
                words.append(temp)
        corpus.append(" ".join(words))
    origin_data = corpus
    clean_empty = re.compile("[-?!.;^+_,':\/\"\[\]\{\}\(\)]")
    origin_data = [clean_empty.sub(' ',sentence) for sentence in origin_data]
    corpus = []
    for sentence in origin_data:
        sentence = re.sub( '\s+',' ',sentence ).strip()
        if sentence != "":
            corpus.append(sentence)
    origin_data = corpus
    saveFile(sys.argv[3],origin_data)
