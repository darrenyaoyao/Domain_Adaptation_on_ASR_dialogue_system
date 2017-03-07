import pandas as pd
import sys
import pickle
import numpy as np

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    return origin_data[:,0]

def minEditDist(target, source):
    ''' Return a pair of aligned target and source'''
    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m+1)] for j in range(n+1)]

    for i in range(1,n+1):
        #distance[i][0] = distance[i-1][0] + insertCost(target[i-1])
        distance[i][0] = distance[i-1][0] + 1

    for j in range(1,m+1):
        #distance[0][j] = distance[0][j-1] + deleteCost(source[j-1])
        distance[0][j] = distance[0][j-1] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            distance[i][j] = min(distance[i-1][j-1]+substCostSen(source[j-1],target[i-1]),
                                 distance[i-1][j]+1,
                                 distance[i][j-1]+1)
    ii = n
    jj = m
    if type(target)==str:
        target_aln = ""
        source_aln = ""
        while (ii > 0) or (jj > 0):
            if distance[ii][jj]-substCostSen(source[jj-1],target[ii-1]) == distance[ii-1][jj-1]:
                target_aln += target[ii-1]
                source_aln += source[jj-1]
                ii -= 1
                jj -= 1
            elif distance[ii][jj] - 1 == distance[ii-1][jj]:
                source_aln += "_"
                target_aln += target[ii-1]
                ii -= 1
            elif distance[ii][jj] - 1 == distance[ii][jj-1]:
                source_aln += source[jj-1]
                target_aln += "_"
                jj -= 1
            else:
                print ("error!")
    else:
        target_aln = []
        source_aln = []
        while (ii > 0) or (jj > 0):
            if distance[ii][jj]-substCostSen(source[jj-1],target[ii-1]) == distance[ii-1][jj-1]:
                target_aln.append(target[ii-1])
                source_aln.append(source[jj-1])
                ii -= 1
                jj -= 1
            elif distance[ii][jj] - 1 == distance[ii][jj-1]:
                source_aln.append(source[jj-1])
                target_aln.append("_")
                jj -= 1
            elif distance[ii][jj] - 1 == distance[ii-1][jj]:
                source_aln.append("_")
                target_aln.append(target[ii-1])
                ii -= 1
            else:
                print ("error!")
     
    target_aln = target_aln[::-1]
    source_aln = source_aln[::-1]
    return (target_aln,source_aln)

def substCost(x,y):
    if x == y: 
        return 0
    else: 
        return 2

def substCostSen(x,y):
    if x == y: 
        return 0
    elif minEditDistStr(x,y) < 3:
        return 1.5
    else:
        return 2

def minEditDistStr(target, source):
    ''' Computes the min edit distance from target to source.'''
    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m+1)] for j in range(n+1)]

    for i in range(1,n+1):
        #distance[i][0] = distance[i-1][0] + insertCost(target[i-1])
        distance[i][0] = distance[i-1][0] + 1

    for j in range(1,m+1):
        #distance[0][j] = distance[0][j-1] + deleteCost(source[j-1])
        distance[0][j] = distance[0][j-1] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            distance[i][j] = min(distance[i-1][j-1]+substCost(source[j-1],target[i-1]),
                                 distance[i-1][j]+1,
                                 distance[i][j-1]+1)
    return distance[n][m]
 
#recurrance approach
def minEditDistR(target, source):
    """ Minimum edit distance. Straight from the recurrence. """
    i = len(target)
    j = len(source)

    if i == 0:  
        return j
    elif j == 0: 
        return i
    
    return(min(minEditDistR(target[:i-1],source)+1, #delete
               minEditDistR(target, source[:j-1])+1, #append
               minEditDistR(target[:i-1], source[:j-1])+substCost(source[j-1], target[i-1]))) #substitution

from collections import Counter

def statistic(container, align_tar, align_source):
    """
    container is a Counter, key is the source string, value is another container A
    A is another Counter, key is the target string, value is the statistic number
    e.g. word "are" is transfered to word "ar", than container["are"]["ar"]+=1
    
    type of align_tar and align_sour is assumed as list
    e.g. ['How','are','you']
    length of align_tar and align_sour should be the same, the function would do
    the counting action of them.
    """
    for i in range(len(align_tar)):
        if container[align_tar[i]]==0:
            temp = Counter()
            temp[align_source[i]]+=1
            container[align_tar[i]]=temp
        else:
            container[align_tar[i]][align_source[i]]+=1

if __name__=="__main__":
    container = Counter()
    org = readFromData("train_50000.csv")
    asr = readFromData("train_50000_asr.csv")
    for i in range(len(org)):
        target,source = minEditDist(org[i].split(" "),asr[i].split(" "))
        statistic(container,target,source)

    print (container)
    #save container
    ofile = open(r'container.p','wb')
    pickle.dump(container,ofile)
    ofile.close()
    '''
    How to read?
    rfile = open(r'container.p','rb')
    container = pickle.load(rfile)
    rfile.close()
    '''
