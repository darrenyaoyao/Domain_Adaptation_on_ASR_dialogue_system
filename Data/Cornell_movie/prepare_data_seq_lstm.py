import random
import pandas as pd
import numpy as np
import copy 

def saveTrainFile(outfileName, trainArray):
    text = np.array(trainArray)
    np.random.shuffle(text)
    df = pd.DataFrame(text,columns=['Context','Utterance','Label'])
    df.to_csv(outfileName,index=False)
    return True

def saveTestFile(outfileName, testArray):
    text = np.array(testArray)
    df = pd.DataFrame(text,columns=['Context','Ground Truth Utterance','Distractor_0',
                                    'Distractor_1','Distractor_2','Distractor_3',
                                    'Distractor_4','Distractor_5','Distractor_6',
                                    'Distractor_7','Distractor_8'])
    df.to_csv(outfileName,index=False)
    return True

''' 
1. Read from 'movie-lines.txt'
2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines=open('original_corpus/movie_lines.txt').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
1. Read from 'movie_conversations.txt'
2. Create a list of [list of line_id's]
'''
def get_conversations(garbageID):
    conv_lines = open('original_corpus/movie_conversations.txt').read().split('\n')
    counter = 0
    convs = [ ]
    movies = [ ]
    movies.append(convs)
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')
        if _line[2] != ("m"+str(counter)):
            counter+=1
            temp = [ ]
            movies.append(temp)
        content = _line[-1][1:-1].replace("'","").replace(" ","")
        id_=content.split(',')
        co = copy.deepcopy(id_)
        for test in co:
            if test in garbageID:
                id_.remove(test)
        del co
        if len(id_) > 1:
            movies[counter].append(id_)
    return movies

'''
1. Get each conversation
2. Separate conversation to context and groundtruth
3. Get each line from conversation 
'''
def conv2contextNans(movies):
    new_movies = [ ] 
    for movieIdx,movie in enumerate(movies):
        new_conv = [ ]
        for conv in movie:
            mult = len(conv)/2
            idx = 0
            while(idx < mult-2):
                pair = [ ]
                pair.append(conv[idx*2:(idx+1)*2+1])
                pair.append(conv[(idx+1)*2+1])
                idx += 1
                new_conv.append(pair)
            pair = [ ]
            pair.append(conv[idx*2:-1])
            pair.append(conv[-1])
            new_conv.append(pair)
        new_movies.append(new_conv)
    return new_movies

'''
1. Get each conversation
2. Get each line from conversation
3. Save each conversation to file
'''
def extract_conversations(movies,id2line,path=''):
    for movieIdx,movie in enumerate(movies):
        idx = 0  
        for conv in movie:
            f_conv = open(path + "movie_"+str(movieIdx)+"-conv_"+str(idx)+'.txt', 'w')
            for line_id in conv:
                f_conv.write(id2line[line_id])
                f_conv.write('\n')
            f_conv.close()
            idx += 1

'''
1. Get each conversation
2. Get each line from conversation
3. Save each conversation to file
'''
def generate_Seq2Seqconv(movies,id2line,path=''):
    # open files
    f_train = open(path + 'train.txt','w')
    f_test = open(path + 'test.txt', 'w')
    f_valid = open(path + 'valid.txt', 'w')
    for movie in movies:    
        idx = 0  
        for conv in movie:
            if idx < 8:
                for line_id in conv[0]:
                    f_train.write(id2line[line_id])
                    f_train.write('    ')
                f_train.write('\t')
                f_train.write(id2line[conv[1]])
                f_train.write('\n')
                idx += 1

            elif idx == 8:
                for line_id in conv[0]:
                    f_valid.write(id2line[line_id])
                    f_valid.write('    ')
                f_valid.write('\t')
                f_valid.write(id2line[conv[1]])
                f_valid.write('\n')
                idx += 1
            else:
                for line_id in conv[0]:
                    f_test.write(id2line[line_id])
                    f_test.write('    ')
                f_test.write('\t')
                f_test.write(id2line[conv[1]])
                f_test.write('\n')
                idx = 0
    
    f_train.close()
    f_test.close()
    f_valid.close()

'''
Get lists of all conversations as Questions and Answers
1. [questions]
2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers

'''
We need 4 files
1. train.enc : Encoder input for training
2. train.dec : Decoder input for training
3. test.enc  : Encoder input for testing
4. test.dec  : Decoder input for testing
'''
def prepare_seq2seq_files(questions, answers, path='',TESTSET_SIZE = 30000):
    # open files
    train_enc = open(path + 'train.enc','w')
    train_dec = open(path + 'train.dec','w')
    test_enc  = open(path + 'test.enc', 'w')
    test_dec  = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i%10000 == 0:
            print '\n>> written %d lines' %(i) 

    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()

def generate_LSTMconv(movies,id2line,path=''):   
    # open files
    trainF = [ ]
    validF = [ ]
    testF  = [ ]
    for i,movie in enumerate(movies):
        print i
        idx = 0  
        for conv in movie:
            if idx < 8:
                #write true label first
                row = [ ]
                string = ""
                for line_id in conv[0]:
                    string+=id2line[line_id]
                    string+=" __eot__ "
                row.append(string)
                row.append(id2line[conv[1]])
                row.append("1.0")
                trainF.append(row)
                #write false lable
                row = [ ]
                row.append(string)
                while(1):
                    randomANS=random.choice(id2line.keys())
                    if randomANS!= conv[1]:
                        break
                row.append(id2line[randomANS])
                row.append("0.0")
                trainF.append(row)
                idx+=1
            elif idx==8 :
                row = [ ]
                string = ""
                for line_id in conv[0]:
                    string+=id2line[line_id]
                    string+=" __eot__ "
                row.append(string)
                row.append(id2line[conv[1]])
                while(1):
                    randomANS=random.sample(id2line.keys(),9)
                    flag = 1
                    for sample in randomANS:
                        if sample == conv[1]:
                            flag = 0
                    if flag ==1:
                        break
                for sample in randomANS:
                    row.append(id2line[sample])
                validF.append(row)
                idx += 1
            else:
                row = [ ]
                string = ""
                for line_id in conv[0]:
                    string+=id2line[line_id]
                    string+=" __eot__ "
                row.append(string)
                row.append(id2line[conv[1]])
                while(1):
                    randomANS=random.sample(id2line.keys(),9)
                    flag = 1
                    for sample in randomANS:
                        if sample == conv[1]:
                            flag = 0
                    if flag ==1:
                        break
                for sample in randomANS:
                    row.append(id2line[sample])
                testF.append(row)
                idx = 0

    saveTrainFile(path+'train.csv', trainF)
    saveTestFile(path+'valid.csv', validF)
    saveTestFile(path+'test.csv', testF)

####
# main()
####

print '>> gathered id2line dictionary.\n'
id2line = get_id2line()
print '>> throw away garbage lines.\n'
garbageline = [ ]
for id_ in id2line.keys():
    if id2line[id_]=="":
        garbageline.append(id_)
for id_ in garbageline:
    del id2line[id_]
print '>> gathered conversations.\n'
movies = get_conversations(garbageline)
print '>> seperate movies.\n'
movies = conv2contextNans(movies)
print 'write file'
generate_Seq2Seqconv(movies,id2line,'corpus_seq2seq/')
generate_LSTMconv(movies,id2line,'corpus_lstm/')
