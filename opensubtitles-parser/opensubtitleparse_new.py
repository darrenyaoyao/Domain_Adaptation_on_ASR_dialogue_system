'''
The point of this script is to parse all subtitle xml data for source target pairs
It will assume each line is the target of the previous line.
This will store the text data in a tokenized format, meant to be parsed by a deep learning
framework and put into a pre-processed data file.
'''
import xml.etree.ElementTree as ET
import argparse
import os
import re
import errno
import numpy as np
from datetime import datetime
import random

raw_file = "raw.npy"
inc = 0

def main():
    parser = argparse.ArgumentParser(description='Set parameters for xml parser.')
    parser.add_argument('--rootXmlDir', default="OpenSubtitles/en/",
                        help='Path to root directory of xml files')
    parser.add_argument('--dataDir', default="data/",
                        help='Path to directory process data will be saved.')
    args = parser.parse_args()
    processed_data_dir = args.dataDir
    raw_data_dir = args.rootXmlDir

    files = findXmlFiles(raw_data_dir)
    print("Have {} to parse!".format(len(files)))
    # Setup folder structure and data file
    mkdir_p(processed_data_dir)
    outputs= [ ]
    for f in files:
        try:
            output = extractTokenizedPhrases(f, processed_data_dir)
            output = combineSen(output)
            outputs.append(output)
            #np.save(processed_data_dir+str(inc)+raw_file, output)
        except KeyboardInterrupt:
            print("Process stopped by user...")
            return 0
        except Exception as e:
            print(e)
            print("Error in " + f)
            pass
    conversations = makeconversation(outputs)
    del outputs[:]
    generatedata(processed_data_dir, conversations)

def generatedata(path, conversations):   
    question = [ ]
    answer = [ ]
    for conversation in conversations:
        if len(conversation)>1:
            for i in range(len(conversation[1:])):
                question.append(conversation[i]['text'].strip())
                answer.append(conversation[i+1]['text'].strip())
    train_enc = open(path + 'train.enc','w')
    train_dec = open(path + 'train.dec','w')
    test_enc  = open(path + 'test.enc', 'w')
    test_dec  = open(path + 'test.dec', 'w')
    
    TESTSET_SIZE = len(question) // 9
    test_ids = random.sample([i for i in range(len(question))],TESTSET_SIZE)

    for i in range(len(question)):
        if i in test_ids:
            test_enc.write(question[i]+'\n')
            test_dec.write(answer[i]+ '\n' )
        else:
            train_enc.write(question[i]+'\n')
            train_dec.write(answer[i]+ '\n' )
        if i%10000 == 0:
            print ('\n>> written %d lines' %(i)) 

    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
 

def makeconversation(outputs):
    conversations = [ ]
    for data in outputs:
        buff = [ ]
        for datum in data:
            if len(buff)==0:
                buff.append(datum)
            else:
                if datum['timeS']!="":
                    time1=datetime.strptime(datum['timeS'], "%H:%M:%S,%f")
                    time0=datetime.strptime(buff[-1]['timeE'], "%H:%M:%S,%f")
                    if (time1-time0).total_seconds() < 0.8:
                        buff.append(datum)
                    else:
                        conversations.append(buff)
                        buff= [ ]
                        buff.append(datum)
                else:
                    time1=datetime.strptime(datum['timeE'], "%H:%M:%S,%f")
                    time0=datetime.strptime(buff[-1]['timeE'], "%H:%M:%S,%f")
                    if (time1-time0).total_seconds() < 3.5:
                        buff.append(datum)
                    else:
                        conversations.append(buff)
                        buff= [ ]
                        buff.append(datum)
    return conversations    

'''
Combine the seperate term sentence
'''
def combineSen(output):
    new_output = [ ]
    buff = [ ]
    for ii, item in enumerate(output):
        if item['timeS']!='' and item['timeE']=='':
            buff.append(item)
        elif item['timeS']=="" and item['timeE']=='':
            buff.append(item)
        elif item['timeS']=="" and item['timeE']!="":
            if len(buff)!=0:
                new_text = ""
                for b in buff:
                    new_text = new_text+" "+b['text']
                new_item = {'timeS':buff[0]['timeS'], 'timeE':item['timeE'], 
                            'text':str(new_text+" "+item['text'])}
                new_output.append(new_item)
                buff = [ ]
            else:
                new_output.append(item)
        else:
            new_output.append(item)
    return new_output

'''
Loops through folders recursively to find all xml files
'''


def findXmlFiles(directory):
    xmlFiles = []
    for f in os.listdir(directory):
        if os.path.isdir(directory + f):
            xmlFiles = xmlFiles + findXmlFiles(directory + f + "/")
        else:
            xmlFiles.append(directory + f)
    return xmlFiles


'''
The assumption is made (for now) that each <s> node in the xml docs represents
a token, meaning everything has already been tokenized. At first observation
this appears to be an ok assumption.

This function has been modified to print to a single file for each movie
This is for memory consideration when processing later down the pipeline
'''


def extractTokenizedPhrases(xmlFilePath, dataDirFilePath):
    global inc
    inc += 1
    mkfile(dataDirFilePath + str(inc) + raw_file)
    tree = ET.parse(xmlFilePath)
    root = tree.getroot()
    output = [ ]
    print("Processing {}...".format(xmlFilePath))
    for child in root.findall('s'):
        A = []
        timeS = ""
        timeE = ""
        for node in child.getiterator():
            if node.tag == 'w':
                A.append(node.text.encode('ascii', 'ignore').replace('-', ''))
            if node.tag == 'time':
                if (node.get('id'))[-1]=="E":
                    timeE = node.get('value')
                elif (node.get('id'))[-1]=="S":
                    timeS = node.get('value')
        text = " ".join(A)
        text = cleanText(text)
        try:
            if text[0] != '[' and text[-1] != ':':
                out = {'timeS':timeS, 'timeE':timeE, 'text':text}
                output.append(out)
                #with open(dataDirFilePath + str(inc) + raw_file, 'a') as f:
                #    f.write(text + "\n")
        except IndexError:
            pass
    
    return output

'''
This function removes funky things in text
There is probably a much better way to do it, but unless the token list is
much bigger this shouldn't really matter how inefficient it is
'''


def cleanText(text):
    t = text.strip('-')
    t = t.lower()
    t = t.strip('\"')
    regex = re.compile('\(.+?\)')
    t = regex.sub('', t)
    t.replace('  ', ' ')
    regex = re.compile('\{.+?\}')
    t = regex.sub('', t)
    t = t.replace('  ', ' ')
    t = t.replace("~", "")
    t = t.strip(' ')
    return t


'''
Taken from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
'''


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mkfile(path):
    try:
        with open(path, 'w+'):
            return 1
    except IOError:
        print("Data file open, ensure it is closed, and re-run!")
        return 0


if __name__ == "__main__":
    main()
