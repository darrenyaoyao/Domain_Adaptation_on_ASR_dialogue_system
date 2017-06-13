import re
import sys
import nltk

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

inputfile = sys.argv[1]
outputfile = sys.argv[2]
output = [ ]
with open(inputfile,'r') as inputdata:
    for line in inputdata:
        temp = nltk.word_tokenize(line)
        text = " ".join(temp)
        text = cleanText(text)
        output.append(text)

out = open(outputfile ,'w')
for i in range(len(output)):
    out.write(output[i]+'\n')

