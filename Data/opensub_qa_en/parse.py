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
encoder = [ ]
decoder = [ ]
with open(inputfile,'r') as inputdata:
    for line in inputdata:
        temp = line.split('?')
        if len(temp)==2:
            #encoder_temp = (temp[0]+"?").encode('ascii', 'ignore')
            encoder_temp = (temp[0]+"?")
            #decoder_temp = temp[1].encode('ascii', 'ignore')
            decoder_temp = temp[1]
            encoder_temp = nltk.word_tokenize(encoder_temp)
            decoder_temp = nltk.word_tokenize(decoder_temp)
            encoder_temp = " ".join(encoder_temp)
            decoder_temp = " ".join(decoder_temp)
            encoder_temp = cleanText(encoder_temp)
            decoder_temp = cleanText(decoder_temp)
            encoder.append(encoder_temp)
            decoder.append(decoder_temp)

out_de = open(outputfile+".dec",'w')
out_en = open(outputfile+".enc",'w')
for i in range(len(encoder)):
    out_de.write(decoder[i]+'\n')
    out_en.write(encoder[i]+'\n')

