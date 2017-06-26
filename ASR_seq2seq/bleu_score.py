import nltk
import argparse
from nltk.translate.bleu_score import SmoothingFunction

chencherry = SmoothingFunction()

parser = argparse.ArgumentParser(description=
                                 'Calculate belu score of original \
                                  predict data and ASR predict data')
parser.add_argument('--original', dest='original')
parser.add_argument('--asr', dest='asr')

args = parser.parse_args()
original = open(args.original)
asr = open(args.asr)

original_data = []
asr_data = []
for line in original:
    original_data.append(line)

for line in asr:
    asr_data.append(line)

BLEUscore = 0
for i in range(len(original_data)):
    hypothesis = asr_data[i]
    reference = original_data[i]
    BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, \
                                                         smoothing_function=chencherry.method2)

print(BLEUscore/len(original_data))
