import nltk
import argparse

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
    BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

print(BLEUscore/len(original_data))
