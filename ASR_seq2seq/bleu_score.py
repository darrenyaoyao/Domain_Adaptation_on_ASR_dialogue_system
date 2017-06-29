import nltk
import argparse
import re
from nltk.translate.bleu_score import SmoothingFunction

chencherry = SmoothingFunction()

_WORDS_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenizer(sentence):
    words = [ ]
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORDS_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]

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
    tokens = basic_tokenizer(line)
    original_data.append(tokens)

for line in asr:
    tokens = basic_tokenizer(line)
    if len(tokens) == 0:
      tokens = ['']
    asr_data.append(tokens)

# BLEUscore = nltk.translate.bleu_score.corpus_bleu(original_data, asr_data, auto_reweigh=True)
# print(BLEUscore)

BLEUscore = 0
for i in range(len(original_data)):
    hypothesis = asr_data[i]
    reference = original_data[i]
    BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis,
                                                         smoothing_function=chencherry.method2,
                                                         auto_reweigh=True)

print(BLEUscore/len(original_data))
