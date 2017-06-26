import nltk
import argparse
import re

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
parser.add_argument('--lines', dest="lines")

args = parser.parse_args()
original = open(args.original)
asr = open(args.asr)
f_line = open(args.lines)
ori_lines = [ ]
for ori_line in f_line:
    ori_lines.append(int(ori_line) -1 )

original_data = []
asr_data = []
for counter,line in enumerate(original):
    if counter in ori_lines:
        tokens = basic_tokenizer(line)
        temp = " ".join(tokens)
        original_data.append(temp)

for line in asr:
    asr_data.append(line)

BLEUscore = 0
for i in range(len(original_data)):
    hypothesis = asr_data[i]
    reference = original_data[i]
    BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

print(BLEUscore/len(original_data))
