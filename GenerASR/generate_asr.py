import pandas as pd
import argparse
import codecs
import csv
import sys
from gtts import gTTS
from pydub import AudioSegment
import time
from requests.exceptions import ConnectionError, HTTPError
import speech_recognition as sr

def ttstt(utterance, start, end):
    sound_file = "dialog{}-{}".format(start, end)
    utterance = utterance.strip()
    if len(utterance) == 0:
        return utterance
    try:
        tts = gTTS(text=utterance, lang='en')
    except RuntimeError:
        print("RuntimeError")
        return utterance
    while True:
        try:
            tts.save(sound_file)
            break
        except ConnectionError:
            print("Connection reset by peer")
            time.sleep(10)
        except HTTPError:
            print("HTTPError")
            time.sleep(10)

    sound = AudioSegment.from_mp3(sound_file)
    sound.export(sound_file, format="wav")

    AUDIO_FILE = sound_file
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source) # read the entire audio file
    # recognize speech using Sphinx
    try:
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
        return utterance
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

def generate_asr_dataset(args):
    f = codecs.open(args.input_path, 'r')
    lines = f.readlines()[args.start:args.end]
    f.close()
    asr_dataset = []
    i = args.start
    for sentence in lines:
        if i % 10 == 0:
            print(str(i))
        try:
            asr_dataset.append(ttstt(sentence, args.start, args.end))
        except:
            print('='*50)
            print('Exception!! Start next from {}'.format(i))
            print('='*50)
            break
        i += 1
    return asr_dataset

def write_file(output_path, asr_dataset):
    with codecs.open(output_path, 'w', 'latin-1') as f:
        for example in asr_dataset:
            f.write(example+'\n')

def main(args):
    asr_dataset = generate_asr_dataset(args)
    output_path = args.output_path + '{}-{}.enc'.format(args.start, args.end)
    write_file(output_path, asr_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./train.enc',
                        help='Path for input file')
    parser.add_argument('--output_path', type=str, default='./train_asr',
                        help='Path for output file')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting line')
    parser.add_argument('--end', type=int, default=10000,
                        help='Ending line')

    args = parser.parse_args()
    main(args)
    '''
    dialog = pd.read_csv(input_file, header=None, delimiter='\t', quoting=csv.QUOTE_NONE).as_matrix()
    l = []
    for i in range(len(dialog)):
        print(i)
        start = time.time()
        l.append(list(text_transform(dialog[i][0].split("   "))))
        end = time.time()
        print(end-start)
    '''
