import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

with open('test.txt', 'r') as f:
   for line in f:
      tts = gTTS(text=line, lang='en')
      tts.save("test")

      sound = AudioSegment.from_mp3("test")
      sound.export("test", format="wav")

      AUDIO_FILE = 'test'
      r = sr.Recognizer()
      with sr.AudioFile(AUDIO_FILE) as source:
          audio = r.record(source) # read the entire audio file
      # recognize speech using Sphinx
      try:
         with open('output.txt', 'w') as ff:
            ff.write(r.recognize_sphinx(audio)+'\n')
      except sr.UnknownValueError:
         print("Sphinx could not understand audio")
      except sr.RequestError as e:
         print("Sphinx error; {0}".format(e))
