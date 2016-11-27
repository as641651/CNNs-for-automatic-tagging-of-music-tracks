import librosa
import numpy as np
import argparse
from audio_processor.processor_base import processor_base


class melgram(processor_base):
   def __init__(self):
      processor_base.__init__(self)
      self._name = "melgram"
      self._feature_length = 96
      self._max_duration = 29.12
      self._num_channels = 1
      self.hop_length = 256
      self.sr = 12000
      self._num_samples = (self._max_duration*self.sr/self.hop_length) + 1

   def process_input(self,input_path,args):
      return self.compute_melgram(input_path)

   def compute_melgram(self, audio_path):
       ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
       96 == #mel-bins and 1366 == #time frame
   
       parameters
       ----------
       audio_path: path for the audio file.
                   Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
   
       '''
   
       # mel-spectrogram parameters
       SR = self.sr
       N_FFT = 512
       N_MELS = self._feature_length
       HOP_LEN = self.hop_length
       DURA = self._max_duration  # to make it 1366 frame..

       src, sr = librosa.load(audio_path, sr=SR)  # whole signal
      # n_sample = src.shape[0]
      # n_sample_fit = int(DURA*SR)

       #if n_sample < n_sample_fit:  # if too short
       #    src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
       #elif n_sample > n_sample_fit:  # if too long
       #    src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
       logam = librosa.logamplitude
       melgram = librosa.feature.melspectrogram
       ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                           n_fft=N_FFT, n_mels=N_MELS)**2,
                   ref_power=1.0)
       ret = ret[np.newaxis, :]
       return ret



def main(args):

   processor = melgram()
   ret = processor.process_input(args.track,args)
   print ret.shape

if __name__ == '__main__':

   parser = argparse.ArgumentParser()

   # INPUT settings
   parser.add_argument('-d',
        default='',
        dest="track",
        help='Input track')

     
   args = parser.parse_args()
   main(args)

