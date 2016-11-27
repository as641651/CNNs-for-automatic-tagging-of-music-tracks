class processor_base(object):
   """audio processor abstract class"""
   def __init__(self):
       self._name = ""
       self._num_channels = 0 #mono / stereo
       self._feature_length = 0 #len(features) every window
       self._max_duration = 0 #sec
       self._num_samples = 0 #sample_rate*sec



   @property
   def name(self):
      return self._name

   @property
   def num_channels(self):
      return self._num_channels

   @num_channels.setter
   def num_channles(self,val):
      self._num_channels = val
   
   @property
   def feature_length(self):
      return self._feature_length

   @feature_length.setter
   def feature_length(self,val):
      self._feature_length = val
   
   @property
   def max_duration(self):
      return self._max_duration

   @max_duration.setter
   def max_duration(self,val):
      self._max_duration = val

   @property
   def num_samples(self):
      return self._num_samples

   @num_samples.setter
   def num_samples(self,val):
      self._num_samples = val

   def process_input(self, input_path, args):
      """ should return an np array of shape
      (num_channels, feature_length, num_samples)
         num_channels = 1 or 2 (mono or stereo)
         feature_length = vector length of the feature extracted every window
         num_samples = usually sample_rate*duration_in_seconds"""

      raise NotImplementedError


