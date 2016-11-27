class parser_base(object):
   """parser abstract class"""
   def __init__(self):
       self._name = ""


   @property
   def name(self):
      return self._name


   def parse_input(self, input_path, args=""):
       """
       should create the following dict

       {
          "<song_name>" : {
            (recusrsive)     song_id : <song_id>
                             "<clipid >" : {
                             (recursive)        path : <path>
                                                labels: <labels>
                                            }
                          }
          Total_Songs : <total songs>
          Total_tracks: <total clips>
       }
                       
       """

       raise NotImplementedError


