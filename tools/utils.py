"""Aravind Sankaran, RWTH"""

import os
import json
import wave

def get_tracks(data_folder = "/home/sankaran/Thesis/data/Waves", years_json = "../stats/years.json",folder = ""):

    """ Returns the list of all tracks in a specified directory
        data_folder : Main director
        year_json : json file containing the list of sub folders
        folder: a spicified sub-directoy. In this case, data will not 
                be loaded from json file"""

    folders = []
    if not folder:
       with open(years_json) as yj:
           years = json.load(yj)
           for y in years:
             folders.append(str(y))
    else:
       folders.append(folder)   

    track_paths = []
    for f in folders:
       path = os.path.join(data_folder,f)
       for track in os.listdir(path):
           if track.endswith(".wav"):
              track_paths.append(os.path.join(path,track))

    return track_paths



def make_chunk(in_audio,out_audio,start,end):

    """extracts a specific chunk from wave file
       Params:
          in_audio: path of input  
          out_audio: path of output
          start,end : start and end time in sec"""

    origAudio = wave.open(in_audio,'r')
    frameRate = origAudio.getframerate()
    nChannels = origAudio.getnchannels()
    sampWidth = origAudio.getsampwidth()

    origAudio.setpos(start*frameRate)
    chunkData = origAudio.readframes(int((end-start)*frameRate))

    chunkAudio = wave.open(out_audio,'w')
    chunkAudio.setnchannels(nChannels)
    chunkAudio.setsampwidth(sampWidth)
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()
