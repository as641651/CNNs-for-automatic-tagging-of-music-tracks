
def addData(process_dict, song_name, song_id, clip_id, path, labels,year=-1,artist="",additional_tags=""):
            clip_dict = {}
            clip_dict["path"] = path
            clip_dict["labels"] = labels
            if not song_name in process_dict:
               process_dict[song_name] = {}
               process_dict[song_name]["song_id"] = song_id
               process_dict[song_name]["info_tags"] = {}
               process_dict[song_name]["info_tags"]["year"] = year
               process_dict[song_name]["info_tags"]["artist"] = artist
               process_dict[song_name]["info_tags"]["additional_tags"] = additional_tags
               if not "song_id_to_name" in process_dict:
                  process_dict["song_id_to_name"] = {}
                  process_dict["song_id_to_name"][song_id] = song_name
               else:
                  process_dict["song_id_to_name"][song_id] = song_name
            process_dict[song_name][clip_id] = clip_dict
