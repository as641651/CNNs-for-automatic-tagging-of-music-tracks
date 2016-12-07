
def addData(process_dict, song_name, song_id, clip_id, path, labels):
            clip_dict = {}
            clip_dict["path"] = path
            clip_dict["labels"] = labels
            if not song_name in process_dict:
               process_dict[song_name] = {}
            process_dict[song_name][clip_id] = clip_dict
            process_dict[song_name]["song_id"] = song_id
