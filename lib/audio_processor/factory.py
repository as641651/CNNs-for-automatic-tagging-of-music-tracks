from melgram import melgram


"""Factory method for easily getting audio processors by name."""
__sets = {}

__sets["melgram"] = melgram()

def get_audio_processor(name):
    """Get the audio processor module by name"""
    if not __sets.has_key(name):
       raise KeyError('Unknown Processor: {}'.format(name))
    return __sets[name]
