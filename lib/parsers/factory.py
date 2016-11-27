from magna_parser import magna_parser


"""Factory method for easily getting audio processors by name."""
__sets = {}

__sets["magna_parser"] = magna_parser()

def get_parser(name):
    """Get the parser module by name"""
    if not __sets.has_key(name):
       raise KeyError('Unknown Parser: {}'.format(name))
    return __sets[name]
