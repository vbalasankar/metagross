__version__ = "0.0.1"


def sanity_add(a, b):


    from . import _C
    return _C.add(a, b)
