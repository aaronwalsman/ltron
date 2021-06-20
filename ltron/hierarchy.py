import re
import numpy

# main utility =================================================================
def map_hierarchies(fn, *a):
    if isinstance(a[0], dict):
        assert all(isinstance(aa, dict) for aa in a[1:])
        assert all(aa.keys() == a[0].keys() for aa in a[1:])
        return {
            key : map_hierarchies(fn, *[aa[key] for aa in a])
            for key in a[0].keys()
        }
    
    elif isinstance(a[0], (tuple, list)):
        assert all(isinstance(aa, (tuple, list)) for aa in a[1:])
        assert all(len(aa) == len(a[0]) for aa in a[1:])
        return [
            map_hierarchies(fn, *[aa[i] for aa in a])
            for i in range(len(a[0]))
        ]
    
    else:
        return fn(*a)

# general ======================================================================
def index_hierarchy(a, index):
    def fn(a):
        return a[index]
    return map_hierarchies(fn, a)

def hierarchy_branch(a, branch_keys):
    for key in branch_keys:
        a = a[key]
    return a

def len_hierarchy(a):
    class HierarchyLenException(Exception):
        def __init__(self, length):
            self.length = length
    def fn(a):
        raise HierarchyLenException(len(a))

    try:
        map_hierarchies(fn, a)
    except HierarchyLenException as e:
        return e.length

    return 0

def x_like_hierarchy(a, value):
    def fn(a):
        return value
    return map_hierarchies(fn, a)

def string_index_hierarchy(a, string_index):
    '''
    This is a little crazy-town.
    '''
    while string_index:
        separator = string_index[0]
        string_index = string_index[1:]
        start, end = re.search('^[^.\\]]+', string_index).span()
        token = string_index[start:end]
        string_index = string_index[end:]
        if separator == '[':
            try:
                token = int(token)
            except ValueError:
                pass
            string_index = string_index[1:]
            a = a[token]
        elif separator == '.':
            a = getattr(a, token)
    
    return a

# numpy ========================================================================
def concatenate_numpy_hierarchies(*a, **kwargs):
    def fn(*a):
        return numpy.concatenate(a, **kwargs)
    return map_hierarchies(fn, *a)

def stack_numpy_hierarchies(*a, **kwargs):
    def fn(*a):
        return numpy.stack(a, **kwargs)
    return map_hierarchies(fn, *a)

def pad_numpy_hierarchy(a, pad, axis=0):
    def fn(a):
        if a.shape[axis] < pad:
            pad_shape = list(a.shape)
            pad_shape[axis] = pad - a.shape[axis]
            z = numpy.zeros(pad_shape, dtype=a.dtype)
            return numpy.concatenate((a, z), axis=axis)

        else:
            return a

    return map_hierarchies(fn, a)

