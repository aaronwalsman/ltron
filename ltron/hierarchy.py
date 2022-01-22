import re
import numpy

# main utilities ===============================================================
def map_hierarchies(
    fn,
    *a,
    InDictClass=dict,
    InListClass=(tuple, list),
    OutDictClass=None,
    OutListClass=None
):
    if isinstance(a[0], InDictClass):
        assert all(isinstance(aa, dict) for aa in a[1:])
        assert all(aa.keys() == a[0].keys() for aa in a[1:]), (
            ':'.join([str(aa.keys()) for aa in a]))
        d = {
            key : map_hierarchies(
                fn, *[aa[key] for aa in a],
                InDictClass=InDictClass,
                InListClass=InListClass,
                OutDictClass=OutDictClass,
                OutListClass=OutListClass,
            )
            for key in a[0].keys()
        }
        if OutDictClass is not None:
            d = OutDictClass(d)
        return d
    
    elif isinstance(a[0], InListClass):
        assert all(isinstance(aa, (tuple, list)) for aa in a[1:]), (
            ':'.join([str(aa) for aa in a]))
        assert all(len(aa) == len(a[0]) for aa in a[1:])
        l = [
            map_hierarchies(
                fn, *[aa[i] for aa in a],
                InDictClass=InDictClass,
                InListClass=InListClass,
                OutDictClass=OutDictClass,
                OutListClass=OutListClass,
            )
            for i in range(len(a[0]))
        ]
        if OutListClass is not None:
            l = OutListClass(l)
        return l
    
    else:
        return fn(*a)

def map_dicts(fn, *a):
    if isinstance(a[0], dict):
        assert all(isinstance(aa, dict) for aa in a[1:])
        assert all(aa.keys() == a[0].keys() for aa in a[1:]), (
            ':'.join([str(aa.keys()) for aa in a]))
        return {
            key : map_dicts(fn, *[aa[key] for aa in a])
            for key in a[0].keys()
        }
    
    else:
        return fn(*a)

# general ======================================================================
def index_hierarchy(a, index):
    def fn(a):
        return a[index]
    return map_hierarchies(fn, a)

def set_index_hierarchy(a, b, index):
    def fn(a, b):
        a[index] = b
        return None
    
    map_hierarchies(fn, a, b)
    return None

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

def auto_pad_stack_numpy_hierarchies(*a, pad_axis=0, stack_axis=0, **kwargs):
    pad = numpy.array([len_hierarchy(aa) for aa in a], dtype=numpy.long)
    max_len = max(pad)
    a = [pad_numpy_hierarchy(aa, max_len, axis=pad_axis) for aa in a]
    a = stack_numpy_hierarchies(*a, axis=stack_axis, **kwargs)
    
    return a, pad

def concatenate_lists(a, **kwargs):
    def fn(a):
        if isinstance(a, (tuple, list)):
            return numpy.concatenate(a, **kwargs)
        else:
            return a
    
    return map_dicts(fn, a)

def increase_capacity(a, factor=2):
    def fn(a):
        new_shape = (int(a.shape[0] * factor), *a.shape[1:])
        new_storage = numpy.zeros(new_shape, dtype=a.dtype)
        new_storage[:a.shape[0]] = a
        return new_storage
    
    return map_hierarchies(fn, a)

# conversion ===================================================================
def deep_list_to_tuple(a):
    return map_hierarchies(lambda aa : aa, a, OutListClass=tuple)
