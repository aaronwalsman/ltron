import numpy
from ltron.geometry.utils import default_rtol, default_atol
from ltron.hierarchy import deep_list_to_tuple

class EpsilonArray:
    '''
    An immutable, hashable numpy-like array that uses numpy.allclose for
    equality comparisons.  This is designed to provide a format that can be
    used in dicts and sets to indicate approximate real-valued containment.
    
    Because equality is compared using numpy.allclose, the only thing that's
    used for the hash is the array's shape, and not it's content.  This means
    that performance will be O(N) (very bad) for standard dict/set operations.
    As such, this should not be considered a high-performance data type, and
    this should not be used in very large sets and dicts.
    '''
    def __init__(self, data, rtol=default_rtol, atol=default_atol):
        data_array = numpy.array(data)
        self.shape = data_array.shape
        data_list = data_array.tolist()
        data_tuple = deep_list_to_tuple(data_list)
        self.data = data_tuple
        self.rtol = rtol
        self.atol = atol

    def __eq__(self, other):
        other_shape = numpy.shape(other)
        if self.shape != other_shape:
            return False
        
        if isinstance(other, EpsilonArray):
            other_data = other.data
        else:
            other_data = other
        
        return numpy.allclose(
            self.data, other_data, atol=self.atol, rtol=self.rtol)
    
    def __array__(self, dtype=None):
        return numpy.array(self.data).astype(dtype)
    
    def __hash__(self):
        return hash(self.shape)
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return 'EpsilonArray(' + repr(self.data) + ')'
