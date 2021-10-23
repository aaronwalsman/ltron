import numpy

class EpsilonArray:
    def __init__(self, array, *args, **kwargs):
        self.array = array
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        if isinstance(other, EpsilonArray):
            other = other.array
        if self.array.shape != other.shape:
            return False
        return numpy.allclose(self.array, other, *self.args, **self.kwargs)

    def __hash__(self):
        return hash(self.array.shape)
    
    def __str__(self):
        return str(self.array)
    
    def __repr__(self):
        return 'EpsilonArray(' + repr(self.array) + ')'
