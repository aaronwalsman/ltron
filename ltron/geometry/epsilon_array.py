import numpy
from ltron.geometry.utils import default_rtol, default_atol

class EpsilonArray:
    def __init__(
        self, array, rtol=default_rtol, atol=default_atol):
        self.array = array
        self.rtol = rtol
        self.atol = atol

    def __eq__(self, other):
        if isinstance(other, EpsilonArray):
            other = other.array
        if self.array.shape != other.shape:
            return False
        return numpy.allclose(self.array, other, atol=self.atol, rtol=self.rtol)

    def __hash__(self):
        return hash(self.array.shape)
    
    def __str__(self):
        return str(self.array)
    
    def __repr__(self):
        return 'EpsilonArray(' + repr(self.array) + ')'
