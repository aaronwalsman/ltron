import numpy

def squared_distance(a, b):
    return sum((aa-bb)**2 for aa,bb in zip(a,b))

def close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

def immutable_vector(vector):
    return tuple(vector)

def matrix_is_mirrored(matrix):
    return numpy.linalg.det(matrix) < 0.

def uniform_unit_quaternion():
    l22 = 2
    while l22 > 1:
        q = [random.rand() for _ in range(4)]
        l22 = squared_distance(q, [0,0,0,0])
    return q / (l22**0.5)
