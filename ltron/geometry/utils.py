import numpy

def squared_distance(a, b):
    return sum((aa-bb)**2 for aa,bb in zip(a,b))

def close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

default_rtol = 0.0
default_atol = 0.01

def default_allclose(a, b):
    return numpy.allclose(a, b, rtol=0, atol=0.01)

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

def unscale_transform(transform):
    transform = transform.copy()
    transform[:3,0] /= numpy.linalg.norm(transform[:3,0])
    transform[:3,1] /= numpy.linalg.norm(transform[:3,1])
    transform[:3,2] /= numpy.linalg.norm(transform[:3,2])
    if numpy.linalg.det(transform) < 0.:
        transform[:3,0] *= -1
    return transform
