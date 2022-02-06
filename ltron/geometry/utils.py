import math
import numpy

def squared_distance(a, b):
    return sum((aa-bb)**2 for aa,bb in zip(a,b))

def metric_close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

def matrix_angle_close_enough(a, b, max_angular_distance):
    trace_threshold = 1. + 2. * math.cos(max_angular_distance)
    r = a @ b.T
    t = numpy.trace(r)
    return t > trace_threshold

def vector_angle_close_enough(a, b, max_angular_distance, allow_negative=False):
    dot_threshold = math.cos(max_angular_distance)
    dot = a @ b
    if allow_negative:
        return dot > dot_threshold or -dot > dot_threshold
    else:
        return dot > dot_threshold

default_rtol = 0.0
default_atol = 0.01

def default_allclose(a, b):
    return numpy.allclose(a, b, rtol=0, atol=0.01)

def immutable_vector(vector):
    return tuple(vector)

def matrix_is_mirrored(matrix):
    return numpy.linalg.det(matrix) < 0.

def unscale_transform(transform):
    transform = transform.copy()
    transform[:3,0] /= numpy.linalg.norm(transform[:3,0])
    transform[:3,1] /= numpy.linalg.norm(transform[:3,1])
    transform[:3,2] /= numpy.linalg.norm(transform[:3,2])
    if numpy.linalg.det(transform) < 0.:
        transform[:3,0] *= -1
    return transform

def translate_matrix(t):
    transform = numpy.eye(4)
    transform[:3,3] = t
    return transform
