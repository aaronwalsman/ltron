import math
import numpy

def squared_distance(a, b):
    return sum((aa-bb)**2 for aa,bb in zip(a,b))

def metric_close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

def matrix_angle_close_enough(a, b, max_angular_distance):
    a = a[:3,:3]
    b = b[:3,:3]
    trace_threshold = 1. + 2. * math.cos(max_angular_distance)
    r = a @ b.T
    t = numpy.trace(r)
    return t > trace_threshold

def matrix_rotation_axis(a):
    a = a[:3,:3]
    #s = a - a.T
    #axis = numpy.array([a[2,1], a[0,2], a[1,0]])
    #if numpy.all(axis == 0):
    #    return axis
    #else:
    #    return axis / numpy.linalg.norm(axis)
    w, v = numpy.linalg.eig(a)
    i = numpy.where(w == 1)[0][0]
    return v[:,i]

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
    n0 = numpy.linalg.norm(transform[:3,0])
    n1 = numpy.linalg.norm(transform[:3,1])
    n2 = numpy.linalg.norm(transform[:3,2])
    if n0 == 0. or n1 == 0. or n2 == 0.:
        raise Exception('zero scaled matrix')
    transform[:3,0] /= n0
    transform[:3,1] /= n1
    transform[:3,2] /= n2
    if numpy.linalg.det(transform) < 0.:
        transform[:3,0] *= -1
    return transform

def translate_matrix(t):
    transform = numpy.eye(4)
    transform[:3,3] = t
    return transform
