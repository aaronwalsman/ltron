import math
import numpy

def squared_distance(a, b):
    return sum((aa-bb)**2 for aa,bb in zip(a,b))

def metric_close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

def surrogate_angle(a, b):
    a = a[:3,:3]
    b = b[:3,:3]
    r = a @ b.T
    return numpy.trace(r)

def matrix_angle_close_enough(a, b, max_angular_distance):
    #a = a[:3,:3]
    #b = b[:3,:3]
    trace_threshold = 1. + 2. * math.cos(max_angular_distance)
    #r = a @ b.T
    #t = numpy.trace(r)
    t = surrogate_angle(a, b)
    return t > trace_threshold

def matrix_rotation_axis(a):
    a = a[:3,:3]
    w, v = numpy.linalg.eig(a)
    tol = 1e-5
    i = numpy.where(
        (w.real < 1+tol) &
        (w.real > 1-tol) &
        (w.imag < 0+tol) &
        (w.imag > 0-tol)
    )[0][0]
    return v[:,i].real

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

def orthogonal_orientations(offset=None):
    if offset is None:
        offset = numpy.eye(4)
    else:
        offset = offset.copy()
        offset[:3,3] = 0
    orientations = []
    for i in range(3):
        for si in (1,-1):
            vi = numpy.array([0,0,0])
            vi[i] = si
            for j in range(3):
                if j == i:
                    continue
                for sj in (1,-1):
                    vj = numpy.array([0,0,0])
                    vj[j] = sj
                    vk = numpy.cross(vi, vj)
                    orientation = numpy.eye(4)
                    orientation[:3,0] = vi
                    orientation[:3,1] = vj
                    orientation[:3,2] = vk
                    orientations.append(offset@orientation)
    
    return orientations

def single_axis_orthogonal_orientations(offset=None):
    if offset is None:
        offset = numpy.eye(4)
    else:
        offset = offset.copy()
        offset[:3,3] = 0
    orientations = []
    orientations.append(offset@numpy.eye(4))
    for i in range(3):
        j = (i+1)%3
        k = (i+2)%3
        
        a = numpy.eye(4)
        a[j,j] = 0
        a[j,k] = 1
        a[k,j] = -1
        a[k,k] = 0
        orientations.append(offset@a)
        
        b = numpy.eye(4)
        b[j,j] = -1
        b[k,k] = -1
        orientations.append(offset@b)
        
        c = numpy.eye(4)
        c[j,j] = 0
        c[j,k] = -1
        c[k,j] = 1
        c[k,k] = 0
        orientations.append(offset@c)
    
    return orientations

def local_pivot(transform):
    return transform, numpy.linalg.inv(transform)

def global_pivot(transform):
    offset = numpy.eye(4)
    offset[:3,3] = transform[:3,3]
    return offset, numpy.linalg.inv(offset)

def projected_global_pivot(transform, offset=None):
    if offset is None:
        offset = numpy.eye(4)
    orthos = orthogonal_orientations()
    sas = [surrogate_angle(transform@o, offset) for o in orthos]
    i = numpy.argmax(sas)
    closest_orthogonal = orthos[i]
    closest_offset = transform @ closest_orthogonal
    closest_offset[:3,3] = transform[:3,3]
    return closest_offset, numpy.linalg.inv(closest_offset)

def space_pivot(space, transform, offset=None):
    if space == 'local':
        return local_pivot(transform)
    elif space == 'global':
        return global_pivot(transform)
    elif space == 'projected_global':
        return projected_global_pivot(transform)
    elif space == 'projected_camera':
        return projected_global_pivot(transform, offset=offset)
