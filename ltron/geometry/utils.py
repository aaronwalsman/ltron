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

'''
# USE PYQUATERNION
def quaternion_to_matrix(q):
    # Martin Baker!
    qx, qy, qz, qw = q
    qx2, qy2, qz2, qw2 = qx**2, qy**2, qz**2, qw**2
    return numpy.array([
        [1 - 2*qy2 - 2*qz2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, 0],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx2 - 2*qz2, 2*qy*qz - 2*qx*qw, 0],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx2 - 2*qy2, 0],
        [                0,                 0,                 0, 1]])

def uniform_rotation_matrix():
    return quaternion_to_matrix(uniform_unit_quaternion())
'''
