def squared_distance(a, b):
    return sum(aa-bb for aa,bb in zip(a,b))

def close_enough(a, b, tolerance):
    return squared_distance(a,b) <= tolerance**2

def immutable_vector(vector):
    return tuple(vector)

def matrix_is_mirrored(matrix):
    return numpy.linalg.det(matrix) < 0.
