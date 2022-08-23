import random
import math

import numpy

import tqdm

from pyquaternion import Quaternion

from ltron.geometry.utils import matrix_rotation_axis

for i in tqdm.tqdm(range(10000)):
    while True:
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        z = random.random() * 2 - 1
        n = (x**2 + y**2 + z**2)**0.5
        if n > 1.:
            continue
        axis = [x/n,y/n,z/n]
        break
    
    angle = random.random() * 2 * math.pi
    q = Quaternion(axis=axis, angle=angle)
    r = q.rotation_matrix
    r[0] *= 0.99999999
    
    axis_recovered = matrix_rotation_axis(r)
    
    d = numpy.dot(axis, axis_recovered)
    tol = 1e-5
    if abs(d) > 1 + tol or abs(d) < 1 - tol:
        print(d)
        print(axis_recovered)
        print(r @ axis_recovered)
        import pdb
        pdb.set_trace()
