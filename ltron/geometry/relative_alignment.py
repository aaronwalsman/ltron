import random

import numpy

from pyquaternion import Quaternion

def relative_alignment(transforms, relative_estimates, iterations):
    for i in range(iterations):
        new_transforms = []
        for i, transform in enumerate(transforms):
            estimates = []
            for j, other in enumerate(transforms):
                estimate = other @ relative_estimates[j][i] # or something it
                estimates.append(estimate)
            average = average_transforms(estimates)
            new_transforms.append(average)
        
        transforms = new_transforms
    
    return transforms

def average_transforms(transforms):
    '''
    Kids, don't do this, this is not ok.
    '''
    average = numpy.eye(4)
    translates = [transform[:3,3] for transform in transforms]
    average_translate = sum(translates)/len(translates)
    average[:3,3] = average_translate
    
    qs = [Quaternion(matrix=transform[:3,:3], rtol=1e-4, atol=1e-4) for transform in transforms]
    base = random.choice(qs)
    flipped_qs = []
    for q in qs:
        if q[0]*base[0] + q[1]*base[1] + q[2]*base[2] + q[3]*base[3] < 0:
            flipped_qs.append(-q)
        else:
            flipped_qs.append(q)
    
    averaged_q = sum(flipped_qs) / len(flipped_qs)
    averaged_q = averaged_q.normalised
    average[:3,:3] = averaged_q.rotation_matrix
    
    return average

def test():
    def random_configuration(n):
        def random_orientation():
            while True:
                q = Quaternion([random.random()*2-1 for _ in range(4)])
                if q.norm <= 1.:
                    return q.normalised
        
        bounds = [-200, 200]
        ts = [[random.uniform(*bounds) for _ in range(3)] for _ in range(n)]
        qs = [random_orientation() for _ in range(n)]
        
        configuration = []
        for t, q in zip(ts, qs):
            transform = numpy.eye(4)
            transform[:3,:3] = q.rotation_matrix
            transform[:3,3] = t
            configuration.append(transform)
        
        return configuration
    
    n = 5
    
    goal_configuration = random_configuration(n)
    current_configuration = random_configuration(n)
    
    offsets = []
    for i, t1 in enumerate(goal_configuration):
        row = []
        for j, t2 in enumerate(goal_configuration):
            offset = numpy.linalg.inv(t1) @ t2
            row.append(offset)
        offsets.append(row)
    
    print('Goal:')
    print(offsets[0][1])
    for i in range(10):
        print('Current (%i):'%i)
        print(numpy.linalg.inv(
            current_configuration[0]) @ current_configuration[1])
        current_configuration = relative_alignment(
            current_configuration, offsets, 1)
