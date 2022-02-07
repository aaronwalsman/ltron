import math

import numpy

from scipy.spatial import cKDTree

def deduplicate(
    points,
    max_distance,
    doublecheck_values=None,
    doublecheck_function=lambda x, y : x == y,
    check_negative=False
):
    '''
    check_negative is for use with Quaternions
    '''
    
    if doublecheck_values is not None:
        assert len(doublecheck_values) == len(points)
    
    if not len(points):
        return []
    
    kdtree = cKDTree(points)
    
    deduplicated_indices = []
    for i, point in enumerate(points):
        matches = kdtree.query_ball_point(point, max_distance)
        if check_negative:
            matches.extend(kdtree.query_ball_point(-point, max_distance))
        
        if doublecheck_values is not None:
            matches = [
                j for j in matches if i == j or doublecheck_function(
                    doublecheck_values[i], doublecheck_values[j])
            ]
        
        if min(matches) == i:
            deduplicated_indices.append(i)
    
    return deduplicated_indices

def rotation_doublecheck_function(max_angular_distance):
    trace_threshold = 1. + 2. * math.cos(max_angular_distance)
    def doublecheck_function(a, b):
        r = a[:3,:3] @ b[:3,:3].T
        t = numpy.trace(r)
        return t > trace_threshold
    
    return doublecheck_function

def deduplicate_transforms(
    transforms,
    max_metric_distance,
    max_angular_distance
):
    points = [transform[:3,3] for transform in transforms]
    
    doublecheck_function = rotation_doublecheck_function(max_angular_distance)
    deduplicated_indices = deduplicate_points(
        points,
        max_metric_distance,
        doublecheck_values=transforms,
        doublecheck_function=doublecheck_function,
    )
    
    return deduplicated_indices
