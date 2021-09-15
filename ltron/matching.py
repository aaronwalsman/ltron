import time

import numpy

from scipy.spatial import cKDTree

from ltron.geometry.grid_bucket import GridBucket

def match_configurations(
    config_a,
    config_b,
    kdtree=None,
    radius=0.01,
):
    t_start = time.time()
    t_lookup = 0.
    t_valid = 0.
    
    na = config_a['class'].shape[0]
    nb = config_b['class'].shape[0]
    
    if kdtree is None:
        kdtree = cKDTree(config_b['pose'][:,:3,3])
    
    ab_tested_matches = set()
    
    unique_a, count_a = numpy.unique(config_a['class'], return_counts=True)
    
    sort_order = numpy.argsort(count_a)
    class_order = unique_a[sort_order]
    
    best_alignment = None
    best_matches = set()
    
    n_checks = 0
    
    for c in class_order:
        if c == 0:
            continue
        
        instance_indices_b = numpy.where(config_b['class'] == c)[0]
        if not len(instance_indices_b):
            continue
        
        instance_indices_a = numpy.where(config_a['class'] == c)[0]
        
        # N^2 hold on!
        for a in instance_indices_a:
            color_a = config_a['color'][a]
            for b in instance_indices_b:
                if (a,b) in ab_tested_matches:
                    continue
                
                color_b = config_b['color'][b]
                if color_a != color_b:
                    continue
                
                n_checks += 1
                
                pose_a = config_a['pose'][a]
                pose_b = config_b['pose'][b]
                a_to_b = pose_b @ numpy.linalg.inv(pose_a)
                transformed_a = numpy.matmul(a_to_b, config_a['pose'])
                
                if True:
                    pos_a = transformed_a[:,:3,3]
                    t_lookup_start = time.time()
                    matches = kdtree.query_ball_point(pos_a, radius)
                    t_lookup_end = time.time()
                    t_lookup += t_lookup_end - t_lookup_start
                    
                    potential_matches = sum(1 for m in matches if len(m))
                    if potential_matches < len(best_matches):
                        continue
                    
                    t_valid_start = time.time()
                    valid_matches = validate_matches(
                        config_a, config_b, matches)
                    t_valid_end = time.time()
                    t_valid += t_valid_end - t_valid_start
                
                if False:
                    # test matching N^3 ???
                    pos_a = transformed_a[:,:3,3]
                    
                    t_lookup_start = time.time()
                    matches = bucket_b.lookup_many(pos_a, radius)
                    t_lookup_end = time.time()
                    t_lookup += t_lookup_end - t_lookup_start
                    
                    valid_matches = validate_matches(
                        config_a, config_b, matches)
                
                if False:
                    t_lookup_start = time.time()
                    valid_matches = slow_match(
                        transformed_a, config_a, config_b)
                    t_lookup_end = time.time()
                    t_lookup += t_lookup_end - t_lookup_start
                
                if len(valid_matches) > len(best_matches):
                    best_alignment = (a,b)
                    best_matches = valid_matches
                ab_tested_matches = ab_tested_matches | valid_matches
    
    t_end = time.time()
    
    print('total time: %.06f'%(t_end-t_start))
    print('lookup time: %.06f'%(t_lookup))
    print('valid time: %.06f'%(t_valid))
    print('checks: %i'%n_checks)
    return best_matches

def slow_match(transformed_poses_a, config_a, config_b):
    valid_matches = set()
    for a, pose_a in enumerate(transformed_poses_a):
        for b, pose_b in enumerate(config_b['pose']):
            if config_a['class'][a] != config_b['class'][b]:
                continue
            if config_a['color'][a] != config_b['color'][b]:
                continue
            if not numpy.allclose(pose_a, pose_b):
                continue
            
            valid_matches.add((a,b))
            break
    
    return valid_matches

def validate_matches(config_a, config_b, matches):
    n = len(matches)
    valid_matches = set()
    total_checks = 0
    for a in range(n):
        for b in matches[a]:
            total_checks += 1
            class_a = config_a['class'][a]
            class_b = config_b['class'][b]
            if class_a != class_b or class_a == 0 or class_b == 0:
                continue
            
            color_a = config_a['color'][a]
            color_b = config_b['color'][b]
            if color_a != color_b:
                continue
            
            pose_a = config_a['pose'][a]
            pose_b = config_b['pose'][b]
            if not numpy.allclose(pose_a, pose_b):
                continue
            
            valid_matches.add((a,b))
            break
    
    print(total_checks)
    return valid_matches
