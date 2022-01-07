import time

import numpy

import tqdm

from scipy.spatial import cKDTree

#from ltron.geometry.utils import default_allclose
from ltron.geometry.symmetry import brick_pose_match_under_symmetry

def match_assemblies(
    assembly_a,
    assembly_b,
    part_names,
    kdtree=None,
    radius=0.01,
):
    '''
    Computes the rigid offset between two assemblies that brings as many of
    their bricks into alignment as possible.  The naive implementation would
    use N^2 checks to test the offset between every pairwise combination of
    bricks in assembly_a and assembly_b.  However we throw a bunch of clever
    hacks as this to make this much more manageable.  The worst case is
    probably still N^2, but this should only come up in pathological cases.
    
    This is optimized for the case where assembly_b is larger than assembly_a.
    '''
    
    # Build the kdtree if one was not passed in.
    if kdtree is None:
        kdtree = cKDTree(assembly_b['pose'][:,:3,3])
    
    # Initialize the set of matches that have been tested already.
    ab_tested_matches = set()
    
    # Order the shapes from least common to common.
    # This makes it more likely that we will find a good match sooner.
    unique_a, count_a = numpy.unique(assembly_a['shape'], return_counts=True)
    sort_order = numpy.argsort(count_a)
    shape_order = unique_a[sort_order]
    
    best_alignment = None
    best_matches = set()
    best_offset = numpy.eye(4)
    
    matched_a = set()
    matched_b = set()
    
    finished = False
    while not finished:
        finished = True
        for s in shape_order:
            if s == 0:
                continue
            
            instance_indices_b = numpy.where(assembly_b['shape'] == s)[0]
            if not len(instance_indices_b):
                continue
            
            instance_indices_a = numpy.where(assembly_a['shape'] == s)[0]
            
            for a in instance_indices_a:
                color_a = assembly_a['color'][a]
                for b in instance_indices_b:
                    # If a and b are matched under the current best offset
                    # even if they are not matched to each other, don't
                    # consider this offset.
                    if a in matched_a and b in matched_b:
                        continue
                    
                    # If the offset between a and b has already been tested
                    # don't reconsider this offset.
                    if (a,b) in ab_tested_matches:
                        continue
                    
                    # If the colors don't match, do not consider this offset.
                    color_b = assembly_b['color'][b]
                    if color_a != color_b:
                        continue
                    
                    # Compute the offset between a and b.
                    pose_a = assembly_a['pose'][a]
                    pose_b = assembly_b['pose'][b]
                    
                    # test the offset to each symmetric pose
                    #symmetry_poses = brick_symmetry_poses(
                    #    part_names[s], pose_a)
                    symmetry_poses = [pose_a]
                    best_sym_matches = []
                    for symmetry_pose in symmetry_poses:
                        a_to_b = pose_b @ numpy.linalg.inv(symmetry_pose)
                        transformed_a = numpy.matmul(a_to_b, assembly_a['pose'])
                        
                        # Compute the closeset points.
                        pos_a = transformed_a[:,:3,3]
                        matches = kdtree.query_ball_point(pos_a, radius)
                        
                        # If the number of matches is less than the current best
                        # skip the validation step.
                        potential_matches = sum(
                            1 for s, m in zip(assembly_a['shape'], matches)
                            if len(m) and s != 0
                        )
                        if potential_matches <= len(best_matches):
                            continue
                        
                        # Validate the matches.
                        valid_matches = validate_matches(
                            assembly_a, assembly_b, matches, a_to_b, part_names)
                        
                        # Update the set of tested matches with everything
                        # that was matched in this comparison, this avoids
                        # reconsidering the same offset again later.
                        ab_tested_matches.update(valid_matches)
                        
                        if len(valid_matches) > len(best_sym_matches):
                            best_sym_matches = valid_matches
                    
                    # If the number of valid matches is the best so far, update
                    # and break.  Breaking will exit all the way out to the
                    # main while loop and start over from the beginning.
                    # This is important because it allows us to reconsider
                    # offsets that we might have skipped because of the first
                    # short-circuit in this block.  Two bricks a and b may
                    # have been connected in the previous alignment, but may
                    # not be connected after this better alignment, so now we
                    # need to consider them again.  As convoluted as this is,
                    # it saves a ton of computation.
                    if len(best_sym_matches) > len(best_matches):
                        best_alignment = (a,b)
                        best_matches.clear()
                        best_matches.update(best_sym_matches)
                        best_offset = a_to_b
                        matched_a.clear()
                        matched_b.clear()
                        matched_a.update(set(a for a,b in best_sym_matches))
                        matched_b.update(set(b for a,b in best_sym_matches))
                        finished = False
                        break
                
                # If we just found a new best, start over from the beginning
                if not finished:
                    break
            
            # If we just found a new best, start over from the beginning
            if not finished:
                break
    
    # Return.
    return best_matches, best_offset

def validate_matches(assembly_a, assembly_b, matches, a_to_b, part_names):
    # Ensure that shapes match, colors match, poses match and that each brick
    # is only matched to one other.
    valid_matches = set()
    for a, a_matches in enumerate(matches):
        for b in a_matches:
            shape_a = assembly_a['shape'][a]
            shape_b = assembly_b['shape'][b]
            if shape_a != shape_b or shape_a == 0 or shape_b == 0:
                continue
            
            color_a = assembly_a['color'][a]
            color_b = assembly_b['color'][b]
            if color_a != color_b or color_a == 0 or color_b == 0:
                continue
            
            transformed_pose_a = a_to_b @ assembly_a['pose'][a]
            pose_b = assembly_b['pose'][b]
            if not brick_pose_match_under_symmetry(
                part_names[shape_a], transformed_pose_a, pose_b
            ):
                continue
            
            valid_matches.add((a,b))
            break
    
    return valid_matches

def match_lookup(matching, assembly_a, assembly_b):
    a_to_b = {a:b for a, b in matching}
    b_to_a = {b:a for a, b in matching}
    a_instances = numpy.where(assembly_a['shape'] != 0)[0]
    miss_a = set(a for a in a_instances if a not in a_to_b)
    b_instances = numpy.where(assembly_b['shape'] != 0)[0]
    miss_b = set(b for b in b_instances if b not in b_to_a)
    
    return a_to_b, b_to_a, miss_a, miss_b
