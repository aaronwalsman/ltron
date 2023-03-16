import time

import numpy

import tqdm

from scipy.spatial import cKDTree

from ltron.constants import SHAPE_CLASS_LABELS
SHAPE_LABEL_NAMES = {value:key for key, value in SHAPE_CLASS_LABELS.items()}
#from ltron.geometry.utils import default_allclose
from ltron.geometry.symmetry import brick_pose_match_under_symmetry

def initialize_kd_tree(assembly):
    return cKDTree(assembly['pose'][:,:3,3])

def match_assemblies(
    assembly_a,
    assembly_b,
    #part_names,
    kdtree=None,
    radius=0.01,
):
    '''
    Computes the rigid offset between two assemblies that brings as many of
    their bricks into alignment as possible.  The naive implementation would
    use N^2 checks to test the offset between every pairwise combination of
    bricks in assembly_a and assembly_b.  However we throw a bunch of
    hacks at this to make this much more manageable.  The worst case is
    probably still N^2, but this should only come up in pathological cases.
    
    This is optimized for the case where assembly_b is larger than assembly_a.
    '''
    
    # Build the kdtree if one was not passed in.
    if kdtree is None:
        kd_tree = initialize_kd_tree(assembly_b)
    
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
                    # consider this offset.  This is valid because if a and b
                    # matched each other and had a better score than the current
                    # matching, then there must be other bricks not currently
                    # matching, but that would be matching when a and b were
                    # matched.  We can skip this check because we will find that
                    # matching when checking the other bricks.
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
                        
                        valid_matches = find_matches_under_transform(
                            assembly_a,
                            assembly_b,
                            #part_names,
                            a_to_b,
                            kdtree,
                            radius,
                            min_matches=len(best_matches),
                        )
                        
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
                    new_best = len(best_sym_matches)
                    old_best = len(best_matches)
                    if new_best > old_best:
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

def find_matches_under_transform(
    assembly_a,
    assembly_b,
    #part_names,
    a_to_b,
    kdtree=None,
    radius=0.01,
    min_matches=-1,
):
    transformed_a = numpy.matmul(a_to_b, assembly_a['pose'])
    
    # initialize the kdtree if necessary
    if kdtree is None:
        kdtree = initialize_kd_tree(assembly_b)
    
    # Compute the closeset points.
    pos_a = transformed_a[:,:3,3]
    matches = kdtree.query_ball_point(pos_a, radius)
    
    # If the number of matches is less than the current best
    # skip the validation step.
    potential_matches = sum(
        1 for s, m in zip(assembly_a['shape'], matches)
        if len(m) and s != 0
    )
    if potential_matches <= min_matches:
        return []
    
    # Validate the matches.
    valid_matches = validate_matches(
        assembly_a, assembly_b, matches, a_to_b) #, part_names)
    
    return valid_matches

def validate_matches(assembly_a, assembly_b, matches, a_to_b): #, part_names):
    # Ensure that shapes match, colors match, poses match and that each brick
    # is only matched to one other.
    valid_matches = set()
    matched_a = set()
    matched_b = set()
    for a, a_matches in enumerate(matches):
        if a in matched_a:
            continue
        
        for b in a_matches:
            if b in matched_b:
                continue
            
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
                SHAPE_LABEL_NAMES[shape_a], transformed_pose_a, pose_b
            ):
                continue
            
            valid_matches.add((a,b))
            matched_a.add(a)
            matched_b.add(b)
            break
    
    return valid_matches

# Categories I want:
# 1. matched (1 to 1)
# 2. unmatched, but shares shape and color and correct connection (1 to many)
# 3. unmatched, but shares shape and color (1 to many)
# 4. unmatched, wrong shape/color (list)

def matching_edges(assembly, i1=None, i2=None, s1=None, s2=None):
    matches = numpy.ones(assembly['edges'].shape[1], dtype=numpy.bool)
    if i1 is not None:
        matches = matches & (assembly['edges'][0] == i1)
    if i2 is not None:
        matches = matches & (assembly['edges'][1] == i2)
    if s1 is not None:
        matches = matches & (assembly['edges'][2] == s1)
    if s2 is not None:
        matches = matches & (assembly['edges'][3] == s2)
    
    return matches


def compute_misaligned(assembly_a, assembly_b, matches):
    all_a = set(numpy.where(assembly_a['shape'] != 0)[0])
    all_b = set(numpy.where(assembly_b['shape'] != 0)[0])
    a_to_b = dict(matches)
    b_to_a = {b:a for a,b in matches}
    
    unmatched_a = all_a - set(a_to_b.keys())
    unmatched_b = all_b - set(b_to_a.keys())
    
    misaligned_connected_a = {}
    misaligned_connected_b = {}
    shape_color_match_a = {}
    shape_color_match_b = {}
    for a in unmatched_a:
        a_shape = assembly_a['shape'][a]
        a_color = assembly_a['color'][a]
        a_edge_indices = matching_edges(assembly_a, i1=a)
        a_edges = assembly_a['edges'][:,a_edge_indices]
        for b in unmatched_b:
            b_shape = assembly_b['shape'][b]
            b_color = assembly_b['color'][b]
            if a_shape == b_shape and a_color == b_color:
                shape_color_match_a.setdefault(a, set())
                shape_color_match_a[a].add(b)
                shape_color_match_b.setdefault(b, set())
                shape_color_match_b[b].add(a)
                b_edge_indices = matching_edges(assembly_b, i1=b)
                b_edges = assembly_b['edges'][:,b_edge_indices]
                for _, ai2, as1, as2 in a_edges.T:
                    # if ai2 is not in a_to_b, then the connected brick is also
                    # not matched, so there's nothing we can do here
                    if ai2 not in a_to_b:
                        continue
                    
                    for _, bi2, bs1, bs2 in b_edges.T:
                        # THIS NEEDS SYMMETRY TREATMENT
                        # IN FACT, WE PROBABLY NEED SNAPS IN SYMMETRY TABLE
                        # TODO TODO TODO TODO TODO
                        if a_to_b[ai2] == bi2 and as1 == bs1 and as2 == bs2:
                            misaligned_connected_a.setdefault(a, set())
                            misaligned_connected_a[a].add((b, as1, bi2, as2))
                            misaligned_connected_b.setdefault(b, set())
                            misaligned_connected_b[b].add((a, bs1, ai2, bs2))
    
    misaligned_disconnected_a = {
        k:v for k,v in shape_color_match_a.items()
        if k not in misaligned_connected_a
    }
    misaligned_disconnected_b = {
        k:v for k,v in shape_color_match_b.items()
        if k not in misaligned_connected_b
    }
    
    shape_color_mismatch_a = unmatched_a - shape_color_match_a.keys()
    shape_color_mismatch_b = unmatched_b - shape_color_match_b.keys()
    
    return (
        misaligned_connected_a,
        misaligned_connected_b,
        misaligned_disconnected_a,
        misaligned_disconnected_b,
        shape_color_mismatch_a,
        shape_color_mismatch_b,
    )

def compute_misaligned_old(matches, assembly_a, assembly_b):
    # find all instances in assembly_a and assembly_b that are not matched
    all_a = set(numpy.where(assembly_a['shape'] != 0)[0])
    all_b = set(numpy.where(assembly_b['shape'] != 0)[0])
    matched_a = set(m[0] for m in matches)
    matched_b = set(m[1] for m in matches)
    misaligned_a = all_a - matched_a
    misaligned_b = all_b - matched_b
    
    return misaligned_a, misaligned_b

def compute_unmatched(matches, assembly_a, assembly_b):
    
    misaligned_a, misaligned_b = compute_misaligned_old(
        matches, assembly_a, assembly_b)
    
    # find the misaligned instances that match shape and color
    misaligned_matches = [
        (a,b)
        for a in misaligned_a for b in misaligned_b
        if (assembly_a['shape'][a] == assembly_b['shape'][b] and
            assembly_a['color'][a] == assembly_b['color'][b])
    ]
    
    # find the misaligned instances that have no appropriate match
    matched_a = set(m[0] for m in misaligned_matches)
    matched_b = set(m[1] for m in misaligned_matches)
    unmatched_a = misaligned_a - matched_a
    unmatched_b = misaligned_b - matched_b
    
    return misaligned_matches, unmatched_a, unmatched_b

def match_lookup(matching, assembly_a, assembly_b):
    #print('DEPRECATED, switch to compute_unmatched')
    a_to_b = {a:b for a, b in matching}
    b_to_a = {b:a for a, b in matching}
    a_instances = numpy.where(assembly_a['shape'] != 0)[0]
    miss_a = set(a for a in a_instances if a not in a_to_b)
    b_instances = numpy.where(assembly_b['shape'] != 0)[0]
    miss_b = set(b for b in b_instances if b not in b_to_a)
    
    return a_to_b, b_to_a, miss_a, miss_b
