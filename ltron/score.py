import copy

from ltron.matching import match_assemblies, match_lookup

def f1(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))

def score_assemblies(proposal, target, part_lookup):
    matching, offset = match_assemblies(proposal, target, part_lookup)
    tp, _, fp, fn = match_lookup(matching, proposal, target)
    score = f1(len(tp), len(fp), len(fn))
    
    return score, matching

def edit_distance(
    assembly_a,
    assembly_b,
    part_names,
    radius=0.01,
    miss_a_penalty=1,
    miss_b_penalty=1,
):
    
    running_a_to_b = {}
    
    first_a = assembly_a
    first_b = assembly_b
    
    def remove_and_match(a, b, remove):
        a = copy.deepcopy(a)
        b = copy.deepcopy(b)
        for aa, bb in remove.items():
            a['shape'][aa] = 0
            a['color'][aa] = 0
            b['shape'][bb] = 0
            b['color'][bb] = 0
        
        match, offset = match_assemblies(a, b, part_names, radius=radius)
        a_to_b, b_to_a, miss_a, miss_b = match_lookup(match, a, b)
        
        return a, b, a_to_b, miss_a, miss_b
    
    first_match, _ = match_assemblies(
        assembly_a, assembly_b, part_names, radius=radius)
    a_to_b, b_to_a, miss_a, miss_b = match_lookup(
        first_match, assembly_a, assembly_b)
    running_a_to_b.update(a_to_b)
    
    assembly_a, assembly_b, a_to_b, miss_a, miss_b = remove_and_match(
        assembly_a, assembly_b, a_to_b)
    running_a_to_b.update(a_to_b)
    
    d = 0
    while len(a_to_b):
        assembly_a, assembly_b, a_to_b, miss_a, miss_b = remove_and_match(
            assembly_a, assembly_b, a_to_b)
        running_a_to_b.update(a_to_b)
        
        d += 1
    
    d += len(miss_a) * miss_a_penalty
    d += len(miss_b) * miss_b_penalty
    
    return d, running_a_to_b
