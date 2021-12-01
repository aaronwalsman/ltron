from ltron.matching import match_assemblies, match_lookup

def f1(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))

def score_assemblies(proposal, target):
    matching, offset = match_assemblies(proposal, target)
    tp, _, fp, fn = match_lookup(matching, proposal, target)
    score = f1(len(tp), len(fp), len(fn))
    
    return score, matching
