import copy
from multiset import Multiset

from ltron.matching import match_assemblies, compute_unmatched

def precision_recall(true_positives, false_positives, false_negatives):
    if true_positives + false_positives == 0:
        precision = 0.
    else:
        precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives == 0:
        recall = 0.
    else:
        recall = true_positives / (true_positives + false_negatives)
    return precision, recall

def f1(precision, recall):
    if (precision + recall) == 0.:
        return 0.
    return 2 * (precision * recall) / (precision + recall)

def ap(scores, ground_truth, false_negatives):
    
    sorted_ground_truth = [
            gt for s, gt
            in reversed(sorted(zip(scores, ground_truth)))]
    gt_total = sum(ground_truth) + false_negatives
    
    pr_curve = []
    gt_so_far = 0
    
    for i, gt in enumerate(sorted_ground_truth):
        gt_so_far += gt
        precision = gt_so_far / (i+1)
        if gt_total == 0:
            recall = 0
        else:
            recall = gt_so_far / gt_total
        pr_curve.append([precision, recall])
    
    concave_pr_curve = []
    last_precision = 0
    for precision, recall in reversed(pr_curve):
        if precision < last_precision:
            precision = last_precision
        else:
            last_precision = precision
        concave_pr_curve.append([precision, recall])
    
    concave_pr_curve = list(reversed(concave_pr_curve))
    
    max_precision = {}
    for precision, recall in concave_pr_curve:
        max_precision[recall] = max(max_precision.get(recall, 0), precision)
    
    if len(max_precision) and gt_total:
        ap_score = sum(max_precision.values()) / gt_total
    else:
        ap_score = 0.0
    
    return pr_curve, concave_pr_curve, ap_score

def f1b(predicted, ground_truth):
    predicted_bricks = Multiset(zip(predicted['shape'], predicted['color']))
    predicted_bricks.remove((0,0))
    ground_truth_bricks = Multiset(
        zip(ground_truth['shape'], ground_truth['color']))
    ground_truth_bricks.remove((0,0))
    tp = len(predicted_bricks & ground_truth_bricks)
    fp = len(predicted_bricks - ground_truth_bricks)
    fn = len(ground_truth_bricks - predicted_bricks)
    p,r = precision_recall(tp, fp, fn)
    return f1(p,r)

def f1a(predicted, ground_truth):
    matches, offset = match_assemblies(predicted, ground_truth)
    fp, fn = compute_unmatched(predicted, ground_truth, matches)
    p,r = precision_recall(len(matches), len(fp), len(fn))
    return f1(p,r)

def aed(
    predicted,
    ground_truth,
    radius=0.01,
    miss_a_penalty=1,
    miss_b_penalty=1,
    pose_penalty=1,
):

    running_p_to_gt = {}

    def remove_and_match(p, gt, remove):
        new_p = copy.deepcopy(p)
        new_gt = copy.deepcopy(gt)
        for rp, rgt in remove.items():
            new_p['shape'][rp] = 0
            new_p['color'][rp] = 0
            new_gt['shape'][rgt] = 0
            new_gt['color'][rgt] = 0

        matches, offset = match_assemblies(new_p, new_gt)
        fp, fn = compute_unmatched(new_p, new_gt, matches)
        
        return new_p, new_gt, dict(matches), fp, fn

    first_matches, _ = match_assemblies(predicted, ground_truth)
    p_to_gt = dict(first_matches)
    running_p_to_gt.update(p_to_gt)

    predicted, ground_truth, p_to_gt, fp, fn = remove_and_match(
        predicted, ground_truth, p_to_gt)
    running_p_to_gt.update(p_to_gt)

    edits = 0
    while len(p_to_gt):
        predicted, ground_truth, p_to_gt, fp, fn = remove_and_match(
            predicted, ground_truth, p_to_gt)
        running_p_to_gt.update(p_to_gt)

        edits += 1
    
    edits += len(fp) * 1
    edits += len(fn) * 2
    
    return edits, running_p_to_gt

def f1e(predicted, ground_truth, predicted_to_ground_truth):
    predicted_edges = set()
    for a, b in zip(predicted['edges'][0], predicted['edges'][1]):
        # unidirectional only
        if a == 0 or b == 0 or b < a:
            continue
        gta = predicted_to_ground_truth.get(a, -1)
        gtb = predicted_to_ground_truth.get(b, -1)
        predicted_edges.add((gta, gtb))
    
    ground_truth_edges = set()
    for a, b in zip(ground_truth['edges'][0], ground_truth['edges'][1]):
        # unidirectional only
        if a == 0 or b == 0 or b < a:
            continue
        ground_truth_edges.add((a,b))
    
    tp = predicted_edges & ground_truth_edges
    fp = predicted_edges - ground_truth_edges
    fn = ground_truth_edges - predicted_edges
    p,r = precision_recall(len(tp), len(fp), len(fn))
    
    return f1(p,r)

'''
def aed(predicted, ground_truth):
    edits = 0
    working_predicted = copy_assembly(predicted)
    working_ground_truth = copy_assembly(ground_truth)
    matches = True
    
    # compute sub-assembly edits
    while matches:
        
        # compute the best matches
        matches, offset = match_assemblies(
            working_predicted, working_ground_truth)
        
        # compute false positives and false negatives
        fp, fn = compute_unmatched(
            working_predicted, working_ground_truth, matches)
        
        # remove the matches from the predicted and ground truth assemblies
        working_predicted['shape'][[p for p,gt in matches]] = 0
        working_predicted['color'][[p for p,gt in matches]] = 0
        working_ground_truth['shape'][[gt for p,gt in matches]] = 0
        working_ground_truth['color'][[gt for p,gt in matches]] = 0
        
        edits += 1
    
    # first global matching is free
    edits -= 1
    
    # at this point all the matches are gone and all we have left is
    # false positives and false negatives
    (connected_misaligned_predicted,
     connected_misaligned_ground_truth,
     disconnected_misaligned_predicted,
     disconnected_misaligned_ground_truth,
     false_positives,
     false_negatives) = compute_misaligned(
        working_predicted, working_ground_truth, matches)
    
    edits += len(connected_misaligned_predicted)
    edits += len(disconnected_misaligned_predicted) * 2
    edits += len(false_positives)
    edits += len(false_negatives) * 3
    
    return edits
'''
