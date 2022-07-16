import numpy

import PIL.Image as Image

import tqdm

import ltron.utils as utils

def tp_fp_fn(prediction, ground_truth, axis = -1):
    count_difference = ground_truth - prediction
    false_negative_locations = count_difference > 0
    false_negatives = count_difference * false_negative_locations
    false_positive_locations = count_difference < 0
    false_positives = -count_difference * false_positive_locations
    true_positives = prediction - false_positives
    
    return true_positives, false_positives, false_negatives

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
    if (precision + recall) == 0:
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

def edge_ap(edges, ground_truth):
    scores = []
    ground_truth_scores = []
    for edge, score in edges.items():
        scores.append(score)
        ground_truth_scores.append(ground_truth.get(edge, 0.0))
    false_negatives = len(set(ground_truth.keys()) - set(edges.keys()))
    return ap(scores, ground_truth_scores, false_negatives)

def instance_map(
        instance_class_predictions, class_false_negatives, extant_classes):
    
    per_class_predictions = {}
    per_class_ground_truth = {}
    for (class_label, score), true_label in instance_class_predictions:
        if class_label not in per_class_predictions:
            per_class_predictions[class_label] = []
            per_class_ground_truth[class_label] = []
        per_class_predictions[class_label].append(float(score))
        per_class_ground_truth[class_label].append(
                float(class_label == true_label))
    
    class_ap = {}
    for class_label in per_class_predictions:
        if class_label not in extant_classes:
            continue
        _, _, class_ap[class_label] = ap(
                per_class_predictions[class_label],
                per_class_ground_truth[class_label],
                class_false_negatives.get(class_label, 0))
    
    return sum(class_ap.values())/len(class_ap), class_ap
