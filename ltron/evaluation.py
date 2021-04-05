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
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall

def f1(precision, recall):
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

'''
def dataset_node_and_edge_ap(model, multi_env, dump_images=False):
    
    num_paths = multi_env.get_attr('num_paths')
    multi_env.call_method(
            'start_over',
            [{'reset_mode' : 'sequential'}] * multi_env.num_processes)
    
    iterate = tqdm.tqdm(range(max(num_paths)))
    predicted_step_edges = [{}]
    ground_truth_edges = {}
    episode_step_ap = []
    for i in iterate:
        
        # start a new batch of episodes
        observations = multi_env.call_method('reset')
        
        # get ground truth edges for each episode
        node_and_edge_labels = multi_env.call_method('get_node_and_edge_labels')
        episode_edge_labels = []
        for j, (nodes, edges) in enumerate(node_and_edge_labels):
            updated_labels = {
                    ((i,j), a-1, b-1, c, d) : 1.0
                    for a,b,c,d in edges}
            ground_truth_edges.update(updated_labels)
            episode_edge_labels.append(updated_labels)
        
        terminal = [False] * multi_env.num_processes
        step = 0
        hidden_state = None
        while not all(terminal):
            
            # dump images
            if dump_images:
                frame_images, segment_images = zip(*observations)
                for j in range(len(frame_images)):
                    episode_id = i * len(frame_images) + j
                    Image.fromarray(frame_images[j]).save(
                            './frame_%i_%i.png'%(episode_id, step))
                    for seg in range(len(segment_images[j])):
                        Image.fromarray(segment_images[j][seg]).save(
                                './seg_%i_%i_%i.png'%(episode_id, step, seg))
            
            # prediction
            actions, node_predictions, edge_matrix, hidden_state = model(
                    observations, hidden_state)
            for j in range(multi_env.num_processes):
                if not terminal[j]:
                    edge_scores = utils.matrix_to_edge_scores(
                            (i,j), node_predictions[j], edge_matrix[j])
                    if len(predicted_step_edges) <= step:
                        predicted_step_edges.append({})
                    predicted_step_edges[step].update(edge_scores)
                    _, _, ap = edge_ap(edge_scores, episode_edge_labels[j])
                    #================
                    #episode_id = i * multi_env.num_processes + j
                    #print('STEP AP (%i, %i): %f'%(episode_id, step, ap))
                    #print(node_predictions[j])
                    #print(edge_matrix[j])
                    #print(edge_scores)
                    #print(episode_edge_labels[j])
                    #================
                    if len(episode_step_ap) <= step:
                        episode_step_ap.append([])
                    episode_step_ap[step].append(ap)
            step_result = multi_env.call_method(
                    'step', [{'action':{'hide':action}} for action in actions])
            observations, _, terminal, _ = zip(*step_result)
            
            step += 1
    
    step_edge_ap = []
    for step_edges in predicted_step_edges:
        pr, cpr, ap = edge_ap(step_edges, ground_truth_edges)
        step_edge_ap.append(ap)
    
    print('Step Edge AP (all)')
    print(step_edge_ap)
    average_episode_step_ap = [sum(ap)/len(ap) for ap in episode_step_ap]
    print('Step Edge AP (individual)')
    print(average_episode_step_ap)
    
    return step_edge_ap
'''
