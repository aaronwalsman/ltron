import numpy

from scipy.optimize import linear_sum_assignment

from ltron.symmetry import symmetry_table

def score_offset(x_offset, y_offset):
    x_type, x_color, x_transform = x_offset
    y_type, y_color, y_transform = y_offset
    
    if x_type != y_type:
        return 0.
    
    if x_color != y_color:
        return 0.
    
    # TODO: symmetry
    
    if not numpy.allclose(x_transform, y_transform):
        return 0.
    
    return 1.

def type_color_offset(brick, neighbor):
    neighbor_type = str(neighbor.brick_type)
    neighbor_color = str(neighbor.color)
    neighbor_offset = numpy.linalg.inv(brick.transform) @ neighbor.transform
    
    return neighbor_type, neighbor_color, neighbor_offset

def compute_matching(scores):
    costs = 1. - scores
    x_best, y_best = linear_sum_assignment(costs)
    
    matching_scores = scores[x_best, y_best]
    x_scores = numpy.zeros(scores.shape[0])
    x_scores[x_best] = matching_scores
    y_scores = numpy.zeros(scores.shape[1])
    y_scores[y_best] = matching_scores
    
    return x_scores, y_scores

def pseudo_f1(x_scores, y_scores):
    
    tp = numpy.sum(x_scores)
    fp = len(x_scores) - tp
    fn = len(y_scores) - tp
    
    return tp / (tp + 0.5 * (fp + fn))

def score_brick(x_brick, x_neighbors, y_brick, y_neighbors):
    if str(x_brick.brick_type) != str(y_brick.brick_type):
        return 0.
    
    if str(x_brick.color) != str(y_brick.color):
        return 0.
    
    y_offsets = []
    for y_neighbor in y_neighbors:
        y_offset = type_color_offset(y_brick, y_neighbor)
        y_offsets.append(y_offset)
    
    xy_scores = numpy.zeros((len(x_neighbors), len(y_neighbors)))
    for x, x_neighbor in enumerate(x_neighbors):
        x_offset = type_color_offset(x_brick, x_neighbor)
        for y, y_offset in enumerate(y_offsets):
            xy_scores[x,y] = score_offset(x_offset, y_offset)
    
    print(xy_scores)
    print('-------')
    
    x_scores, y_scores = compute_matching(xy_scores)
    return pseudo_f1(x_scores, y_scores)

def score_configurations(x_bricks, x_neighbors, y_bricks, y_neighbors):
    xy_scores = numpy.zeros((len(x_bricks), len(y_bricks)))
    for x, (x_brick, x_neighbor) in enumerate(zip(x_bricks, x_neighbors)):
        for y, (y_brick, y_neighbor) in enumerate(zip(y_bricks, y_neighbors)):
            xy_scores[x, y] = score_brick(
                x_brick, x_neighbor, y_brick, y_neighbor)
    
    print(xy_scores)
    
    x_scores, y_scores = compute_matching(xy_scores)
    
    return x_scores, y_scores
