import numpy

from scipy.optimize import linear_sum_assignment

from ltron.symmetry import symmetry_table

# hungarian matching
def compute_matching(scores):
    costs = 1. - scores
    x_best, y_best = linear_sum_assignment(costs)
    
    matching_scores = scores[x_best, y_best]
    x_scores = numpy.zeros(scores.shape[0])
    x_scores[x_best] = matching_scores
    y_scores = numpy.zeros(scores.shape[1])
    y_scores[y_best] = matching_scores
    
    return x_scores, y_scores, x_best, y_best

def score_neighbor(
    x_neighbor,
    y_neighbor,
    wrong_pose_discount=0.0,
):
    x_type, x_color, x_transform = x_neighbor
    y_type, y_color, y_transform = y_neighbor
    
    if x_type != y_type:
        return 0.
    
    if x_color != y_color:
        return 0.
    
    # TODO: symmetry
    
    if not numpy.allclose(x_transform, y_transform):
        return wrong_pose_discount
    
    return 1.

def type_color_offset(brick, neighbor):
    neighbor_type = str(neighbor.brick_type)
    neighbor_color = str(neighbor.color)
    neighbor_offset = numpy.linalg.inv(brick.transform) @ neighbor.transform
    
    return neighbor_type, neighbor_color, neighbor_offset

def pseudo_f1(x_scores, y_scores):
    
    tp = numpy.sum(x_scores)
    fp = len(x_scores) - tp
    fn = len(y_scores) - tp
    
    return tp / (tp + 0.5 * (fp + fn))

def score_brick(
    x_brick, x_offsets, y_brick, y_offsets, wrong_pose_discount=0.5
):
    
    if str(x_brick.brick_type) != str(y_brick.brick_type):
        return 0.
    
    if str(x_brick.color) != str(y_brick.color):
        return 0.
    
    xy_scores = numpy.zeros((len(x_offsets), len(y_offsets)))
    for x, x_offset in enumerate(x_offsets):
        for y, y_offset in enumerate(y_offsets):
            xy_scores[x,y] = score_neighbor(
                x_offset, y_offset, wrong_pose_discount)
    
    x_scores, y_scores, x_best, y_best = compute_matching(xy_scores)
    return pseudo_f1(x_scores, y_scores)

def score_configurations(
    x_bricks, x_neighbors, y_bricks, y_neighbors, wrong_pose_discount=0.5
):
    # for each brick in the list of x_bricks, get the offset, color and type
    # of all neighbors
    x_offsets = [
        [type_color_offset(b, n) for n in neighbors]
        for b, neighbors in zip(x_bricks, x_neighbors)
    ]
    # for each brick in the list of y_bricks, get the offset, color and type
    # of all neighbors
    y_offsets = [
        [type_color_offset(b, n) for n in neighbors]
        for b, neighbors in zip(y_bricks, y_neighbors)
    ]
    # score every pairwise association between bricks
    xy_scores = numpy.zeros((len(x_bricks), len(y_bricks)))
    for x, (x_brick, x_offset) in enumerate(zip(x_bricks, x_offsets)):
        for y, (y_brick, y_offset) in enumerate(zip(y_bricks, y_offsets)):
            xy_scores[x, y] = score_brick(
                x_brick, x_offset, y_brick, y_offset, wrong_pose_discount)
    
    # compute the best matching between x bricks and y bricks
    x_scores, y_scores, x_best, y_best = compute_matching(xy_scores)
    
    return pseudo_f1(x_scores, y_scores), x_best, y_best
