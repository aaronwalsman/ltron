import numpy

from scipy.optimize import linear_sum_assignment

from ltron.symmetry import symmetry_table
from ltron.matching import match_configurations

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

# f1 score, except that examples can be in the range 0-1 where 0 represents a
# false positive and 1 represents true positive and values in between represent
# partial correctness
def pseudo_f1(x_scores, y_scores):
    
    tp = numpy.sum(x_scores)
    fp = len(x_scores) - tp
    fn = len(y_scores) - tp
    
    return tp / (tp + 0.5 * (fp + fn))

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

'''
def type_color_offset(brick, neighbor):
    neighbor_type = str(neighbor.brick_type)
    neighbor_color = str(neighbor.color)
    neighbor_offset = numpy.linalg.inv(brick.transform) @ neighbor.transform
    
    return neighbor_type, neighbor_color, neighbor_offset
'''

def score_brick_assignment(
    x_brick_type,
    x_brick_color,
    x_neighbors,
    y_brick_type,
    y_brick_color,
    y_neighbors,
    wrong_pose_discount=0.
):
    
    if str(x_brick_type) != str(y_brick_type):
        return 0.
    
    if str(x_brick_color) != str(y_brick_color):
        return 0.
    
    if not len(x_neighbors):
        if not len(y_neighbors):
            return 1.
        else:
            return 0.
    
    xy_scores = numpy.zeros((len(x_neighbors), len(y_neighbors)))
    for x, x_neighbor in enumerate(x_neighbors):
        for y, y_neighbor in enumerate(y_neighbors):
            xy_scores[x,y] = score_neighbor(
                x_neighbor, y_neighbor, wrong_pose_discount)
    
    x_scores, y_scores, x_best, y_best = compute_matching(xy_scores)
    pf1 = pseudo_f1(x_scores, y_scores)
    return pf1

def get_max_configuration_instance(configuration):
    try:
        return numpy.max(numpy.where(configuration['class'] != 0))
    except ValueError:
        return 0

def get_neighbors(configuration):
    neighbors = []
    #edges = configuration['edges']['edge_index']
    edges = configuration['edges']
    #for i in range(1, configuration['num_instances']+1):
    #print(numpy.unique(edges[0]))
    #for i in numpy.unique(edges[0]):
    max_x = get_max_configuration_instance(configuration)
    for i in range(1, max_x+1):
        edge_locations = numpy.where(edges[0] == i)[0]
        neighbor_ids = edges[1,edge_locations]
        if len(neighbor_ids):
            neighbor_brick_type = configuration['class'][neighbor_ids]
            neighbor_color = configuration['color'][neighbor_ids]
            neighbor_poses = configuration['pose'][neighbor_ids]
            i_pose = configuration['pose'][i]
            neighbor_offset = [
                numpy.linalg.inv(i_pose) @ neighbor_pose
                for neighbor_pose in neighbor_poses
            ]
            neighbors.append(list(zip(
                neighbor_brick_type, neighbor_color, neighbor_offset)))
        else:
            neighbors.append([])
    
    return neighbors

def score_configurations(
    #x_bricks, x_neighbors, y_bricks, y_neighbors,
    x_configuration,
    y_configuration,
    wrong_pose_discount=0.0
):
    
    # for each brick in the list of x_bricks, get the offset, color and type
    # of all neighbors
    #x_offsets = [
    #    [type_color_offset(b, n) for n in neighbors]
    #    for b, neighbors in zip(x_bricks, x_neighbors)
    #]
    x_neighbors = get_neighbors(x_configuration)
    # for each brick in the list of y_bricks, get the offset, color and type
    # of all neighbors
    #y_offsets = [
    #    [type_color_offset(b, n) for n in neighbors]
    #    for b, neighbors in zip(y_bricks, y_neighbors)
    #]
    y_neighbors = get_neighbors(y_configuration)
    # score every pairwise association between bricks
    #xy_scores = numpy.zeros((len(x_bricks), len(y_bricks)))
    
    #num_x = numpy.max(numpy.where(x_configuration['class'] != 0))
    #num_y = numpy.max(numpy.where(y_configuration['class'] != 0))
    max_x = get_max_configuration_instance(x_configuration)
    max_y = get_max_configuration_instance(y_configuration)
    xy_scores = numpy.zeros((max_x, max_y))
    #xy_scores = numpy.zeros(
    #    (x_configuration['num_instances'], y_configuration['num_instances']))
    #for x in range(1, x_configuration['num_instances']+1):
    #    for y in range(1, y_configuration['num_instances']+1):
    for x in range(1, max_x+1):
        for y in range(1, max_y+1):
            xy_scores[x-1, y-1] = score_brick_assignment(
                x_configuration['class'][x],
                x_configuration['color'][x],
                x_neighbors[x-1],
                y_configuration['class'][y],
                y_configuration['color'][y],
                y_neighbors[y-1],
            )
    
    '''
    for x, (x_brick, x_offset) in enumerate(zip(x_bricks, x_offsets)):
        for y, (y_brick, y_offset) in enumerate(zip(y_bricks, y_offsets)):
            xy_scores[x, y] = score_brick_assignment(
                x_brick, x_offset, y_brick, y_offset, wrong_pose_discount)
    '''
    
    # compute the best matching between x bricks and y bricks
    x_scores, y_scores, x_best, y_best = compute_matching(xy_scores)
    
    return (
        pseudo_f1(x_scores, y_scores),
        x_scores, y_scores,
        x_best, y_best,
        xy_scores,
    )
