import numpy

from gym import spaces

import renderpy.masks as masks

class ValidFrameSpace(spaces.Discrete):
    def __init__(self):
        super(ValidFrameSpace, 2)

class ImageSpace(spaces.Box):
    '''
    A height x width x 3 uint8 image.
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(ImageSpace, self).__init__(
                low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)

class SegmentationSpace(spaces.Box):
    '''
    A height x width array, where each pixel contains a long refering to
    a segmentation index.
    '''
    def __init__(self, width, height, max_instances=masks.NUM_MASKS-1):
        self.width = width,
        self.height = height
        self.max_instances = max_instances
        super(SegmentationSpace, self).__init__(
                low=0, high=max_instances, shape=(height, width),
                dtype=numpy.long)

class StepSpace(spaces.Discrete):
    '''
    A discrete value to represent the current step index in an episode
    '''
    def __init__(self, max_steps):
        self.max_steps = max_steps
        super(StepSpace, self).__init__(self.max_steps)

class SingleInstanceSelectionSpace(spaces.Discrete):
    '''
    A discrete value to represent selecting one of many istances in a scene
    Instances are 1-indexed, selecting 0 represents selecting nothing.
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(SingleInstanceSelectionSpace, self).__init__(
                self.max_num_instances+1)

class MultiInstanceSelectionSpace(spaces.Box):
    '''
    A list of binary values to represent selecting multiple isntances in a scene
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(MultiInstanceSelectionSpace, self).__init__(
                low=False,
                high=True,
                shape=self.max_num_instances,
                dtype=numpy.bool)

class PixelSelectionSpace(spaces.Box):
    '''
    A binary pixel mask for selecting multiple screen pixels simultaneously
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(PixelSelectionSpace, self).__init__(
                low=False, high=True, shape=(height, width),
                dtype=numpy.bool)

class InstanceListSpace(spaces.Box):
    '''
    A variable length discrete vector of instance class ids represented as
    a fixed length (larger than necessary) long array.
    Class 0 always represents the null class in order to allow
    variable length vectors.
    Also, element zero represents the null instance and should always be zero.
    This is so that edge lists which are also one-indexed point to the right
    location.
    '''
    def __init__(self, num_classes, max_instances):
        self.num_classes = num_classes
        self.max_instances = max_instances
        super(InstanceListSpace, self).__init__(
                low=0,
                high=num_classes,
                shape=(max_instances+1, 1),
                dtype=numpy.long)

class EdgeSpace(spaces.Box):
    '''
    A variable length discrete matrix of instance-id pairs (shape 2xN) where
    N is the maximum number of edges.
    Instance 0 always represents the null instance in order to allow
    variable length lists of edges.
    '''
    def __init__(self, max_instances, max_edges):
        self.max_instances = max_instances
        self.max_edges = max_edges
        super(EdgeSpace, self).__init__(
                low=0,
                high=max_instances,
                shape=(2, max_edges),
                dtype=numpy.long)

class EdgeScoreSpace(spaces.Box):
    '''
    Scores for edges
    '''
    def __init__(self, max_edges):
        self.max_edges = max_edges
        super(EdgeScoreSpace, self).__init__(
                low=0, high=1, shape=(max_edges,), dtype=numpy.float32)

class InstanceGraphSpace(spaces.Dict):
    '''
    A space containing an InstanceListSpace and an EdgeSpace
    '''
    def __init__(self, num_classes, max_instances, max_edges,
            include_edge_score=False):
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.max_edges = max_edges
        instance_list_space = InstanceListSpace(num_classes, max_instances)
        edge_space = EdgeSpace(max_instances, max_edges)
        num_instances_space = SingleInstanceSelectionSpace(max_instances)
        dict_space = {
                'instances':instance_list_space,
                'edges':edge_space,
                'num_instances':num_instances_space}
        if include_edge_score:
            edge_score_space = EdgeScoreSpace(max_edges)
            dict_space['edge_scores'] = edge_score_space
        super(InstanceGraphSpace, self).__init__(dict_space)
