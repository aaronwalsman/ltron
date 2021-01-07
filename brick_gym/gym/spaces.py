import numpy

from gym import spaces

import renderpy.masks as masks

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
    def __init__(self, max_steps):
        self.max_steps = max_steps
        super(StepSpace, self).__init__(self.max_steps)

class InstanceSelectionSpace(spaces.Discrete):
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(InstanceSelectionSpace, self).__init__(self.max_num_instances+1)

class PixelSelectionSpace(spaces.MultiDiscrete):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(PixelSelectionSpace, self).__init__(tuple(width, height))

class NodeSpace(spaces.Box):
    '''
    A variable length discrete vector of class ids represented as
    a fixed length (larger than necessary) long array.
    Class 0 always represents the null class in order to allow
    variable length vectors.
    '''
    def __init__(self, num_classes, max_nodes):
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        super(NodeSpace, self).__init__(
                low=0, high=num_classes, shape=(max_nodes,),
                dtype=numpy.long)

class EdgeScoreSpace(spaces.Box):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
        super(EdgeScoreSpace, self).__init__(
                low=0, high=1, shape=(max_nodes, max_nodes),
                dtype=numpy.float)

class SparseEdgeSpace(spaces.Box):
    def __init__(self, max_nodes, max_edges):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        super(SparseEdgeSpace, self).__init__(
                low=0, high=max_nodes, shape=(max_edges,2),
                dtype=numpy.long)

class SparseEdgeScoreSpace(spaces.Box):
    def __init__(self, max_edges):
        self.max_edges = max_edges
        super(VectorScoreSpace, self).__init__(
                low=0., high=1., shape=(max_edges),
                dtype=numpy.float)

class GraphScoreSpace(spaces.Dict):
    def __init__(self, num_classes, max_nodes):
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        node_space = NodeSpace(num_classes, max_nodes)
        edge_space = EdgeScoreSpace(max_nodes)
        super(GraphScoreSpace, self).__init__(
                {'nodes':node_space, 'edges':edge_space})

class SparseGraphScoreSpace(spaces.Dict):
    def __init__(self, num_classes, max_nodes, max_edges):
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        node_space = NodeSpace(num_classes, max_nodes)
        sparse_edge_space = SparseEdgeSpace(max_nodes, max_edges)
        sparse_edge_score_space = SparseEdgeScoreSpace(max_edges)
        super(SparseGraphScoreSpace, self).__init__({
                'nodes':node_space,
                'edges':sparse_edge_space,
                'scores':sparse_edge_score_space})
