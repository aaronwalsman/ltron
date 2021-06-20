import numpy

from gym import spaces

import splendor.masks as masks

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

class SnapSegmentationSpace(spaces.Box):
    '''
    A height x width x 2 array, where each pixel contains a long refering to
    a brick instance index, and another long referring to a connection point
    index.
    '''
    def __init__(self, width, height, max_id=masks.NUM_MASKS-1):
        self.width = width
        self.height = height
        self.max_id = max_id
        super(SnapSegmentationSpace, self).__init__(
                low=0, high=max_id, shape=(height, width, 2),
                dtype=numpy.long)

class StepSpace(spaces.Discrete):
    '''
    A discrete value to represent the current step index in an episode
    '''
    def __init__(self, max_steps):
        self.max_steps = max_steps
        super(StepSpace, self).__init__(self.max_steps)

class SingleInstanceIndexSpace(spaces.Discrete):
    '''
    A discrete value to represent selecting one of many istances in a scene
    Instances are 1-indexed, selecting 0 represents selecting nothing.
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(SingleInstanceIndexSpace, self).__init__(
                self.max_num_instances+1)

class SingleSnapIndexSpace(spaces.MultiDiscrete):
    def __init__(self, max_num_instances, max_num_snaps):
        self.max_num_instances = max_num_instances
        self.max_num_snaps = max_num_snaps
        super(SingleSnapIndexSpace, self).__init__(
            [self.max_num_instances+1, max_num_snaps])

class MultiInstanceSelectionSpace(spaces.Box):
    '''
    A list of binary values to represent selecting multiple isntances in a scene
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(MultiInstanceSelectionSpace, self).__init__(
                low=False,
                high=True,
                shape=(self.max_num_instances+1,),
                dtype=numpy.bool)

class SinglePixelSelectionSpace(spaces.MultiDiscrete):
    '''
    A single pixel loection for drag-and-drop operations
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(SinglePixelSelectionSpace, self).__init__(
            [self.height, self.width])

class MultiPixelSelectionSpace(spaces.Box):
    '''
    A binary pixel mask for selecting multiple screen pixels simultaneously
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(MultiPixelSelectionSpace, self).__init__(
                low=False, high=True, shape=(height, width),
                dtype=numpy.bool)

class MultiInstanceDirectionSpace(spaces.Box):
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(MultiInstanceDirectionSpace, self).__init__(
                low=-1.0,
                high=1.0,
                shape=(self.max_num_instances+1,3))

class SingleSE3Space(spaces.Box):
    def __init__(self, scene_min=-1000, scene_max=1000):
        shape = (4,4)
        low = numpy.zeros(shape, dtype=numpy.float32)
        low[:] = -1
        low[:3,3] = scene_min
        high = numpy.zeros(shape, dtype=numpy.float32)
        high[:] = 1
        high[:3,3] = scene_max
        super(SingleSE3Space, self).__init__(low=low, high=high, shape=shape)

class MultiSE3Space(spaces.Box):
    def __init__(self, max_num_instances, scene_min=-1000, scene_max=1000):
        self.max_num_instances = max_num_instances
        shape = (max_num_instances+1,4,4)
        low = numpy.zeros(shape, dtype=numpy.float32)
        low[:] = -1
        low[:,:3,3] = scene_min
        high = numpy.zeros(shape, dtype=numpy.float32)
        high[:] = 1
        high[:,:3,3] = scene_max
        super(MultiSE3Space, self).__init__(low=low, high=high, shape=shape)

class ClassLabelSpace(spaces.Box):
    def __init__(self, num_classes, max_instances):
        self.num_classes = num_classes
        self.max_instances = max_instances
        
        super(ClassLabelSpace, self).__init__(
            low=0.,
            high=num_classes,
            shape=(max_instances+1, 1),
            dtype=numpy.long,
        )

class ClassDistributionSpace(spaces.Box):
    def __init__(self, num_classes, max_instances):
        self.num_classes = num_classes
        self.max_instances = max_instances
        
        super(ClassDistributionSpace, self).__init__(
            low=0.,
            high=1.,
            shape=(max_instances+1, num_classes),
        )

class InstanceListSpace(spaces.Dict):
    '''
    A variable length vector of instances
    '''
    def __init__(self, num_classes, max_instances, include_score=False):
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.include_score = include_score
        
        space_dict = {
            'num_instances' : SingleInstanceIndexSpace(max_instances),
            'label' : spaces.Box(
                low=0,
                high=num_classes,
                shape=(max_instances+1,),
                dtype=numpy.long),
        }
        
        if include_score:
            space_dict['score'] = spaces.Box(
                    low=0.,
                    high=1.,
                    shape=(max_instances+1, 1))
        
        super(InstanceListSpace, self).__init__(space_dict)
    
    def from_scene(self, scene, class_lookup, score=None):
        result = {}
        result['num_instances'] = len(scene.instances)
        result['label'] = numpy.zeros(
            (self.max_instances+1,), dtype=numpy.long)
        for instance_id, instance in scene.instances.items():
            brick_type_name = str(instance.brick_type)
            class_id = class_lookup[brick_type_name]
            result['label'][instance_id] = class_id
        
        if score is not None:
            assert self.include_score
            result['score'] = score
        
        return result

class EdgeSpace(spaces.Dict):
    def __init__(
        self,
        max_instances,
        max_snaps,
        max_edges,
        include_score=False
    ):
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.include_score = include_score
        
        space_dict = {
            'num_edges' : spaces.Discrete(max_edges+1),
            'edge_index' : spaces.Box(
                low=0,
                high=max(max_instances, max_snaps-1),
                shape=(4, max_edges),
                dtype=numpy.long),
        }
        
        if include_score:
            space_dict['score'] = spaces.Box(
                    low=0,
                    high=1.,
                    shape=(max_edges, 1))
        
        super(EdgeSpace, self).__init__(space_dict)
    
    def from_scene(self, scene, score=None):
        result = {}
        edges = scene.get_all_edges(unidirectional=True)
        num_edges = edges.shape[-1]
        if num_edges < self.max_edges:
            edges = numpy.concatenate(
                (edges,
                 numpy.zeros((4, self.max_edges-num_edges), dtype=numpy.long)),
                axis=1)
        result['num_edges'] = num_edges
        result['edge_index'] = edges
        if score is not None:
            assert self.include_score
            result['score'] = score
        
        return result

class InstanceGraphSpace(spaces.Dict):
    '''
    A space containing an InstanceListSpace and an EdgeSpace
    '''
    def __init__(self,
            num_classes,
            max_instances,
            max_snaps,
            max_edges,
            include_edge_score=False,
            include_instance_score=False):
        
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.max_snaps = max_snaps
        self.max_edges = max_edges
        self.include_instance_score = include_instance_score
        self.include_edge_score = include_edge_score
        
        dict_space = {
            'instances' : InstanceListSpace(
                num_classes,
                max_instances,
                self.include_instance_score
            ),
            'edges' : EdgeSpace(
                max_instances,
                max_snaps,
                max_edges,
                self.include_edge_score,
            ),
        }
        
        super(InstanceGraphSpace, self).__init__(dict_space)
    
    def from_scene(
        self,
        scene,
        class_lookup,
        instance_score=None,
        edge_score=None,
    ):
        result = {}
        result['instances'] = self['instances'].from_scene(
            scene, class_lookup, score=instance_score)
        result['edges'] = self['edges'].from_scene(
            scene, score=edge_score)
        
        return result
