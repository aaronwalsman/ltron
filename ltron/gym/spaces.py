from itertools import product

import numpy

from collections import OrderedDict

from gym.spaces import Box, Discrete, MultiDiscrete, Dict

import splendor.masks as masks

from ltron.constants import MAX_SNAPS_PER_BRICK, WORLD_BBOX
from ltron.name_span import NameSpan

DEFAULT_LDU_MIN = -100000
DEFAULT_LDU_MAX = 100000

# observation spaces -----------------------------------------------------------
class ImageSpace(Box):
    '''
    A height x width x channels uint8 image.
    '''
    def __init__(self, width, height, channels=3):
        self.width = width
        self.height = height
        self.channels = channels
        super(ImageSpace, self).__init__(
            low=0,
            high=255,
            shape=(height, width, channels),
            dtype=numpy.uint8,
        )

class BinaryMaskSpace(Box):
    '''
    A height x width bool array
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super().__init__(
            low=0, high=1, shape=(height, width), dtype=numpy.bool)

class InstanceMaskSpace(Box):
    '''
    A height x width array, where each pixel contains a long refering to
    a segmentation index.
    '''
    def __init__(self, width, height, max_instances=masks.NUM_MASKS-1):
        self.width = width,
        self.height = height
        self.max_instances = max_instances
        super().__init__(
            low=0, high=max_instances, shape=(height, width), dtype=numpy.long)

class SnapMaskSpace(Box):
    '''
    A height x width x 2 array, where each pixel contains a long refering to
    a brick instance index, and another long referring to a connection point
    index.
    '''
    def __init__(self, width, height, max_id=masks.NUM_MASKS-1):
        self.width = width
        self.height = height
        self.max_id = max_id
        super().__init__(
            low=0, high=max_id, shape=(height, width, 2), dtype=numpy.long)

class MaskedTiledImageSpace(Dict):
    def __init__(self, width, height, tile_width, tile_height, channels=3):
        assert width % tile_width == 0
        assert height % tile_height == 0
        self.width = width
        self.height = height
        self.channels = channels
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_width = width//tile_width
        self.mask_height = height//tile_height
        image_space = ImageSpace(width, height, channels=channels)
        tile_space = BinaryMaskSpace(self.mask_width, self.mask_height)
        super().__init__({'image':image_space, 'tile_mask':tile_space})

class TimeStepSpace(Discrete):
    '''
    A discrete value to represent the current step index in an episode
    '''
    def __init__(self, max_steps):
        self.max_steps = max_steps
        super(TimeStepSpace, self).__init__(self.max_steps)

class PhaseSpace(Discrete):
    def __init__(self, num_phases):
        self.num_phases = num_phases
        super(PhaseSpace, self).__init__(self.num_phases)

class InstanceIndexSpace(Discrete):
    '''
    A discrete value to represent selecting one of many istances in a scene
    Instances are 1-indexed, selecting 0 represents selecting nothing.
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super(InstanceIndexSpace, self).__init__(
                self.max_num_instances+1)

class SnapIndexSpace(MultiDiscrete):
    def __init__(self, max_num_instances, max_num_snaps):
        self.max_num_instances = max_num_instances
        self.max_num_snaps = max_num_snaps
        super(SnapIndexSpace, self).__init__(
            [self.max_num_instances+1, max_num_snaps])

class PixelSpace(MultiDiscrete):
    '''
    A single pixel location for drag-and-drop operations
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(PixelSpace, self).__init__([self.height, self.width])

class MultiPixelSpace(Box):
    '''
    A binary pixel mask for selecting multiple screen pixels simultaneously
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super().__init__(
            low=False, high=True, shape=(height, width), dtype=numpy.bool)
    
    '''
    def extract_subspace(self, x, name):
        start, end = self.layout.name_range(name)
        return x[..., start:end]
    '''
    '''
    def ravel_subspace(self, x, out=None):
        result = []
        for name in self.layout.keys():
            start, end = self.layout.name_range(name)
            if out is not None:
                out[..., start:end] = x[name]
            else:
                result.append(name_x)
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, axis=-1)
    '''

class SE3Space(Box):
    def __init__(self, world_bbox=WORLD_BBOX):
        shape = (4,4)
        low = numpy.zeros(shape, dtype=numpy.float32)
        low[:] = -1
        low[:3,3] = world_bbox[0]
        high = numpy.zeros(shape, dtype=numpy.float32)
        high[:] = 1
        high[:3,3] = world_bbox[1]
        super(SE3Space, self).__init__(low=low, high=high, shape=shape)

class MultiSE3Space(Box):
    def __init__(self,
        max_elements,
        world_bbox=WORLD_BBOX,
    ):
        self.max_elements = max_elements
        shape = (max_elements,4,4)
        low = numpy.zeros(shape, dtype=numpy.float32)
        low[:] = -1
        low[:,:3,3] = world_bbox[0]
        low[:,3,3] = 1
        high = numpy.zeros(shape, dtype=numpy.float32)
        high[:] = 1
        high[:,3,:3] = 0
        high[:,:3,3] = world_bbox[1]
        super(MultiSE3Space, self).__init__(low=low, high=high, shape=shape)

class EdgeSpace(Box):
    def __init__(self, max_instances, max_edges):
        low = numpy.zeros((4, max_edges), dtype=numpy.long)
        high = numpy.zeros((4, max_edges), dtype=numpy.long)
        high[:2,:] = max_instances
        high[2:,:] = MAX_SNAPS_PER_BRICK-1
        super(EdgeSpace, self).__init__(
            low=low,
            high=high,
            shape=(4, max_edges),
            dtype=numpy.long,
        )

class AssemblySpace(Dict):
    '''
    A variable length vector of instances, each with a class-label,
    color and pose, along with a list of edges connections.
    '''
    def __init__(
        self,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        world_bbox=WORLD_BBOX,
    ):
        self.shape_ids = shape_ids
        num_shapes = max(self.shape_ids.values())
        self.color_ids = color_ids
        num_colors = max(self.color_ids.values())
        self.max_instances = max_instances
        self.max_edges = max_edges
        
        self.space_dict = {
            'shape' : Box(
                low=0,
                high=num_shapes,
                shape=(max_instances+1,),
                dtype=numpy.long,
            ),
            'color' : Box(
                low=0,
                high=num_colors,
                shape=(max_instances+1,),
                dtype=numpy.long,
            ),
            'pose' : MultiSE3Space(max_instances+1, world_bbox),
            'edges' : EdgeSpace(max_instances, max_edges),
        }
        
        super(AssemblySpace, self).__init__(self.space_dict)
    
    def from_scene(self, scene):
        return scene.get_assembly(
            self.shape_ids,
            self.color_ids,
            self.max_instances,
            self.max_edges,
        )

class MaskedAssemblySpace(Box):
    def __init__(self, max_instances):
        super().__init__(
            low=0, high=1, shape=(max_instances+1), dtype=numpy.bool)

# action spaces ----------------------------------------------------------------
class DiscreteLayoutSpace(Discrete):
    def __init__(self, layout):
        self.layout = layout
        super().__init__(self.layout.total)
    
    def __getattr__(self, attr):
        return getattr(self.layout, attr)

class SymbolicSnapSpace(DiscreteLayoutSpace):
    def __init__(self, max_instances):
        self.max_instances = max_instances
        layout = NameSpan(
            NO_OP=1, **{name:(i, MAX_SNAPS_PER_BRICK)
            for name, i in max_instances.items()}
        )
        super().__init__(layout)

class MultiScreenPixelSpace(DiscreteLayoutSpace):
    '''
    A single pixel location from multiple screens for drag-and-drop operations
    '''
    def __init__(self, screen_shapes): #, channels=1):
        layout = NameSpan()
        layout.add_names(NO_OP=1, DESELECT=1, **screen_shapes)
        
        super(MultiScreenPixelSpace, self).__init__(layout)
    
    def ravel_maps(self, maps, out=None):
        result = []
        for name in self.layout.keys():
            shape = maps[name].shape
            m = maps[name].reshape(*shape[:-3], -1)
            if out is not None:
                start, end = self.layout.name_range(name)
                out[..., start:end] = m
            else:
                result.append(m)
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, axis=-1)
    
    def unravel_maps(self, x):
        maps = {}
        for name in self.layout.keys():
            start, end = self.layout.name_range(name)
            shape = self.layout.get_shape(name)
            x_name = x[..., start:end]
            maps[name] = x_name.reshape(*x.shape[:-1], *shape)
        
        return maps


class DiscreteChain(DiscreteLayoutSpace):
    def __init__(self, subspaces):
        self.subspaces = subspaces
        layout = NameSpan()
        for name, subspace in self.subspaces.items():
            if isinstance(subspace, DiscreteLayoutSpace):
                shape = subspace.layout
            elif isinstance(subspace, Discrete):
                shape = subspace.n
            elif isinstance(subspace, MultiDiscrete):
                shape = subspace.nvec
            else:
                raise ValueError('DiscreteChain accepts '
                    'Discrete, MultiDiscrete spaces only')
            layout.add_names(**{name:shape})
        
        super().__init__(layout)
