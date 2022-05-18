from itertools import product

import numpy

from collections import OrderedDict

from gym.spaces import Box, Discrete, MultiDiscrete, Dict

import splendor.masks as masks

from ltron.constants import MAX_SNAPS_PER_BRICK, WORLD_BBOX
from ltron.name_span import NameSpan

DEFAULT_LDU_MIN = -100000
DEFAULT_LDU_MAX = 100000

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

class DiscreteSpanSpace(Discrete):
    def __init__(self, span):
        self.span = span
        super().__init__(self.span.total)
    
    def __getattr__(self, attr):
        return getattr(self.span, attr)

class SymbolicSnapSpace(DiscreteSpanSpace):
    def __init__(self, max_instances):
        self.max_instances = max_instances
        span = NameSpan(NO_OP=1,
            **{name:(i, MAX_SNAPS_PER_BRICK)
            for name, i in max_instances.items()}
        )
        super().__init__(span)

class PixelSpace(MultiDiscrete):
    '''
    A single pixel location for drag-and-drop operations
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(PixelSpace, self).__init__(
            [self.height, self.width])

class MultiPixelSpace(Box):
    '''
    A binary pixel mask for selecting multiple screen pixels simultaneously
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(MultiPixelSpace, self).__init__(
                low=False, high=True, shape=(height, width),
                dtype=numpy.bool)

class DiscreteChain(Discrete):
    def __init__(self, subspaces): #, ignore):
        self.subspaces = subspaces
        self.subspace_span = NameSpan()
        #self.chain_to_name_action = {}
        #self.name_action_to_chain = {}
        #self.name_range = {}
        #chain_length = 0
        for name, subspace in self.subspaces.items():
            if isinstance(subspace, Discrete):
                #sub_actions = range(subspace.n)
                if hasattr(subspace, 'span'):
                    shape = subspace.span
                else:
                    shape = subspace.n
            elif isinstance(subspace, MultiDiscrete):
                #sub_actions = product(range(n) for n in subspace.nvec)
                shape = subspace.nvec
            else:
                raise ValueError('DiscreteChain accepts '
                    'Discrete, MultiDiscrete spaces only')
            self.subspace_span.add_names(**{name:shape})
            #self.name_action_to_chain[name] = {}
            #range_start = chain_length
            #for i in sub_actions:
            #    if i not in ignore[name]:
            #        self.chain_to_name_action[chain_length] = name, i
            #        self.name_action_to_chain[name][i] = chain_length
            #        chain_length += 1
            #    
            #self.name_range[name] = range_start, chain_length
        
        super().__init__(self.subspace_span.total)
    
    def all_subspace_actions(self, name):
        #return self.name_action_to_chain[name].values()
        start, end = self.subspace_span.name_range(name)
        return list(range(start, end))
    
    def extract_subspace(self, x, name):
        #start, end = self.name_range[name]
        start, end = self.subspace_span.name_range(name)
        return x[..., start:end]
    
    def unravel(self, i):
        #for name, (start, end) in self.name_range.items():
        #    if i >= start and i < end:
        #        return name, i-start
        #raise ValueError('index %i less than 0 or greater than %i'%(i, end))
        return self.subspace_span.unravel(i)
    
    def ravel(self, name, *i):
        #return self.name_action_to_chain[name][i]
        return self.subspace_span.ravel(name, *i)
    
    def unravel_subspace(self, x):
        #result = {}
        #for name, (start, end) in self.name_range.items():
        #    result[name] = x[...,start:end]
        #return result
        #return self.subspace_span.unravel_all(x)
        result = {}
        for name in self.subspace_span.keys():
            start, end = self.subspace_span.name_range(name)
            result[name] = x[...,start:end]
        
        return result
    
    def ravel_subspace(self, x, out=None):
        result = []
        for name in self.subspace_span.keys():
            start, end = self.subspace_span.name_range(name)
            if out is not None:
                out[..., start:end] = x[name]
            else:
                result.append(name_x)
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, axis=-1)

class MultiScreenPixelSpace(Discrete):
    '''
    A single pixel location from multiple screens for drag-and-drop operations
    '''
    def __init__(self, screen_shapes): #, channels=1):
        #self.channels = channels
        #self.screen_data = OrderedDict()
        self.screen_span = NameSpan()
        #self.total_elements = 1
        self.screen_span.add_names(NO_OP=1, DESELECT=1, **screen_shapes)
        #for name, (h,w) in screen_shapes.items():
            #self.screen_data[name] = {
            #    'shape' : (h,w,channels),
            #    'elements' : h*w*channels,
            #    'offset' : self.total_elements
            #}
            #self.total_elements += h*w*channels
            #self.screen_span.add_name(name, shape)
        
        super(MultiScreenPixelSpace, self).__init__(
            self.screen_span.total)
    
    def ravel(self, name, *yxc):
        return self.screen_span.ravel(name, *yxc)
    
    def unravel(self, i):
        return self.screen_span.unravel(i)
    
    def ravel_maps(self, maps, out=None):
        result = []
        #for name, data in self.screen_data.items():
        for name in self.screen_span.keys():
            shape = maps[name].shape
            m = maps[name].reshape(*shape[:-3], -1)
            if out is not None:
                #start = data['offset']
                #end = data['offset']+data['elements']
                start, end = self.screen_span.name_range(name)
                out[..., start:end] = m
            else:
                result.append(m)
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, axis=-1)
    
    def unravel_maps(self, x):
        maps = {}
        #for name, data in self.screen_data.items():
        for name in self.screen_span.keys():
            #start = data['offset'] - 1 # gross gross
            #end = data['offset'] + data['elements'] - 1 # gross gross
            start, end = self.screen_span.name_range(name)
            shape = self.screen_span.get_shape(name)
            x_name = x[..., start:end]
            maps[name] = x_name.reshape(*x.shape[:-1], *shape)
        
        return maps
    
    '''
    def ravel_index(self, name, y, x, c=0):
        offset = self.screen_data[name]['offset']
        h,w,channels = self.screen_data[name]['shape']
        return offset + y * w*channels + x * channels + c
    
    def unravel_index(self, i):
        if i == 0:
            return 'NO_OP', 0, 0, 0
        ii = i
        for name, data in self.screen_data.items():
            if ii >= data['offset']:
                n = name
                continue
            break
        ii -= self.screen_data[n]['offset']
        h,w,channels = self.screen_data[n]['shape']
        y = ii // (w*channels)
        ii -= y * (w*channels)
        x = ii // channels
        c = ii - x * channels
        
        return n, y, x, c
        #
        #ii = i - 1
        #c = ii // self.total_pixels
        #ii -= c * self.total_pixels
        #for name, data in self.screen_data.items():
        #    if ii >= data['offset']:
        #        n = name
        #        continue
        #    break
        #ii -= self.screen_data[n]['offset']
        #width = self.screen_data[n]['shape'][1]
        #y = ii // width
        #x = ii % width
        #
        #return n, y, x, c
        #
    
    def ravel_maps(self, maps, out=None):
        result = []
        for name, data in self.screen_data.items():
            shape = maps[name].shape
            m = maps[name].reshape(*shape[:-3], -1)
            if out is not None:
                start = data['offset']
                end = data['offset']+data['elements']
                out[..., start:end] = m
            else:
                result.append(m)
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, axis=-1)
    
    def unravel_maps(self, x):
        # this is hella gross... what do we do about these no-op actions?
        x = x[..., -(self.n-1):]#.reshape(*x.shape[:-1], self.channels, -1)
        maps = {}
        for name, data in self.screen_data.items():
            start = data['offset'] - 1 # gross gross
            end = data['offset'] + data['elements'] - 1 # gross gross
            x_name = x[..., start:end]
            try:
                maps[name] = x_name.reshape(
                    *x.shape[:-1], *data['shape'])
            except:
                import pdb
                pdb.set_trace()
        
        return maps
    '''

class ActionModeSelectorSpace(Discrete):
    def __init__(self, action_blocks):
        self.action_blocks = action_blocks
        self.action_offsets = OrderedDict()
        self.total_actions = 1
        for name, a in self.action_blocks.items():
            self.action_offsets[name] = self.total_actions
            self.total_actions += len(a)
        
        super().__init__(self.total_actions)
    
    def ravel_index(self, name, a):
        return self.action_offsets[name] + self.action_blocks[name].index(a)
    
    def unravel_index(self, i):
        for name, offset in self.action_offsets.items():
            if i >= offset:
                i -= offset
            else:
                break
        return self.action_blocks[name][i]

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
