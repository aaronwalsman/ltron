from itertools import product

import numpy

from collections import OrderedDict

from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict

import splendor.masks as masks

from supermecha.gym.spaces import IntegerMaskSpace, SE3Space, MultiSE3Space

from ltron.constants import (
    MAX_SNAPS_PER_BRICK,
    DEFAULT_WORLD_BBOX,
    SHAPE_CLASS_LABELS,
    NUM_SHAPE_CLASSES,
    COLOR_CLASS_LABELS,
    NUM_COLOR_CLASSES,
    MAX_INSTANCES_PER_SCENE,
    MAX_EDGES_PER_SCENE,
)
from ltron.name_span import NameSpan

# utility spaces ---------------------------------------------------------------
class DiscreteLayoutSpace(Discrete):
    '''
    A discrete action space with a built-in NameSpan layout.
    
    Used by:
    spaces.SymbolicSnapSpace (super)
    spaces.MultiScreenPixelSpace (super)
    spaces.DiscreteChain (super)
    '''
    def __init__(self, layout):
        self.layout = layout
        super().__init__(self.layout.total)
    
    def __getattr__(self, attr):
        return getattr(self.layout, attr)
    
    def __eq__(self, other):
        if isinstance(other, DiscreteLayoutSpace):
            return self.layout == other.layout
        else:
            return super().__eq__(other)

# observation spaces -----------------------------------------------------------
# moved to supermecha
#class ImageSpace(Box):
#    '''
#    A height x width x channels uint8 image.
#    
#    Used by:
#    components.render.ColorRenderComponent (observation_space)
#    '''
#    def __init__(self, width, height, channels=3):
#        self.width = width
#        self.height = height
#        self.channels = channels
#        super().__init__(
#            low=0,
#            high=255,
#            shape=(height, width, channels),
#            dtype=numpy.uint8,
#        )
#
#class BinaryMaskSpace(Box):
#    '''
#    A height x width bool array
#    
#    Used by:
#    spaces.MaskedTiledImageSpace (component)
#    '''
#    def __init__(self, width, height):
#        self.width = width
#        self.height = height
#        super().__init__(
#            low=0, high=1, shape=(height, width), dtype=numpy.bool)
#
#class InstanceMaskSpace(Box):
#    '''
#    A height x width array, where each pixel contains a long refering to
#    a segmentation index.
#    
#    Used by:
#    components.render.InstanceRenderComponent (observation_space)
#    '''
#    def __init__(self, width, height, max_instances=masks.NUM_MASKS-1):
#        self.width = width,
#        self.height = height
#        self.max_instances = max_instances
#        super().__init__(
#            low=0, high=max_instances, shape=(height, width), dtype=numpy.long)

class InstanceMaskSpace(IntegerMaskSpace):
    def __init__(self, width, height, max_instances=masks.NUM_MASKS-1):
        super().__init__(width, height, max_instances)

class SnapMaskSpace(Box):
    '''
    A height x width x 2 array, where each pixel contains a long refering to
    a brick instance index, and another long referring to a connection point
    index.
    
    Used by:
    components.render.SnapRenderComponent (observation_space)
    '''
    def __init__(self, width, height, max_id=masks.NUM_MASKS-1):
        self.width = width
        self.height = height
        self.max_id = max_id
        super().__init__(
            low=0, high=max_id, shape=(height, width, 2), dtype=numpy.long)

class MaskedTiledImageSpace(Dict):
    '''
    An ImageSpace representing a rendered color image, along with
    tile_width and tile_height parameters defining the size of individual tiles
    that the image should be broken into, as well as a BinaryMaskSpace
    representing which tiles in the image have changed since the previous frame.
    
    Used by:
    components.tile.DeduplicateTileMaskComponent (observation_space)
    '''
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

#class TimeStepSpace(Discrete):
#    '''
#    A discrete value to represent the current step index in an episode.
#    
#    Used by:
#    components.time_step.TimeStepComponent (observation_space)
#    '''
#    def __init__(self, max_steps):
#        self.max_steps = max_steps
#        super().__init__(self.max_steps)

class PhaseSpace(Discrete):
    '''
    A discrete value to represent the current phase in an episode.
    
    Used by:
    components.phase.PhaseSwitch (observation_space)
    '''
    def __init__(self, num_phases):
        self.num_phases = num_phases
        super().__init__(self.num_phases+1)

class InstanceIndexSpace(Discrete):
    '''
    A discrete value to represent selecting one of many instances in a scene
    Instances are 1-indexed, selecting 0 represents selecting nothing.
    
    UNUSED
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super().__init__(
                self.max_num_instances+1)

class SnapIndexSpace(MultiDiscrete):
    '''
    Two discrete values representing an instance index and a snap index
    in a scene.  Instances are 1-indexed, selecting 0 represents selecting
    nothing.
    
    UNUSED
    '''
    def __init__(self, max_num_instances):
        self.max_num_instances = max_num_instances
        super().__init__(
            [self.max_num_instances+1, MAX_SNAPS_PER_BRICK])

class SinglePixelSpace(MultiDiscrete):
    '''
    A single pixel location.
    
    UNUSED
    '''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super().__init__([self.height, self.width])

# moved to supermecha
#class SE3Space(Box):
#    '''
#    A single SE3 (4x4) transform
#    
#    UNUSED (possibly to be used in viewpoint)
#    '''
#    def __init__(self, world_bbox=DEFAULT_WORLD_BBOX):
#        shape = (4,4)
#        low = numpy.zeros(shape, dtype=numpy.float32)
#        low[:] = -1
#        low[:3,3] = world_bbox[0]
#        high = numpy.zeros(shape, dtype=numpy.float32)
#        high[:] = 1
#        high[:3,3] = world_bbox[1]
#        super().__init__(low=low, high=high, shape=shape)
#
#class MultiSE3Space(Box):
#    '''
#    Multiple SE3 (4x4) transforms
#    
#    Used by:
#    spaces.AssemblySpace (component)
#    '''
#    def __init__(self,
#        max_elements,
#        world_bbox=DEFAULT_WORLD_BBOX,
#    ):
#        self.max_elements = max_elements
#        shape = (max_elements,4,4)
#        low = numpy.zeros(shape, dtype=numpy.float32)
#        low[:] = -1
#        low[:,:3,3] = world_bbox[0]
#        low[:,3,3] = 1
#        high = numpy.zeros(shape, dtype=numpy.float32)
#        high[:] = 1
#        high[:,3,:3] = 0
#        high[:,:3,3] = world_bbox[1]
#        super().__init__(low=low, high=high, shape=shape)

class EdgeSpace(Box):
    '''
    Multiple connections between snaps.
    
    Used by:
    spaces.AssemblySpace (component)
    '''
    def __init__(self, max_instances, max_edges):
        low = numpy.zeros((4, max_edges), dtype=numpy.long)
        high = numpy.zeros((4, max_edges), dtype=numpy.long)
        high[:2,:] = max_instances
        high[2:,:] = MAX_SNAPS_PER_BRICK-1
        super().__init__(
            low=low,
            high=high,
            shape=(4, max_edges),
            dtype=numpy.long,
        )

class AssemblySpace(Dict):
    '''
    A variable length vector of instances, each with a class-label,
    color and pose, along with a list of edges connections.
    
    Used by:
    components.assembly.AssemblyComponent (component)
    '''
    def __init__(
        self,
        max_instances=None,
        max_edges=None,
        world_bbox=DEFAULT_WORLD_BBOX,
    ):
        if max_instances is None:
            max_instances = MAX_INSTANCES_PER_SCENE
        self.max_instances = max_instances
        if max_edges is None:
            max_edges = MAX_EDGES_PER_SCENE
        self.max_edges = max_edges
        
        self.space_dict = {
            'shape' : Box(
                low=0,
                high=NUM_SHAPE_CLASSES,
                shape=(max_instances+1,),
                dtype=numpy.long,
            ),
            'color' : Box(
                low=0,
                high=NUM_COLOR_CLASSES,
                shape=(max_instances+1,),
                dtype=numpy.long,
            ),
            'pose' : MultiSE3Space(max_instances+1, world_bbox),
            'edges' : EdgeSpace(max_instances, max_edges),
        }
        
        super().__init__(self.space_dict)
    
    def from_scene(self, scene):
        return scene.get_assembly()
            #self.shape_ids,
            #self.color_ids,
            #self.max_instances,
            #self.max_edges,
        #)

class MaskedAssemblySpace(Box):
    '''
    A mask defining which instances have changed in an assembly.
    
    Used by:
    components.assembly.DeduplicateAssemblyComponent (observation_space)
        [in progress]
    '''
    def __init__(self, max_instances):
        super().__init__(
            low=0, high=1, shape=(max_instances+1), dtype=numpy.bool)

class MultiScreenInstanceSnapSpace(MultiDiscrete):
    '''
    A single instance/snap from one of many screens.
    
    Used by:
    components.cursor.SymbolicCursor (observation_space)
    '''
    def __init__(self, screens, max_instances):
        self.screens = screens
        self.screen_index = {}
        self.screen_index['NO_OP'] = 0
        self.screen_index.update(
            {screen:i for i, screen in enumerate(screens, start=1)}
        )
        self.max_instances = max_instances
        
        super().__init__(
            [len(self.screen_index), max_instances+1, MAX_SNAPS_PER_BRICK]
        )
    
    def ravel(self, *coords):
        return self.screen_index[coords[0]], *coords[1:]

#class MultiScreenSinglePixelSpace(DiscreteLayoutSpace):
#    '''
#    '''
#    def __init__(self, coarse_span, fine_shape):
#        layout = NameSpan('coarse':coarse_span, 'fine':fine_shape)
#        super().__init__(layout)

# action spaces ----------------------------------------------------------------
class SymbolicSnapSpace(DiscreteLayoutSpace):
    '''
    A selector for a single snap from one of multiple screens
    
    Used by:
    components.cursor.SymbolicCursor
    '''
    def __init__(self, max_instances):
        self.max_instances = max_instances
        layout = NameSpan(
            NO_OP=1, **{name:(max_i+1, MAX_SNAPS_PER_BRICK)
            for name, max_i in max_instances.items()}
        )
        super().__init__(layout)

class MultiScreenPixelSpace(DiscreteLayoutSpace):
    '''
    A single pixel location from multiple screens for drag-and-drop operations.
    
    Used by:
    components.cursor.MultiViewCursor (action_space)
    '''
    def __init__(self, screen_shapes, include_no_op=False): #, channels=1):
        layout = NameSpan()
        if include_no_op:
            layout.add_names(NO_OP=1)
        #layout.add_names(DESELECT=1, **screen_shapes)
        #for name in screen_shapes.keys():
        #    layout.add_names(**{'DESELECT_%s'%name:1})
        screen_layout = {
            screen : NameSpan(deselect=1, screen=shape)
            for screen, shape in screen_shapes.items()
        }
        layout.add_names(**screen_layout)
        
        super().__init__(layout)
    
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
    '''
    A chain of multiple other Discrete and MultiDiscrete spaces laid out into
    one large Discrete space using a NameSpan layout.
    
    Used by:
    envs.ltron_env.LtronEnv (action_space)
    '''
    def __init__(self, subspaces, strict=True):
        self.subspaces = subspaces
        layout = NameSpan()
        
        def get_shape(name, space, strict=True):
            if isinstance(space, DiscreteLayoutSpace):
                return space.layout
            elif isinstance(space, Discrete):
                return space.n
            elif isinstance(space, MultiDiscrete):
                return space.nvec
            elif isinstance(space, Dict):
                shape = NameSpan()
                for subname, subspace in space.items():
                    subshape = get_shape(subname, subspace, strict=strict)
                    if subshape is not None:
                        shape.add_names(**{subname:subshape})
                return shape
            else:
                if strict:
                    raise ValueError('DiscreteChain accepts '
                        'Discrete, MultiDiscrete spaces only, got: %s for %s'%
                        (space, name))
                else:
                    return None
            
        for name, subspace in self.subspaces.items():
            shape = get_shape(name, subspace, strict=strict)
            if shape is not None:
                layout.add_names(**{name:shape})
        
        super().__init__(layout)

class BrickShapeColorSpace(MultiDiscrete):
    '''
    A selector for choosing a brick shape/color.
    
    Used by:
    components.brick_inserter.BrickInserter (action_space)
    '''
    def __init__(self, shape_ids, color_ids):
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.num_shapes = max(shape_ids.values())+1
        self.num_colors = max(color_ids.values())+1
        super().__init__((self.num_shapes, self.num_colors))

class BrickShapeColorPoseSpace(Dict):
    def __init__(self, shape_ids, color_ids, world_bbox=DEFAULT_WORLD_BBOX):
        super().__init__({
            'shape_color' : BrickShapeColorSpace(shape_ids, color_ids),
            'pose' : SE3Space(world_bbox=world_bbox),
        })
