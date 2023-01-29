from collections import OrderedDict

from gymnasium.spaces import Dict, Discrete, MultiDiscrete

from supermecha import SuperMechaContainer, SuperMechaComponent
from supermecha.gym.spaces import IgnoreSpace

class CursorComponent(SuperMechaComponent):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
        action_space = OrderedDict()
        action_space['button'] = Discrete(3)
        action_space['click'] = MultiDiscrete((self.height, self.width))
        action_space['release'] = MultiDiscrete((self.height, self.width))
        self.action_space = Dict(action_space)
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=options)
        self.button = 0
        self.click = 0,0
        self.release = 0,0
        
        return None, {}
    
    def step(self, action):
        self.button = action['button']
        self.click = action['click']
        self.release = action['release']
        return None, 0., False, False, {}
    
    def no_op_action(self):
        return {
            'button' : 0
            'click' : (0,0),
            'release' : (0,0),
        }

'''
class ScreenCursor(SuperMechaComponent):
    def __init__(self,
        scene_component,
        snap_render_component,
        observe_selected=False,
        observe_snap_map=False,
    ):
        self.scene_component = scene_component
        self.snap_render_component = snap_render_component
        
        self.screen_height = snap_render_component.height
        self.screen_width = snap_render_component.width
        
        self.observe_selected = observe_selected
        self.observe_snap_map = observe_snap_map
        
        observation_space = {}
        if observe_selected:
            raise NotImplementedError
            #self.observation_space = Dict({'selected' : Discrete(2)})
        
        if len(observation_space):
            self.observation_space = Dict(observation_space)
        
        action_space = OrderedDict()
        action_space['activate'] = Discrete(2)
        action_space['coords'] = MultiDiscrete(
            (self.screen_height, self.screen_width))
        action_space['polarity'] = Discrete(2)
        self.action_space = CursorActionSpace(action_space)
    
    def compute_observation(self):
        self.snap_map, _ = self.snap_render_component.observe(
            polarity=self.polarity)
        self.observation = {}
        if self.observe_snap_map:
            self.observation['snap_map'] = self.snap_map
        if not len(self.observation):
            self.observation = None
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=options)
        self.y = 0
        self.x = 0
        self.polarity = 0
        
        self.compute_observation()
        return self.observation, None
    
    def step(self, action):
        if action['activate']:
            self.y, self.x, self.polarity = self.action_to_coordinates(action)
        
        self.compute_observation()
        return self.observation, 0., False, False, None
    
    def get_selected_snap(self):
        instance, snap = self.snap_map[self.y,self.x]
        return instance, snap
    
    def actions_to_select_snap(self, instance, snap):
        actions = []
        snap = self.scene_component.brick_scene.instances[instance].snaps[snap]
        snap_map = self.snap_render_component.observe(polarity=snap.polarity)
        ys, xs = numpy.where(
            (snap_map[:,:,0] == instance) & (snap_map[:,:,1] == snap)
        )
        for y, x in zip(ys, xs):
            actions.append(self.coordinates_to_action(y, x))
        
        return actions
    
    def action_to_coordinates(self, action):
        y, x = action['coords']
        polarity = action['polarity']
        return y, x, polarity
    
    def coordinates_to_action(self, y, x):
        return {'activate':True, 'coords':(y,x)}
    
    def no_op_action(self):
        return {
            'activate' : 0,
            'coords' : (0,0),
            'polarity' : 0,
        }
    
    def get_state(self):
        return (self.y, self.x, self.polarity)
    
    def set_state(self, state):
        self.y, self.x, self.polarity = state

class TiledScreenCursor(ScreenCursor):
    def __init__(self,
        scene_component,
        snap_render_component,
        observe_selected=True,
        tile_height=16,
        tile_width=16,
        observe_snap_map=False,
    ):
        super().__init__(
            scene_component,
            snap_render_component,
            observe_selected=observe_selected,
            observe_snap_map=observe_snap_map,
        )
        
        self.tile_height = tile_height
        self.tile_width = tile_width
        
        assert self.screen_height % self.tile_height == 0
        assert self.screen_width % self.tile_width == 0
        
        self.block_height = self.screen_height // self.tile_height
        self.block_width = self.screen_width // self.tile_width
        
        action_space = OrderedDict()
        action_space['activate'] = Discrete(2)
        action_space['block_coords'] = MultiDiscrete(
                (self.block_height, self.block_width))
        action_space['tile_coords'] = MultiDiscrete(
                (self.tile_height, self.tile_width))
        action_space['polarity'] = Discrete(2)
        self.action_space = CursorActionSpace(action_space)
    
    def action_to_coordinates(self, action):
        by, bx = action['block_coords']
        ty, tx = action['tile_coords']
        y = by * self.tile_height + ty
        x = bx * self.tile_width + tx
        polarity = action['polarity']
        return y, x, polarity
    
    def coordinates_to_action(self, y, x, p):
        by = y // self.tile_height
        bx = x // self.tile_width
        ty = y % self.tile_height
        tx = x % self.tile_height
        return {
            'activate' : True,
            'block_coords' : (by, bx),
            'tile_coords' : (ty, tx),
            'polarity' : p,
        }
    
    def no_op_action(self):
        return {
            'activate' : 0,
            'block_coords' : (0,0),
            'tile_coords' : (0,0),
            'polarity' : 0,
        }

class PickAndPlaceCursor(SuperMechaContainer):
    
    def __init__(self,
        pick_component,
        place_component,
    ):
        components = OrderedDict()
        components['pick'] = pick_component
        components['place'] = place_component
        
        super().__init__(components)
    
    def actions_to_pick_snap(self, instance, snap):
        return self.components['pick'].actions_to_select_snap(instance, snap)
    
    def actions_to_place_snap(self, instance, snap):
        return self.components['place'].actions_to_select_snap(instance, snap)
'''
