from collections import OrderedDict

from gymnasium.spaces import Dict, Discrete, MultiDiscrete

from supermecha import SuperMechaContainer, SuperMechaComponent

class ScreenCursor(SuperMechaComponent):
    def __init__(self,
        scene_component,
        snap_render_component,
        max_instances_per_scene,
        observe_selected=False,
    ):
        self.scene_component = scene_component
        self.snap_render_component = snap_render_component
        
        self.max_instances_per_scene = max_instances_per_scene
        
        self.screen_height = snap_render_component.height
        self.screen_width = snap_render_component.width
        
        if observe_selected:
            raise NotImplementedError
            #self.observation_space = Dict({'selected' : Discrete(2)})
        
        action_space = {
            'activate' : Discrete(2),
            'coords' : MultiDiscrete((self.screen_height, self.screen_width)),
            'polarity' : Discrete(2)
        }
        self.action_space = Dict(action_space)
    
    def reset(self, seed=None, rng=None):
        super().reset(seed=seed, rng=rng)
        self.y = 0
        self.x = 0
        self.polarity = 0
        
        return None, None
    
    def step(self, action):
        if action['activate']:
            self.y, self.x, self.polarity = self.action_to_coordinates(action)
    
        return None, 0., False, False, None
    
    def get_selected_snap(self):
        snap_map = self.snap_render_component.observe(polarity=self.polarity)
        instance, snap = snap_map[self.y,self.x]
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
        max_instances_per_scene,
        observe_selected=True,
        tile_height=16,
        tile_width=16,
    ):
        super().__init__(
            scene_component,
            snap_render_component,
            max_instances_per_scene,
            observe_selected=observe_selected,
        )
        
        self.tile_height = tile_height
        self.tile_width = tile_width
        
        assert self.screen_height % self.tile_height == 0
        assert self.screen_width % self.tile_width == 0
        
        self.block_height = self.screen_height // self.tile_height
        self.block_width = self.screen_width // self.tile_width
        
        action_space = {
            'activate' : Discrete(2),
            'block_coords' : MultiDiscrete(
                (self.block_height, self.block_width)),
            'tile_coords' : MultiDiscrete(
                (self.tile_height, self.tile_width)),
            'polarity' : Discrete(2)
        }
        self.action_space = Dict(action_space)
    
    def action_to_coordinates(self, action):
        by, bx = action['block_coords']
        ty, tx = action['tile_coords']
        y = by * self.tile_height + ty
        x = bx * self.tile_width + tx
        polarity = action['polarity']
        return y, x, polarity
    
    def coordinates_to_action(self, y, x):
        by = y // self.tile_height
        bx = x // self.tile_width
        ty = y % self.tile_height
        tx = x % self.tile_height
        return {
            'activate' : True,
            'block_coords' : (by, bx),
            'tile_coords' : (ty, tx),
        }
    
    def no_op_action(self):
        return {
            'activate' : 0,
            'block_coords' : (0,0),
            'tile_coords' : (0,0),
            'polarity' : 0,
        }

'''
class RelativeScreenCursor(ScreenCursor):
    def __init__(self,
        scene_component,
        snap_render_component,
        max_instances_per_scene,
        cursor_offsets = ((1,0),(0,1),(-1,0),(0,-1)),
    ):
        super().__init__(
            scene_component,
            snap_render_component,
            max_instances_per_scene,
        )
        
        self.cursor_offsets = ((0,0),) + cursor_offsets
        
        self.observation_space = Discrete(len(self.cursor_offsets))
    
    def action
    
    def no_op_action(self):
        return 0
'''

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
