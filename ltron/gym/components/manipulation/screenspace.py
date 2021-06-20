from gym.spaces import (
    Discrete,
    Tuple,
)
from ltron.gym.spaces import (
    SingleSnapIndexSpace,
    SinglePixelSelectionSpace,
)
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class PickAndPlace2D(LtronGymComponent):
    def __init__(self,
        scene_component,
        neg_snap_render_component,
        pos_snap_render_component,
    ):
        self.scene_component = scene_component
        self.neg_snap_render_component = neg_snap_render_component
        self.pos_snap_render_component = pos_snap_render_component
        
        width = pos_snap_render_component.width
        height = pos_snap_render_component.height
        assert neg_snap_render_component.width == width
        assert neg_snap_render_component.height == height
        
        pick_polarity_space = Discrete(2)
        pick_space = SinglePixelSelectionSpace(width, height)
        place_space = SinglePixelSelectionSpace(width, height)
        self.action_space = Tuple(
            (pick_polarity_space, pick_space, place_space))
        
        return None, 0., False, {}
    
    def step(self, action):
        scene = self.scene_component.brick_scene
        
        polarity = action[0]
        pick_y, pick_x = action[1]
        place_y, place_x = action[2]
        
        if polarity == 0:
            pick_map = self.neg_snap_render_component.observation
            place_map = self.pos_snap_render_component.observation
        else:
            pick_map = self.pos_snap_render_component.observation
            place_map = self.neg_snap_render_component.observation
        
        pick = pick_map[pick_y, pick_x]
        place = pick_map[pick_y, pick_x]
        
        scene.pick_and_place_snaps(pick, place)

class BrickManip2D(LtronGymComponent):
    def __init__(self,
            scene_component,
            connection_point_component):
        self.scene_component = scene_component
        self.connection_point_component = connection_point_component
    
    def step(self, action):
        # get the connection point at the indicated position
        position = action['position']
        connector = self.connector_component.get_connector_at_position(
                position)
        
        mode = action['mode']
        # translate a brick by dragging one connection point onto another
        if mode == 'translate':
            pass
        
        # rotate a brick about a connection point
        elif mode == 'rotate':
            pass
