from gym.spaces import (
    Discrete,
    Tuple,
)
from ltron.gym.spaces import (
    OptionSpace,
    PixelSelectionSpace,
)
from ltron.gym.components.brick_env_component import BrickEnvComponent

class PickAndPlace2D(BrickEnvComponent):
    def __init__(self,
            scene_component,
            pick_width,
            pick_height,
            place_width,
            place_height):
        self.pick_width = pick_width
        self.pick_height = pick_height
        self.place_width = place_width
        self.place_height = place_height
        self.scene_component = scene_component
        self.scene_component.make_renderable()
        
        self.pick_frame_buffer = FrameBufferWrapper(
                self.pick_width, self.pick_height, anti_alias=False)
        self.place_frame_buffer = FrameBufferWrapper(
                self.place_width, self.place_height, anti_alias=False)
        
        polarity_space = Discrete(2)
        pick_space = PixelSelectionSpace(width, height)
        place_space = PixelSelectionSpace(width, height)
        
        self.action_space = Tuple((polarity_space, pick_space, place_space))
    
    def step(self, action):
        scene = self.scene_component.brick_scene
        
        polarity = action[0]
        
        # SOMETHING ABOUT A CAMERA
        
        pick_polarity = ['positive', 'negative'][polarity]
        pick_map = scene.snap_render(
                self.pick_frame_buffer, polarity = pick_polarity)
        pick_y, pick_x = action[1]
        pick_snap = pick_map[pick_y, pick_x]
        
        # SOMETHING ABOUT A CAMERA
        
        self.place_frame_bufer.enable()
        place_polarity = ['positive', 'negative'][1-polarity]
        place_map = scene.snap_render(
                self.place_frame_buffer, polarity = place_polarity)
        place_y, place_x = action[2]
        place_snap = place_map[place_y, place_x]
        
        scene.align_snaps(

class BrickManip2D(BrickEnvComponent):
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
