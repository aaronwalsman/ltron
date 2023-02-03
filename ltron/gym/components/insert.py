import numpy

from gymnasium.spaces import MultiDiscrete

from supermecha import SuperMechaComponent

class InsertBrickComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        shape_ids,
        color_ids,
        place_above_scene_offset=48,
    ):
        self.scene_component = scene_component
        self.shape_ids = shape_ids
        self.id_to_shape = {value:key for key,value in self.shape_ids.items()}
        self.color_ids = color_ids
        self.id_to_color = {value:key for key,value in self.color_ids.items()}
        self.place_above_scene_offset = place_above_scene_offset
        assert 0 not in self.id_to_shape
        assert 0 not in self.id_to_color
        
        num_shapes = len(self.shape_ids)
        num_colors = len(self.color_ids)
        self.action_space = MultiDiscrete((num_shapes+1, num_colors+1))
    
    def step(self, action):
        scene = self.scene_component.brick_scene
        shape_id, color_id = action
        if shape_id != 0 and color_id != 0:
            shape_name = self.id_to_shape[shape_id]
            color_name = self.id_to_color[color_id]
            instance = scene.add_instance(shape_name, color_name, scene.upright)
            scene.place_above_scene(
                [instance], offset=self.place_above_scene_offset)
        
        return None, 0., False, False, {}
    
    def no_op_action(self):
        return numpy.array([0,0])
