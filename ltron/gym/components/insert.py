import numpy

from gymnasium.spaces import MultiDiscrete

from supermecha import SuperMechaComponent

from ltron.constants import (
    SHAPE_CLASS_LABELS,
    COLOR_CLASS_LABELS,
    NUM_SHAPE_CLASSES,
    NUM_COLOR_CLASSES,
)

class InsertBrickComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        place_above_scene_offset=48,
    ):
        self.scene_component = scene_component
        shape_class_labels = SHAPE_CLASS_LABELS
        num_shape_classes = NUM_SHAPE_CLASSES
        
        color_class_labels = COLOR_CLASS_LABELS
        num_color_classes = NUM_COLOR_CLASSES
        
        self.id_to_shape = {
            value:key for key,value in shape_class_labels.items()}
        self.id_to_color = {
            value:key for key,value in color_class_labels.items()}
        self.place_above_scene_offset = place_above_scene_offset
        assert 0 not in self.id_to_shape
        assert 0 not in self.id_to_color
        
        self.action_space = MultiDiscrete(
            (num_shape_classes, num_color_classes))
    
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
