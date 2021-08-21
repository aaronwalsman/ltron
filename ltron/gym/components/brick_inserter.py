from gym.spaces import Dict, Discrete

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class HandspaceBrickInserter(LtronGymComponent):
    def __init__(self, handspace_component, class_ids, colors):
        self.handspace_component = handspace_component
        self.brick_type_to_id = class_ids
        self.id_to_brick_type = {value:key for key, value in class_ids.items()}
        self.colors = colors
        
        self.observation_space = Dict({'success' : Discrete(2)})
        self.action_space = Dict({
            'class_id' : Discrete(max(self.id_to_brick_type.keys())+1),
            'color' : Discrete(len(colors)),
        })
    
    def reset(self):
        return {'success' : False}
    
    def step(self, action):
        success = False
        if action['class_id'] in self.id_to_brick_type:
            scene = self.handspace_component.brick_scene
            scene.clear_instances()
            brick_type = self.id_to_brick_type[action['class_id']]
            color = self.colors[action['color']]
            scene.add_instance(brick_type, color, scene.upright)
            success = True
        
        return {'success' : success}, 0, False, {}
