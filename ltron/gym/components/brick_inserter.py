from gym.spaces import Dict, Discrete

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class HandspaceBrickInserter(LtronGymComponent):
    def __init__(
        self,
        handspace_component,
        workspace_component,
        class_ids,
        color_ids,
        max_instances
    ):
        self.handspace_component = handspace_component
        self.workspace_component = workspace_component
        self.brick_type_to_id = class_ids
        self.id_to_brick_type = {value:key for key, value in class_ids.items()}
        self.color_name_to_id = color_ids
        self.id_to_color_name = {value:key for key, value in color_ids.items()}
        self.max_instances = max_instances
        
        self.observation_space = Dict({'success' : Discrete(2)})
        self.action_space = Dict({
            'class_id' : Discrete(max(self.id_to_brick_type.keys())+1),
            'color_id' : Discrete(max(self.id_to_color_name.keys())+1),
        })
    
    def reset(self):
        return {'success' : False}
    
    def step(self, action):
        if len(self.workspace_component.brick_scene.instances):
            num_instances = max(
                self.workspace_component.brick_scene.instances.keys())
        else:
            num_instances = 0
        if num_instances > self.max_instances:
            success = False
        elif action['class_id'] in self.id_to_brick_type:
            scene = self.handspace_component.brick_scene
            scene.clear_instances()
            brick_type = self.id_to_brick_type[action['class_id']]
            color_name = self.id_to_color_name[action['color_id']]
            scene.add_instance(brick_type, color_name, scene.upright)
            success = True
        else:
            success = False
        
        return {'success' : success}, 0, False, {}
    
    def no_op_action(self):
        return {'class_id':0, 'color_id':0}
