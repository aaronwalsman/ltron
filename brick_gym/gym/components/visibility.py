import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class VisibilityComponent(BrickEnvComponent):
    def __init__(self,
                scene_component,
                terminate_when_all_hidden=False):
        self.scene_component = scene_component
        self.terminate_when_all_hidden = terminate_when_all_hidden
    
    def hide_instance(self, instance_index):
        if instance_index != 0:
            scene = self.scene_component.brick_scene
            scene.hide_instance(instance_index)
    
    def check_terminal(self):
        if self.terminate_when_all_hidden:
            scene = self.scene_component.brick_scene
            all_hidden = all(scene.instance_hidden(instance)
                    for instance in scene.instances)
            return all_hidden
            
        else:
            return False
    
    def get_state(self):
        scene = self.scene_component.brick_scene
        return {str(instance):scene.instance_hidden(instance)
                for instance in scene.instances}
    
    def set_state(self, state):
        scene = self.scene_component.brick_scene
        for instance, hide in state.items():
            if hide:
                scene.hide_instance(instance)
            else:
                scene.show_instance(instance)

class InstanceVisibilityComponent(VisibilityComponent):
    def __init__(self,
            max_instances,
            scene_component,
            terminate_when_all_hidden=False):
        
        super(InstanceVisibilityComponent, self).__init__(
                scene_component = scene_component,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.max_instances = max_instances
        
        self.action_space = bg_spaces.InstanceSelectionSpace(
                self.max_instances)
    
    def step(self, action):
        self.hide_instance(action)
        
        return None, 0., self.check_terminal(), None

class PixelVisibilityComponent(VisibilityComponent):
    def __init__(self,
            width,
            height,
            scene_component,
            segmentation_component,
            terminate_when_all_hidden=False):
        
        super(InstanceVisibilityComponent, self).__init__(
                scene_component = scene_component,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.width = width
        self.height = height
        self.mask_render_component = mask_render_component
        
        self.action_space = bg_spaces.PixelSelectionSpace(
                self.width, self.height)
    
    def step(self, action):
        x, y = action[self.pixel_key]
        instance_map = self.mask_render_component.segmentation
        instance_index = instance_map[y,x]
        self.hide_instance(instance_index)
        
        return None, 0., self.check_terminal(), None
