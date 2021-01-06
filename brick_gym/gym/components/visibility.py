import brick_gym.gym.spaces as bg_spaces
from brick_gym.envs.components.brick_env_component import BrickEnvComponent

class VisibilityComponent(BrickEnvComponent):
    def __init__(self,
                renderer_key='renderer',
                terminate_when_all_hidden=False):
        self.renderer_key = renderer_key
        self.terminate_when_all_hidden = terminate_when_all_hidden
    
    def hide_instance(self, state, instance_index):
        if instance_index != 0:
            renderer = state[self.renderer_key]
            instance_name = 'instance_%i'%(instance_index)
            if renderer.instance_exists(instance_name):
                renderer.hide_instance(instance_name)
    
    def check_terminal(self, state):
        if self.terminate_when_all_hidden:
            renderer = state[self.renderer_key]
            all_hidden = all(renderer.instance_hidden(instance)
                    for instance in renderer.list_instances())
            #print('hidden', [renderer.instance_hidden(instance)
            #        for instance in renderer.list_instances()])
            return all_hidden
            
        else:
            return False

class InstanceVisibilityComponent(VisibilityComponent):
    def __init__(self,
            max_num_instances,
            action_key='visibility',
            renderer_key='renderer',
            terminate_when_all_hidden=False):
        
        super(InstanceVisibilityComponent, self).__init__(
                renderer_key = renderer_key,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.max_num_instances = max_num_instances
        self.action_key = action_key
        self.terminate_when_all_hidden
    
    def update_action_space(self, action_space):
        action_space[self.action_key] = bg_spaces.InstanceSelectionSpace(
                self.max_num_instances)
    
    def update_state(self, state, action):
        instance_index = action[self.action_key] + 1
        self.hide_instance(state, instance_index)

class PixelVisibilityComponent(VisibilityComponent):
    def __init__(self,
            height,
            width,
            action_key='visibility',
            instance_map_key='instance_map',
            renderer_key='renderer',
            terminate_when_all_hidden=False):
        
        super(PixelVisibilityComponent, self).__init__(
                renderer_key = renderer_key,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.height = height
        self.width = width
        self.action_key = action_key
        self.instance_map_key = instance_map_key
    
    def update_action_space(self, action_space):
        action_space[self.action_key] = bg_spaces.PixelSelectionSpace(
                self.height, self.width)
    
    def update_state(self, state, action):
        y, x = action[self.pixel_key]
        instance_map = state[self.instance_map_key]
        instance_index = instance_map[y,x]
        self.hide_instance(state, instance_index)
