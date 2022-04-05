import numpy

import gym.spaces as gym_spaces

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class VisibilityComponent(LtronGymComponent):
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
            multi=False,
            terminate_when_all_hidden=False):
        
        super(InstanceVisibilityComponent, self).__init__(
                scene_component = scene_component,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.max_instances = max_instances
        self.multi = multi
        
        if multi:
            self.action_space = bg_spaces.MultiInstanceSelectionSpace(
                    self.max_instances)
        else:
            self.action_space = bg_spaces.SingleInstanceIndexSpace(
                    self.max_instances)
    
    def step(self, action):
        if self.multi:
            for instance in numpy.nonzero(action)[0]:
                if instance != 0:
                    self.hide_instance(instance)
        else:
            self.hide_instance(action)
        
        return None, 0., self.check_terminal(), None

class PixelVisibilityComponent(VisibilityComponent):
    def __init__(self,
            width,
            height,
            scene_component,
            segmentation_component,
            terminate_when_all_hidden=False):
        
        super(PixelVisibilityComponent, self).__init__(
                scene_component = scene_component,
                terminate_when_all_hidden = terminate_when_all_hidden)
        self.width = width
        self.height = height
        self.segmentation_component = segmentation_component
        
        self.action_space = bg_spaces.PixelSelectionSpace(
                self.width, self.height)
    
    def step(self, action):
        y, x = numpy.nonzero(action)
        instance_map = self.segmentation_component.observation
        instance_indices = instance_map[y,x]
        for instance_index in instance_indices:
            self.hide_instance(instance_index)
        
        return None, 0., self.check_terminal(), None
