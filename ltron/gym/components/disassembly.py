import numpy

import gym.spaces as gym_spaces

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DisassemblyComponent(LtronGymComponent):
    def __init__(self,
        scene_component,
        check_collision,
    ):
        self.scene_component = scene_component
        self.check_collision = check_collision
        if self.check_collision:
            scene = self.scene_component.brick_scene
            assert scene.collision_checker is not None
        
        self.observation_space = Dict({'success': Discrete(2)})
    
    def reset(self):
        return {'success': False}
    
    def remove_instance(
        self,
        instance_index,
        snap_index=None,
        direction='attach',
    ):
        success = False
        if instance_index != 0:
            scene = self.scene_component.brick_scene
            if self.check_collision:
                instance = scene.instances[instance_index]
                snap = instance.get_snap(snap_index)
                collision = scene.check_snap_collision(
                    [instance], snap, direction)
                if not collision:
                    scene.remove_instance(instance_index)
                    success = True
            else:
                scene.remove_instance(instance_index)
                success = True
        
        return success


class IndexDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(InstanceDisassemblyComponent, self).__init__(
            scene_component, check_collision=False)
        self.max_instances = max_instances
        
        activate_space = Discrete(2)
        index_space = bg_spaces.SingleInstanceIndexSpace(self.max_instances)
        self.action_space = Tuple(activate_space, index_space)
    
    def step(self, action):
        activate, instance_index = action
        success = False
        if activate:
            success = self.remove_instance(instance_index)
        
        return {'success':success}, 0., False, None


class IndexDisassemblyWithCollisionComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(IndexDisassemblyWithCollisionComponent, self).__init__(
            scene_component, check_collision=True)
        self.max_instances = max_instances
        
        activate_space = Discrete(2)
        direction_space = Discrete(2)
        index_snap_space = bg_spaces.SingleSnapIndexSpace(self.max_snaps)
        self.action_space = Tuple(
            activate_space, direction_space, index_snap_space)
        
    def step(self, action):
        activate, direction, (instance_index, snap_index) = action
        success = False
        if activate and instance_index != 0:
            direction = ('detach', 'attach')[direction]
            success = self.remove_instance(
                instance_index, snap_index, direction=direction)

class PixelDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        scene_component,
        pos_snap_render_component,
        neg_snap_render_component,
        check_collision=False,
    ):
        
        super(PixelDisassemblyComponent, self).__init__(
            scene_component, check_collision)
        self.width = segmentation_component.width
        self.height = segmentation_component.height
        self.pos_snap_render = pos_snap_render_component,
        self.neg_snap_render = neg_snap_render_component,
        self.multi=multi
        self.enable_collision = enable_collision
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        direction_space = Discrete(2)
        pick_space = bg_spaces.SinglePixelSelectionSpace(
            self.width, self.height)
        
        self.action_space = Tuple(
            (activate_space, polarity_space, direction_space, pick_space))
    
    def step(self, action):
        activate, polarity, direction, (y, x) = action
        success = False
        if activate:
            polarity = '-+'[polarity]
            direction = ('detach', 'attach')[direction]
            if polarity == '+':
                pick_map = self.pos_snap_render.observation
            elif polarity == '-':
                pick_map = self.neg_snap_render.observation
            instance_index, snap_index = pick_map[y,x]
            if instance_index != 0:
                success = self.remove_instance(
                    instance_index, snap_index, direction=direction)
        
        return {'success':success}, 0., False, None
