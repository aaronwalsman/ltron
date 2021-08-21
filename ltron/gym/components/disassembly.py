import numpy

from gym.spaces import Dict, Discrete

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DisassemblyComponent(LtronGymComponent):
    def __init__(self,
        scene_component,
        handspace_component=None,
        check_collisions=False,
    ):
        self.scene_component = scene_component
        self.handspace_component = handspace_component
        self.check_collisions = check_collisions
        if self.check_collisions:
            scene = self.scene_component.brick_scene
            assert scene.collision_checker is not None
        
        self.observation_space = Dict({'success': Discrete(2)})
    
    def reset(self):
        return {'success': False}
    
    def disassemble(
        self,
        instance_index,
        snap_index=None,
        direction='push',
    ):
        success = False
        if instance_index != 0:
            scene = self.scene_component.brick_scene
            if self.check_collisions:
                instance = scene.instances[instance_index]
                snap = instance.get_snap(snap_index)
                collision = scene.check_snap_collision(
                    [instance], snap, direction)
                if not collision:
                    self.remove_instance(instance_index)
                    success = True
            else:
                self.remove_instance(instance_index)
                success = True
        
        return success
    
    def remove_instance(self, instance_index):
        scene = self.scene_component.brick_scene
        
        if self.handspace_component is not None:
            instance = scene.instances[instance_index]
            brick_type = str(instance.brick_type)
            color = instance.color
            handspace_scene = self.handspace_component.brick_scene
            handspace_scene.clear_instances()
            handspace_scene.add_instance(
                brick_type, color, handspace_scene.upright)
        
        scene.remove_instance(instance_index)

class IndexDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(InstanceDisassemblyComponent, self).__init__(
            scene_component, check_collisions=False)
        self.max_instances = max_instances
        
        activate_space = Discrete(2)
        index_space = bg_spaces.SingleInstanceIndexSpace(self.max_instances)
        self.action_space = Tuple(activate_space, index_space)
    
    def step(self, action):
        activate, instance_index = action
        success = False
        if activate:
            success = self.disassemble(instance_index)
        
        return {'success':success}, 0., False, None

class IndexDisassemblyWithCollisionComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(IndexDisassemblyWithCollisionComponent, self).__init__(
            scene_component, check_collisions=True)
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
            direction = ('pull', 'push')[direction]
            success = self.disassemble(
                instance_index, snap_index, direction=direction)

class PixelDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        scene_component,
        pos_snap_render_component,
        neg_snap_render_component,
        handspace_component=None,
        check_collisions=False,
    ):
        super(PixelDisassemblyComponent, self).__init__(
            scene_component,
            handspace_component=handspace_component,
            check_collisions=check_collisions,
        )
        assert (pos_snap_render_component.width ==
            neg_snap_render_component.width)
        assert (pos_snap_render_component.height ==
            neg_snap_render_component.height)
        self.width = pos_snap_render_component.width
        self.height = pos_snap_render_component.height
        self.pos_snap_render = pos_snap_render_component
        self.neg_snap_render = neg_snap_render_component
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        direction_space = Discrete(2)
        pick_space = bg_spaces.SinglePixelSelectionSpace(
            self.width, self.height)
        
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'direction':direction_space,
            'pick':pick_space
        })
    
    def step(self, action):
        activate = action['activate']
        polarity = action['polarity']
        direction = action['direction']
        y, x = action['pick']
        success = False
        if activate:
            polarity = '-+'[polarity]
            direction = ('pull', 'push')[direction]
            if polarity == '+':
                pick_map = self.pos_snap_render.observation
            elif polarity == '-':
                pick_map = self.neg_snap_render.observation
            instance_index, snap_index = pick_map[y,x]
            if instance_index != 0:
                success = self.disassemble(
                    instance_index, snap_index, direction=direction)
        
        return {'success':success}, 0., False, None
