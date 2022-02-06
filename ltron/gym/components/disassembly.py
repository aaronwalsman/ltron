import numpy

from gym.spaces import Dict, Discrete

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DisassemblyComponent(LtronGymComponent):
    def __init__(self,
        max_instances,
        scene_component,
        hand_scene_component=None,
        check_collision=False,
    ):
        self.scene_component = scene_component
        self.hand_scene_component = hand_scene_component
        self.check_collision = check_collision
        if self.check_collision:
            scene = self.scene_component.brick_scene
            assert scene.collision_checker is not None
        
        self.observation_space = Dict({
            'success': Discrete(2),
            'instance_id': bg_spaces.SingleInstanceIndexSpace(max_instances),
        })
    
    def reset(self):
        return {'success': False, 'instance_id': 0}
    
    def disassemble(
        self,
        instance_index,
        snap_index=None,
    ):
        success = False
        instance_id = 0
        if instance_index != 0:
            scene = self.scene_component.brick_scene
            if self.check_collision:
                instance = scene.instances[instance_index]
                snap = instance.snaps[snap_index]
                collision = scene.check_snap_collision(
                    [instance], snap)
                if not collision:
                    self.remove_instance(instance_index)
                    success = True
                    instance_id =instance_index
            else:
                self.remove_instance(instance_index)
                success = True
                instance_id = instance_index
        
        return success, instance_id
    
    def remove_instance(self, instance_index):
        scene = self.scene_component.brick_scene
        
        if self.hand_scene_component is not None:
            instance = scene.instances[instance_index]
            brick_shape = str(instance.brick_shape)
            color = instance.color
            handspace_scene = self.hand_scene_component.brick_scene
            handspace_scene.clear_instances()
            handspace_scene.add_instance(
                brick_shape, color, handspace_scene.upright)
        
        scene.remove_instance(instance_index)

class IndexDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(InstanceDisassemblyComponent, self).__init__(
            max_instances, scene_component, check_collision=False)
        self.max_instances = max_instances
        
        activate_space = Discrete(2)
        index_space = bg_spaces.SingleInstanceIndexSpace(self.max_instances)
        self.action_space = Tuple(activate_space, index_space)
    
    def step(self, action):
        activate, instance_index = action
        success = False
        instance_id = 0
        if activate:
            success, instance_id = self.disassemble(instance_index)
        
        return {'success':success, 'instance_id':instance_id}, 0., False, None
    
    def no_op_action(self):
        return (0, 0)

class IndexDisassemblyWithCollisionComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
    ):
        super(IndexDisassemblyWithCollisionComponent, self).__init__(
            max_instances, scene_component, check_collision=True)
        self.max_instances = max_instances
        
        activate_space = Discrete(2)
        index_snap_space = bg_spaces.SingleSnapIndexSpace(self.max_snaps)
        self.action_space = Tuple(activate_space, index_snap_space)
        
    def step(self, action):
        activate, (instance_index, snap_index) = action
        success = False
        instance_id = 0
        if activate and instance_index != 0:
            success, instance_id = self.disassemble(instance_index, snap_index)
        
        return {'success':success, 'instance_id':instance_id}, 0., False, None
    
    def no_op_action(self):
        return (0, 0)

class PixelDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
        pos_snap_render_component,
        neg_snap_render_component,
        hand_scene_component=None,
        check_collision=False,
    ):
        super(PixelDisassemblyComponent, self).__init__(
            max_instances,
            scene_component,
            hand_scene_component=hand_scene_component,
            check_collision=check_collision,
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
        pick_space = bg_spaces.SinglePixelSelectionSpace(
            self.width, self.height)
        
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'pick':pick_space
        })
    
    def step(self, action):
        activate = action['activate']
        polarity = action['polarity']
        y, x = action['pick']
        success = False
        instance_id = 0
        if activate:
            polarity = '-+'[polarity]
            if polarity == '+':
                pick_map = self.pos_snap_render.observation
            elif polarity == '-':
                pick_map = self.neg_snap_render.observation
            instance_index, snap_index = pick_map[y,x]
            if instance_index != 0:
                success, instance_id = self.disassemble(
                    instance_index, snap_index)
        
        return {'success':success, 'instance_id':instance_id}, 0., False, None
    
    def no_op_action(self):
        return {'activate':0, 'polarity':0, 'pick':numpy.array([0,0])}

class CursorDisassemblyComponent(DisassemblyComponent):
    def __init__(self,
        max_instances,
        scene_component,
        cursor_component,
        hand_scene_component=None,
        check_collision=False,
    ):
        super(CursorDisassemblyComponent, self).__init__(
            max_instances,
            scene_component,
            hand_scene_component=hand_scene_component,
            check_collision=check_collision,
        )
        self.cursor_component = cursor_component
        
        self.action_space = Discrete(2)
    
    def step(self, action):
        success = False
        instance_id = 0
        if action:
            instance_id = self.cursor_component.instance_id
            snap_id = self.cursor_component.snap_id
            if instance_id != 0:
                success, instance_id = self.disassemble(instance_id, snap_id)
        
        return {'success':success, 'instance_id':instance_id}, 0., False, None
    
    def no_op_action(self):
        return 0
