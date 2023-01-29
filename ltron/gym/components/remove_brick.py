import numpy

from gymnasium.spaces import Discrete

from supermecha import SuperMechaComponent

class RemoveBrickComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        floating_brick_component=None,
        check_collision=True,
    ):
        self.scene_component = scene_component
        self.floating_brick_component = floating_brick_component
        self.check_collision = check_collision
        if self.check_collision:
            scene = self.scene_component.brick_scene
            assert scene.collision_checker is not None
        
    def disassemble(self, instance_index, snap_index=None):
        success = False
        removed_instance_index = 0
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
                    removed_instance_index = instance_index
            else:
                self.remove_instance(instance_index)
                success = True
                removed_instance_index = instance_index
        
        return success, removed_instance_index
    
    def remove_instance(self, instance_index):
        scene = self.scene_component.brick_scene

        if self.floating_brick_component is not None:
            instance = scene.instances[instance_index]
            brick_shape = str(instance.brick_shape)
            color = instance.color
            self.floating_brick_component.replace_brick(
                brick_shape, brick_color)
        
        scene.remove_instance(instance_index)
    
class CursorRemoveBrickComponent(RemoveBrickComponent):
    def __init__(self,
        scene_component,
        cursor_component,
        floating_brick_component=None,
        check_collision=False,
    ):
        super().__init__(
            scene_component,
            floating_brick_component=floating_brick_component,
            check_collision=check_collision,
        )
        self.cursor_component = cursor_component
        
        self.action_space = Discrete(2)
    
    def step(self, action):
        if action:
            #i, s = self.cursor_component.get_selected_snap()
            i, s = self.cursor_component.click_snap
            self.disassemble(i, s)
        
        return None, 0., False, False, {}
    
    def no_op_action(self):
        return 0
