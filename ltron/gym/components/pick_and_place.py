import numpy

from gymnasium.spaces import Discrete

from supermecha import SuperMechaComponent

from ltron.geometry.utils import unscale_transform

class PickAndPlaceComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        check_collision=True,
        place_above_scene_offset=48,
    ):
        self.scene_component = scene_component
        self.check_collision = check_collision
        self.place_above_scene_offset = place_above_scene_offset
        if self.check_collision:
            scene = self.scene_component.brick_scene
            assert scene.collision_checker is not None
    
    def get_instance_snap(self, instance_id, snap_id):
        if int(instance_id) != 0:
            scene = self.scene_component.brick_scene
            instance = scene.instances.get(instance_id, None)
            if instance is None:
                return None, None
            else:
                if snap_id < len(instance.snaps):
                    snap = instance.snaps[snap_id]
                    return instance, snap
                else:
                    return None, None
        else:
            return None, None
    
    def pick_and_place(self,
        pick_instance,
        pick_snap,
        place_instance,
        place_snap
    ):
        success = False
        
        pick_instance, pick_snap = self.get_instance_snap(
            pick_instance, pick_snap)
        
        place_instance, place_snap = self.get_instance_snap(
            place_instance, place_snap)
        
        scene = self.scene_component.brick_scene
        
        if pick_instance is None:
            return False
        
        if pick_instance == place_instance:
            return False
        
        #overlay_instance = self.overlay_brick_component.get_instance()
        #if (overlay_instance is not None and
        #    overlay_instance == place_instance
        #):
        #    return False
        
        #pick_overlay = self.overlay_brick_component.is_overlaid(pick_instance)
        
        #if pick_overlay:
        #    original_transform = pick_instance.transform.copy()
        #    unscaled_transform = unscale_transform(original_transform)
        #    scene.move_instance(pick_instance, unscaled_transform)
        #    success = scene.pick_and_place_snap(
        #        pick_snap,
        #        place_snap,
        #        check_pick_collision = False,
        #        check_place_collision = self.check_collision,
        #        ignore_collision_instances = [overlay_instance],
        #    )
        #    if not success:
        #        scene.move_instance(pick_instance, original_transform)
        #    self.overlay_brick_component.set_overlay_instance(0)
        if place_instance is None:
            if self.check_collision:
                if scene.check_snap_collision([pick_instance], pick_snap):
                    return False
            scene.place_above_scene(
                [pick_instance], offset=self.place_above_scene_offset)
        else:
            success = scene.pick_and_place_snap(
                pick_snap,
                place_snap,
                check_pick_collision=self.check_collision,
                check_place_collision=self.check_collision,
                #ignore_collision_instances = [overlay_instance],
            )
        #else:
        #    self.overlay_brick_component.set_overlay_instance(
        #        int(pick_instance))
        #    success = True
        
        return success

class CursorPickAndPlaceComponent(PickAndPlaceComponent):
    def __init__(self,
        scene_component,
        cursor_component,
        #overlay_brick_component=None,
        check_collision=False,
        truncate_on_failure=False,
    ):
        super().__init__(
            scene_component,
            #overlay_brick_component=overlay_brick_component,
            check_collision=check_collision,
        )
        self.cursor_component = cursor_component
        self.truncate_on_failure = truncate_on_failure
        
        self.action_space = Discrete(2)
    
    def step(self, action):
        truncate = False
        if action:
            ci, cs = self.cursor_component.click_snap
            ri, rs = self.cursor_component.release_snap
            success = self.pick_and_place(ci, cs, ri, rs)
            if self.truncate_on_failure and not success:
                truncate = True
        
        return None, 0., False, truncate, {}
    
    def no_op_action(self):
        return 0
