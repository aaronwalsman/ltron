import numpy

from gym.spaces import Discrete, Dict

from supermecha import SuperMechaComponent

#from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.exceptions import ThisShouldNeverHappen

class RemoveBrickComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        cursor_component,
        check_collision=False,
    ):
        self.scene_component = scene_component
        self.cursor_component = cursor_component
        self.check_collision = check_collision
        
        self.action_space = Discrete(2)
        
    def step(self, action):
        
        # if no action was taken, return
        if not action:
            return None, 0., False, False, None
        
        # get the pick scene/instance/snap
        pick_i, pick_s = self.pick_cursor_component.get_selected_snap()
        
        # if nothing is picked, return
        if pick_i == 0:
            return None, 0., False, False, None
        
        # pick and remove
        self.pick_and_remove(pick_i, pick_s)
        
        return None, 0., False, False, None
    
    def pick_and_remove(self, pick_i, pick_s):
        
        # get the pick scene and instance
        pick_scene = self.scene_component.brick_scene
        pick_instance = pick_scene.instances[pick_i]
        
        # check if the picked brick can be removed without colliding
        if self.check_collision:
            pick_snap = pick_scene.snap_tuple_to_snap((pick_i, pick_s))
            if pick_snap is None:
                return False
            
            collision = pick_scene.check_snap_collision(
                [pick_instance], pick_snap
            )
            if collision:
                return False
        
        pick_scene.remove_instance(pick_i)
        
        return True
    
    def no_op_action(self):
        return 0
        

class PickAndPlace(LtronGymComponent):
    def __init__(self,
        scene_components,
        pick_cursor_component,
        place_cursor_component,
        max_instances_per_scene=None,
        check_collision=False,
    ):
        self.scene_components = scene_components
        self.pick_cursor_component = pick_cursor_component
        self.place_cursor_component = place_cursor_component
        self.max_instances_per_scene = max_instances_per_scene
        self.check_collision = check_collision
        
        self.action_space = Discrete(3)
    
    def step(self, action):
        
        # if no action was taken, return
        if not action:
            return None, 0., False, {}
        
        # get the pick scene/instance/snap
        pick_n, pick_i, pick_s = self.pick_cursor_component.get_selected_snap()
        
        # if nothing is picked, return
        if pick_i == 0:
            return None, 0., False, {}
        
        # get the place instance/snap
        (place_n,
         place_i,
         place_s) = self.place_cursor_component.get_selected_snap()
        
        # pick and place
        self.pick_and_place(
            action, pick_n, pick_i, pick_s, place_n, place_i, place_s)
        
        return None, 0., False, {}
    
    def pick_and_place(
        self,
        action,
        pick_n,
        pick_i,
        pick_s,
        place_n,
        place_i,
        place_s,
    ):
        # get the pick scene and instance
        pick_scene = self.scene_components[pick_n].brick_scene
        pick_instance = pick_scene.instances[pick_i]
        
        # get the place scene
        place_scene = self.scene_components[place_n].brick_scene
        
        # if the action is 0, do nothing
        if action == 0:
            return
        
        # if the action is 1, the brick selected by the pick cursor will be
        # connected to the brick selected by the place cursor
        elif action == 1:
            place_snap = place_scene.snap_tuple_to_snap((place_i, place_s))
            if place_snap is None:
                return
        
        # if the action is 2, the scene selected by the place cursor will be
        # cleared and the brick selected by the pick cursor will be placed at
        # the origin
        elif action == 2:
            place_snap = None
        
        # the action should never be something other than 0, 1 or 2
        else:
            raise ThisShouldNeverHappen
        
        # if the bricks are in the same scene, there is no need to transfer it
        if pick_n == place_n:
            transferred_instance = pick_instance
        
        # if the bricks are not from the same screen
        # add a matching new brick to the place scene
        else:
            
            # make sure this will not add too many instances to the scene
            if self.max_instances_per_scene is not None:
                if (place_scene.instances.next_instance_id >
                    self.max_instances_per_scene):
                    return False
            
            # check if the picked brick can be removed without colliding
            if self.check_collision:
                pick_snap = pick_scene.snap_tuple_to_snap((pick_i, pick_s))
                if pick_snap is None:
                    return False
                
                collision = pick_scene.check_snap_collision(
                    [pick_instance], pick_snap
                )
                if collision:
                    return False
            
            # get the shape and color of the picked brick
            shape = str(pick_instance.brick_shape)
            color = int(pick_instance.color)
            
            # construct the transform for the new instance
            pick_view_matrix = pick_scene.get_view_matrix()
            place_view_matrix = place_scene.get_view_matrix()
            transferred_transform = (
                numpy.linalg.inv(place_view_matrix) @
                pick_view_matrix @
                pick_instance.transform
            )
            
            # add the new instance to the place scene
            transferred_instance = place_scene.add_instance(
                shape, color, transferred_transform)
        
        # try to place the brick
        if pick_s >= len(transferred_instance.snaps):
            success = False
        else:
            check_pick_collision = (self.check_collision and pick_n == place_n)
            check_place_collision = (self.check_collision and action != 2)
            success = place_scene.pick_and_place_snap(
                transferred_instance.snaps[pick_s],
                place_snap,
                check_pick_collision=check_pick_collision,
                check_place_collision=self.check_collision,
            )
        
        # if we are transferring between scenes
        # we need to either remove the original pick instance on success
        # or remove the new instance we added to the place scene on failure
        if pick_n != place_n:
            if success:
                pick_scene.remove_instance(pick_i)
            else:
                place_scene.remove_instance(transferred_instance)
        
        # if the action is 2, remove all other bricks from the place scene
        if success and action == 2:
            instances_to_remove = []
            for i, instance in place_scene.instances.items():
                if i != transferred_instance.instance_id:
                    instances_to_remove.append(i)
            
            for i in instances_to_remove:
                place_scene.remove_instance(i)
        
        return
    
    def no_op_action(self):
        return 0
