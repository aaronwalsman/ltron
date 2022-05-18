import numpy

from ltron.gym.components.ltron_gym_component import LtronGymComponent
from gym.spaces import Discrete, Dict

class MultiScenePickAndPlace(LtronGymComponent):
    def __init__(self,
        scene_components,
        pick_cursor_component,
        place_cursor_component,
        check_collision=False,
    ):
        self.scene_components = scene_components
        self.pick_cursor_component = pick_cursor_component
        self.place_cursor_component = place_cursor_component
        self.check_collision = check_collision
        
        #self.observation_space = Dict({'success':Discrete(2)})
        self.action_space = Discrete(2)
        
        #self.failure = {'success':False}, 0., False, {}
    
    #def reset(self):
    #    #return {'success':False}
    
    def step(self, action):
        # the return value for a failed action
        
        # if no action was taken, return failure
        if not action:
            #return self.failure
            return None, 0., False, {}
        
        # get the pick instance/snap
        pick_n, pick_i, pick_s = self.pick_cursor_component.get_selected_snap()
        
        if pick_i == 0:
            #return self.failure
            return None, 0., False, {}
        
        # get the place instance/snap
        (place_n,
         place_i,
         place_s) = self.place_cursor_component.get_selected_snap()
        success = self.pick_and_place(
            pick_n, pick_i, pick_s, place_n, place_i, place_s)
        
        return {'success':success}, 0., False, None
    
    def pick_and_place(
        self,
        pick_n,
        pick_i,
        pick_s,
        place_n,
        place_i,
        place_s,
    ):
        #print(
        #    'pick-and-placing',
        #    pick_n, pick_i, pick_s, place_n, place_i, place_s,
        #)
        pick_scene = self.scene_components[pick_n].brick_scene
        pick_instance = pick_scene.instances[pick_i]
        place_scene = self.scene_components[place_n].brick_scene
        '''
        if len(place_scene.instances):
            if place_i == 0:
                return False
            place_snap = place_scene.snap_tuple_to_snap((place_i, place_s))
        else:
            place_snap = None
        '''
        
        # if the place instance is 0, the instance will be placed at the origin
        if place_i == 0:
            # does this still work?
            place_snap = None
        else:
            place_snap = place_scene.snap_tuple_to_snap((place_i, place_s))
            if place_snap is None:
                #print('place invalid, breaking')
                return False
        
        # if the bricks are in the same scene, there is no need to transfer it
        if pick_n == place_n:
            transferred_instance = pick_instance
        
        # if the bricks are not from the same screen
        # add a matching new brick to the place scene
        else:
            
            # check if the picked brick can be removed without colliding
            if self.check_collision:
                pick_snap = pick_scene.snap_tuple_to_snap((pick_i, pick_s))
                if pick_snap is None:
                    #print('pick invalid, breaking')
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
            success = place_scene.pick_and_place_snap(
                transferred_instance.snaps[pick_s],
                place_snap,
                check_pick_collision=check_pick_collision,
                check_place_collision=self.check_collision,
            )
        
        #print('success:', success)
        
        # if we are transferring between scenes
        # we need to either remove the original pick instance on success
        # or remove the new instance we added to the place scene on failure
        if pick_scene != place_scene:
            if success:
                pick_scene.remove_instance(pick_i)
            else:
                place_scene.remove_instance(transferred_instance)
        
        return success
    
    def no_op_action(self):
        return 0
