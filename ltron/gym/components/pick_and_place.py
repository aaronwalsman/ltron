import math

import numpy

from pyquaternion import Quaternion

from ltron.gym.components.ltron_gym_component import LtronGymComponent
from gym.spaces import (
    Discrete,
    Tuple,
    Dict,
    MultiDiscrete
)
from ltron.gym.spaces import PixelSpace
from ltron.geometry.utils import matrix_is_mirrored

from ltron.geometry.collision import check_collision

'''
class HandspacePickAndPlace(LtronGymComponent):
    def __init__(self,
        workspace_scene_component,
        workspace_pos_snap_component,
        workspace_neg_snap_component,
        handspace_scene_component,
        handspace_pos_snap_component,
        handspace_neg_snap_component,
        check_collision=False,
    ):
        self.workspace_scene_component = workspace_scene_component
        self.workspace_pos_snap_component = workspace_pos_snap_component
        self.workspace_neg_snap_component = workspace_neg_snap_component
        self.handspace_scene_component = handspace_scene_component
        self.handspace_pos_snap_component = handspace_pos_snap_component
        self.handspace_neg_snap_component = handspace_neg_snap_component
        self.check_collision = check_collision
        
        self.workspace_width = self.workspace_pos_snap_component.width
        self.workspace_height = self.workspace_pos_snap_component.height
        self.handspace_width = self.handspace_pos_snap_component.width
        self.handspace_height = self.handspace_pos_snap_component.height
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        pick_space = PixelSpace(self.handspace_width, self.handspace_height)
        place_space = PixelSpace(self.workspace_width, self.workspace_height)
        place_at_origin_space = Discrete(2)
        self.observation_space = Dict({'success':Discrete(2)})
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'pick':pick_space,
            'place':place_space,
            'place_at_origin':place_at_origin_space,
        })
    
    def reset(self):
        return {'success':False}
    
    def step(self, action):
        activate = action['activate']
        if not activate:
            return {'success':False}, 0., False, {}
        polarity = action['polarity']
        pick_y, pick_x = action['pick']
        place_y, place_x = action['place']
        place_at_origin = action['place_at_origin']
        
        if polarity == 1:
            pick_map = self.handspace_pos_snap_component.observation
            place_map = self.workspace_neg_snap_component.observation
        else:
            pick_map = self.handspace_neg_snap_component.observation
            place_map = self.workspace_pos_snap_component.observation
        
        try:
            pick_instance_id, pick_snap_id = pick_map[pick_y, pick_x]
        except IndexError:
            pick_instance_id = 0
        
        try:
            place_instance_id, place_snap_id = place_map[place_y, place_x]
        except IndexError:
            place_instance_id = 0
        
        if pick_instance_id == 0:
            return {'success':0}, 0, False, None
        
        if place_instance_id == 0 and not place_at_origin:
            return {'success':0}, 0, False, None
        
        workspace_scene = self.workspace_scene_component.brick_scene
        handspace_scene = self.handspace_scene_component.brick_scene
        pick_instance = handspace_scene.instances[pick_instance_id]
        pick_brick_shape = pick_instance.brick_shape
        pick_brick_color = pick_instance.color
        brick_shape_snap = pick_brick_shape.snaps[pick_snap_id]
        brick_shape_snap_transform = brick_shape_snap.transform
        if matrix_is_mirrored(brick_shape_snap_transform):
            brick_shape_snap_transform[0:3,0] *= -1
        
        workspace_view_matrix = workspace_scene.get_view_matrix()
        handspace_view_matrix = handspace_scene.get_view_matrix()
        best_workspace_transform = None
        best_pseudo_angle = -float('inf')
        for i in range(4):
            angle = i * math.pi / 2
            rotation = Quaternion(axis=(0,1,0), angle=angle)
            workspace_transform = (
                workspace_scene.upright @
                rotation.transformation_matrix @
                numpy.linalg.inv(brick_shape_snap_transform)
            )
            handspace_camera_local = (
                handspace_view_matrix @ pick_instance.transform)
            workspace_camera_local = (
                workspace_view_matrix @ workspace_transform)
            
            offset = (
                workspace_camera_local @
                numpy.linalg.inv(handspace_camera_local)
            )
            pseudo_angle = numpy.trace(offset[:3,:3])
            if pseudo_angle > best_pseudo_angle:
                best_pseudo_angle = pseudo_angle
                best_workspace_transform = workspace_transform
        new_brick = workspace_scene.add_instance(
            str(pick_brick_shape),
            pick_brick_color,
            best_workspace_transform,
        )
        
        if place_at_origin:
            if self.check_collision:
                collision = workspace_scene.check_snap_collision(
                    [new_brick], new_brick.snaps[pick_snap_id])
                if collision:
                    workspace_scene.remove_instance(new_brick)
                    success = False
                else:
                    success = True
            else:
                success = True
        
        else:
            workspace_scene.pick_and_place_snap(
                (new_brick.instance_id, pick_snap_id),
                (place_instance_id, place_snap_id),
            )
            if self.check_collision:
                collision = workspace_scene.check_snap_collision(
                    [new_brick], new_brick.snaps[pick_snap_id])
                if collision:
                    workspace_scene.remove_instance(new_brick)
                    success = False
                else:
                    success = True
            else:
                success = True
        
        if success:
            handspace_scene.clear_instances()
        
        return {'success':success}, 0., False, {}
    
    def no_op_action(self):
        return {
            'activate':0,
            'polarity':0,
            'pick':numpy.array([0,0]),
            'place':numpy.array([0,0]),
            'place_at_origin':0,
        }
'''

class MultiScreenPickAndPlace(LtronGymComponent):
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
        
        self.observation_space = Dict({'success':Discrete(2)})
        self.action_space = Discrete(2)
        
        self.failure = {'success':False}, 0., False, {}
    
    def reset(self):
        return {'success':False}
    
    def step(self, action):
        # the return value for a failed action
        
        # if no action was taken, return failure
        if not action:
            return self.failure
        
        # get the pick instance/snap
        pick_n, pick_i, pick_s = self.pick_cursor_component.get_selected_snap()
        
        if pick_i == 0:
            return self.failure
        
        # get the place instance/snap
        (place_n,
         place_i,
         place_s) = self.place_cursor_component.get_selected_snap()
        success = self.pick_and_place(
            pick_n, pick_i, pick_s, place_n, place_i, place_s)
        
        return {'success':success}, 0, False, None
    
    def pick_and_place(
        self,
        pick_n,
        pick_i,
        pick_s,
        place_n,
        place_i,
        place_s,
    ):
        pick_scene = self.scene_components[pick_n].brick_scene
        pick_instance = pick_scene.instances[pick_i]
        
        place_scene = self.scene_components[place_n].brick_scene
        if len(place_scene.instances):
            if place_i == 0:
                return False
            place_snap = place_scene.snap_tuple_to_snap((place_i, place_s))
        else:
            place_snap = None
        
        # if the bricks are not from the same screen
        # add a matching new brick to the place scene
        if pick_n == place_n:
            transferred_instance = pick_instance
        else:
            if self.check_collision:
                pick_snap = pick_scene.snap_tuple_to_snap((pick_i, pick_s))
                collision = pick_scene.check_snap_collision(
                    [pick_instance], pick_snap
                )
                if collision:
                    return False
            
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
        success = place_scene.pick_and_place_snap(
            transferred_instance.snaps[pick_s],
            place_snap,
            check_pick_collision=(self.check_collision and pick_n == place_n),
            check_place_collision=self.check_collision,
        )
        
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


class CursorHandspacePickAndPlace(LtronGymComponent):
    def __init__(self,
        workspace_scene_component,
        workspace_cursor_component,
        handspace_scene_component,
        handspace_cursor_component,
        check_collision=False,
    ):
        self.workspace_scene_component = workspace_scene_component
        self.workspace_cursor_component = workspace_cursor_component
        self.handspace_scene_component = handspace_scene_component
        self.handspace_cursor_component = handspace_cursor_component
        self.check_collision = check_collision
        
        self.observation_space = Dict({'success':Discrete(2)})
        self.action_space = Discrete(3)
    
    def reset(self):
        return {'success':False}
    
    def step(self, action):
        if not action:
            return {'success':False}, 0., False, {}
        place_at_origin = action == 2
        
        pick_instance_id = self.handspace_cursor_component.instance_id
        pick_snap_id = self.handspace_cursor_component.snap_id
        place_instance_id = self.workspace_cursor_component.instance_id
        place_snap_id = self.workspace_cursor_component.snap_id
        
        if pick_instance_id == 0:
            return {'success':0}, 0, False, None
        
        if place_instance_id == 0 and not place_at_origin:
            return {'success':0}, 0, False, None
        
        workspace_scene = self.workspace_scene_component.brick_scene
        handspace_scene = self.handspace_scene_component.brick_scene
        pick_instance = handspace_scene.instances[pick_instance_id]
        pick_brick_shape = pick_instance.brick_shape
        pick_brick_color = pick_instance.color
        brick_shape_snap = pick_brick_shape.snaps[pick_snap_id]
        brick_shape_snap_transform = brick_shape_snap.transform
        if matrix_is_mirrored(brick_shape_snap_transform):
            brick_shape_snap_transform[0:3,0] *= -1
        
        workspace_view_matrix = workspace_scene.get_view_matrix()
        handspace_view_matrix = handspace_scene.get_view_matrix()
        
        transferred_transform = (
            numpy.linalg.inv(workspace_view_matrix) @
            handspace_view_matrix @
            pick_instance.transform
        )
        
        new_brick = workspace_scene.add_instance(
            str(pick_brick_shape),
            pick_brick_color,
            transferred_transform,
        )
        
        if place_at_origin:
            place = None
        else:
            #place = (place_instance_id, place_snap_id)
            place = workspace_scene.snap_tuple_to_snap(
                (place_instance_id, place_snap_id))
        
        success = workspace_scene.pick_and_place_snap(
            #(new_brick.instance_id, pick_snap_id),
            new_brick.snaps[pick_snap_id],
            place,
            check_collision=self.check_collision,
        )
        
        if success:
            handspace_scene.clear_instances()
        else:
            workspace_scene.remove_instance(new_brick)
        
        return {'success':success}, 0, False, None
        
    def no_op_action(self):
        return 0

class PickAndPlace(LtronGymComponent):
    def __init__(self,
        scene, pos_snap_render, neg_snap_render, check_collision
    ):
        self.scene_component = scene
        self.action_executed = 0
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render
        self.width = self.pos_snap_render.width
        self.height = self.pos_snap_render.height
        self.check_collision = check_collision
        assert self.neg_snap_render.width == self.width
        assert self.neg_snap_render.height == self.height
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        pick_space = PixelSpace(self.width, self.height)
        place_space = PixelSpace(self.width, self.height)
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'pick':pick_space,
            'place':place_space,
        })

        self.observation_space = Dict({'success': Discrete(2)})

    def reset(self):
        return {'success':False}

    def step(self, action):
        #if action is None: return None, 0, False, None
        #polarity = action[0]  # Integer, 0 or 1
        # pick_x, pick_y = action[1]+100, action[2]+100 # a tuple, corresponding to the coordinate of pick location
        # place_x, place_y = action[3]+150, action[4]+150 # a tuple, corresponding to the coordinate of place location
        #pick_x, pick_y = action[1], action[2]
        #place_x, place_y = action[3], action[4]
        
        #activate, polarity, pick, place = action
        activate = action['activate']
        if not activate:
            return {'success': 0}, 0., False, {}
        polarity = action['polarity']
        pick_y, pick_x = action['pick']
        place_y, place_x = action['place']
        
        if polarity == 1:
            pick_map = self.pos_snap_render.observation
            place_map = self.neg_snap_render.observation
        else:
            pick_map = self.neg_snap_render.observation
            place_map = self.pos_snap_render.observation
        
        pick_instance, pick_id = pick_map[pick_y, pick_x]
        place_instance, place_id = place_map[place_y, place_x]
        # if check_collision(self.scene_component.brick_scene, pick_instance, abs(polarity - 1), (self.width, self.height)):
        #     return {'pick_place_succeed': 0}, 0, False, None
        
        if pick_instance == 0 and pick_id == 0:
            return {'success' : 0}, 0, False, None
        
        if place_instance == 0 and place_id == 0:
            return {'success' : 0}, 0, False, None
        
        if pick_instance == place_instance:
            return {'success' : 0}, 0, False, None
        
        if self.check_collision:
            instance = self.scene_component.brick_scene.instances[pick_instance]
            initial_transform = instance.transform
            snap = instance.snaps[pick_id]
            collision = self.scene_component.brick_scene.check_snap_collision(
                [instance], snap)

            if collision:
                self.scene_component.brick_scene.move_instance(
                    instance, initial_transform)
                return {'success': 0}, 0, False, None

            place_instance = self.scene_component.brick_scene.instances[place_instance]
            self.scene_component.brick_scene.pick_and_place_snap(
                (pick_instance, pick_id), (place_instance, place_id))
            collision = self.scene_component.check_snap_collision(
                [instance], snap)

            if collision:
                self.scene_component.brick_scene.move_instance(
                    instance, initial_transform)
                return {'success': 0}, 0, False, None
            else:
                return {'success': 1}, 0, False, None
            
        else:
            self.scene_component.brick_scene.pick_and_place_snap(
                (pick_instance, pick_id), (place_instance, place_id))

            return {'success' : 1}, 0, False, None  # the observation is whether the action succeeds or not
    
    def no_op_action(self):
        return {
            'activate':0,
            'polarity':0,
            'pick':numpy.array([0,0]),
            'place':numpy.array([0,0]),
        }
