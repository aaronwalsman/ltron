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
from ltron.gym.spaces import (
    SinglePixelSelectionSpace,
)

from ltron.geometry.collision import check_collision

class HandspacePickAndPlace(LtronGymComponent):
    def __init__(self,
        workspace_scene_component,
        workspace_pos_snap_component,
        workspace_neg_snap_component,
        handspace_scene_component,
        handspace_pos_snap_component,
        handspace_neg_snap_component,
        check_collisions=False,
    ):
        self.workspace_scene_component = workspace_scene_component
        self.workspace_pos_snap_component = workspace_pos_snap_component
        self.workspace_neg_snap_component = workspace_neg_snap_component
        self.handspace_scene_component = handspace_scene_component
        self.handspace_pos_snap_component = handspace_pos_snap_component
        self.handspace_neg_snap_component = handspace_neg_snap_component
        self.check_collisions = check_collisions
        
        self.workspace_width = self.workspace_pos_snap_component.width
        self.workspace_height = self.workspace_pos_snap_component.height
        self.handspace_width = self.handspace_pos_snap_component.width
        self.handspace_height = self.handspace_pos_snap_component.height
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        pick_space = SinglePixelSelectionSpace(
            self.handspace_width, self.handspace_height)
        place_space = SinglePixelSelectionSpace(
            self.workspace_width, self.workspace_height)
        place_at_origin_space = Discrete(2)
        self.observation_space = Dict({'success':Discrete(2)})
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'pick':pick_space,
            'place':place_space,
            'place_at_origin':place_space,
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
            print('pick out of bounds')
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
        pick_brick_type = pick_instance.brick_type
        pick_brick_color = pick_instance.color
        brick_type_snap = pick_brick_type.snaps[pick_snap_id]
        
        
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
                numpy.linalg.inv(brick_type_snap.transform)
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
            str(pick_brick_type),
            pick_brick_color,
            best_workspace_transform,
        )
        
        if place_at_origin:
            if self.check_collisions:
                collision = workspace_scene.check_snap_collision(
                    [new_brick],
                    new_brick.get_snap(pick_snap_id),
                    'push',
                    dump_images='push')
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
            if self.check_collisions:
                collision = workspace_scene.check_snap_collision(
                    [new_brick], new_brick.get_snap(pick_snap_id), 'push')
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

class PickAndPlace(LtronGymComponent):
    def __init__(self,
        scene, pos_snap_render, neg_snap_render, check_collisions
    ):
        self.scene_component = scene
        self.action_executed = 0
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render
        self.width = self.pos_snap_render.width
        self.height = self.pos_snap_render.height
        self.check_collisions = check_collisions
        assert self.neg_snap_render.width == self.width
        assert self.neg_snap_render.height == self.height
        
        activate_space = Discrete(2)
        polarity_space = Discrete(2)
        pick_direction_space = Discrete(2)
        pick_space = SinglePixelSelectionSpace(self.width, self.height)
        place_space = SinglePixelSelectionSpace(self.width, self.height)
        self.action_space = Dict({
            'activate':activate_space,
            'polarity':polarity_space,
            'direction':pick_direction_space,
            'pick':pick_space,
            'place':place_space,
        })
        #self.action_space = MultiDiscrete([2, self.width, self.height, self.width, self.height])
        # self.action_space = MultiDiscrete([2, 20, 20, 20, 20])

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
        
        #activate, polarity, direction, pick, place = action
        activate = action['activate']
        if not activate:
            return {'success': 0}, 0., False, {}
        polarity = action['polarity']
        direction = action['direction']
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
        
        if self.check_collisions and False:
            instance = scene.instances[pick_instance]
            snap = instance.get_snap(pick_id)
            collision = scene.check_snap_collision(
                [instance], snap, direction)
            if collision:
                return {'success': 0}, 0, False, None
            
            initial_transform = instance.transform
            self.scene_component.brick_scene.pick_and_place_snap(
                (pick_instance, pick_id), (place_instance, place_id))
            collision = scene.check_snap_collision(
                [instance], snap, 'push')
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
