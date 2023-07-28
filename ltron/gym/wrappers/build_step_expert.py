import random
import math
from copy import deepcopy

import numpy

from gymnasium import make, ObservationWrapper
from gymnasium.spaces import Discrete
from gymnasium.vector.utils.spaces import batch_space

from steadfast.hierarchy import (
    stack_numpy_hierarchies,
    pad_numpy_hierarchy,
)

from ltron.constants import SHAPE_CLASS_NAMES
from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import SnapFinger
from ltron.geometry.utils import (
    unscale_transform,
    matrix_angle_close_enough,
    orthogonal_orientations,
    projected_global_pivot,
    surrogate_angle,
)
from ltron.matching import (
    matching_edges,
    find_matches_under_transform,
    compute_misaligned,
)

def wrapped_build_step_expert(env_name, **kwargs):
    return BuildStepExpert(make(env_name, **kwargs))

class BuildStepExpert(ObservationWrapper):
    def __init__(self, env, max_instructions=16, max_instructions_per_cursor=1):
        super().__init__(env)
        self.max_instructions = max_instructions
        self.max_instructions_per_cursor = max_instructions_per_cursor
        
        '''
        options
        1. do the thing where we make the observation space a tuple version
            of the full action space
        2. shrink it down to just what we need to convey what we want
            2.1. would need mode, click, release, rotate angle
        '''
        observation_space = deepcopy(self.env.observation_space)
        observation_space['expert'] = batch_space(
            self.env.action_space, max_instructions)
        observation_space['num_expert_actions'] = Discrete(max_instructions)
        self.observation_space = observation_space
    
    def observation(self, observation):
        # get assemblies
        current_assembly = observation['assembly']
        target_assembly = observation['target_assembly']
        
        # get the current matches (assumes current and target are aligned)
        matches = find_matches_under_transform(
            current_assembly, target_assembly, numpy.eye(4))
        ct_matches = {c:t for c,t in matches}
        tc_matches = {t:c for c,t in matches}
        (ct_connected,
         tc_connected,
         ct_disconnected,
         tc_disconnected,
         fp,
         fn) = compute_misaligned(
            current_assembly, target_assembly, matches)
        
        num_connected = len(ct_connected)
        num_disconnected = len(ct_disconnected)
        num_misplaced = num_connected + num_disconnected
        
        #assert num_misplaced <= 1
        #if num_misplaced > 1:
        #    breakpoint()
        
        # three cases
        # 0. something we can't handle -> []
        # 1. no misaligned -> DONE
        # 2. connected -> rotate
        # 3. disconnected, incorrect orientation -> rotate
        # 4. disconnected, correct orientation -> pick_and_place
        
        # this only inserts and connects bricks, and will not remove them
        if len(fp) != 0 or len(fn) > 1 or num_misplaced > 1:
            actions = []
        
        elif len(fn) == 1:
            actions = self.insert_actions(current_assembly, target_assembly, fn)
        
        elif num_misplaced == 0:
            actions = self.done_actions()
        
        elif num_connected == 1:
            instance_to_rotate = next(iter(ct_connected.keys()))
            snap_set = next(iter(ct_connected.values()))
            target_instance, snap_to_rotate, target_connected, _ = next(
                iter(snap_set))
            actions = self.rotate_actions(
                observation,
                current_assembly,
                target_assembly,
                ct_matches,
                tc_matches,
                instance_to_rotate,
                snap_to_rotate,
                target_instance,
                target_connected,
            )
        
        elif num_disconnected == 1:
            misplaced_current = next(iter(ct_disconnected.keys()))
            misplaced_target = next(iter(ct_disconnected.keys()))
            current_transform = current_assembly['pose'][misplaced_current]
            target_transform = target_assembly['pose'][misplaced_target]
            correct_orientation = matrix_angle_close_enough(
                current_transform, target_transform, math.radians(30.))
            if correct_orientation:
                actions = self.pick_and_place_actions(
                    observation,
                    current_assembly,
                    target_assembly,
                    ct_matches,
                    tc_matches,
                    ct_disconnected,
                    tc_disconnected,
                )
            else:
                instance_to_rotate = misplaced_current
                actions = []
                for tgt_dis_i in tc_disconnected:
                    missing_edges = matching_edges(target_assembly, tgt_dis_i)
                    missing_edges = target_assembly['edges'][:,missing_edges]
                    for _, tgt_con_i, dis_s, con_s in missing_edges.T:
                        actions.extend(self.rotate_actions(
                            observation,
                            current_assembly,
                            target_assembly,
                            ct_matches,
                            tc_matches,
                            instance_to_rotate,
                            dis_s,
                            tgt_dis_i,
                            tgt_con_i,
                        ))
        
        num_expert_actions = min(len(actions), self.max_instructions)
        if len(actions) == 0:
            actions.append(self.env.no_op_action())
        actions = actions[:self.max_instructions]
        
        actions = stack_numpy_hierarchies(*actions)
        actions = pad_numpy_hierarchy(actions, self.max_instructions)
        observation['expert'] = actions
        observation['num_expert_actions'] = num_expert_actions
        
        return observation
    
    def insert_actions(self, current_assembly, target_assembly, fn):
        action = self.env.no_op_action()
        mode_space = self.env.action_space['action_primitives']['mode']
        try:
            insert_index = mode_space.names.index('insert')
        except ValueError:
            print('Warning: no "insert" action primitive found')
            return []
        action['action_primitives']['mode'] = insert_index
        fn = next(iter(fn))
        shape = target_assembly['shape'][fn]
        color = target_assembly['color'][fn]
        action['action_primitives']['insert'][0] = shape
        action['action_primitives']['insert'][1] = color
        return [action]
    
    def done_actions(self):
        action = self.env.no_op_action()
        mode_space = self.env.action_space['action_primitives']['mode']
        try:
            done_index = mode_space.names.index('done')
        except ValueError:
            print('Warning: no "done" action primitive found')
            return []
        action['action_primitives']['mode'] = done_index
        action['action_primitives']['done'] = 1
        
        return [action]
    
    def rotate_actions(self,
        observation,
        current_assembly,
        target_assembly,
        ct_matches,
        tc_matches,
        instance_to_rotate,
        snap_to_rotate,
        target_instance,
        target_connected,
    ):
        click_loc = self.get_snap_locations(
            observation, instance_to_rotate, snap_to_rotate)
        if (not(len(click_loc)) or
            'rotate' not in self.env.no_op_action()['action_primitives']
        ):
            return []
        
        r = self.compute_attached_discrete_rotation(
            target_assembly,
            current_assembly,
            snap_to_rotate,
            target_instance,
            target_connected,
            instance_to_rotate,
            tc_matches[target_connected],
        )
        actions = []
        for p, y, x in click_loc:
            action = self.env.no_op_action()
            mode_space = self.env.action_space['action_primitives']['mode']
            try:
                rotate_index = mode_space.names.index('rotate')
            except ValueError:
                print('Warning: no "rotate" action primitive found')
                return []
            action['action_primitives']['mode'] = rotate_index
            action['action_primitives']['rotate'] = r
            action['cursor']['button'] = p
            action['cursor']['click'] = numpy.array([y, x], dtype=numpy.int64)
            actions.append(action)
        #return [action]
        return actions
    
    def compute_attached_discrete_rotation(self,
        target_assembly,
        current_assembly,
        snap_id,
        target_instance,
        target_connected_instance,
        current_instance,
        current_connected_instance,
    ):
        # compute the target offset between the bricks
        target_transform = target_assembly['pose'][target_instance]
        connected_target_transform = target_assembly['pose'][
            target_connected_instance]
        inv_connected_target_transform = numpy.linalg.inv(
            connected_target_transform)
        target_offset = (
            inv_connected_target_transform @
            target_transform
        )
        
        # compute the target transform of the current instance
        current_transform = current_assembly['pose'][current_instance]
        connected_current_transform = current_assembly['pose'][
            current_connected_instance]
        shape_id = target_assembly['shape'][target_instance]
        target_transform = connected_current_transform @ target_offset
        
        #rotate_space = self.action_space['action_primitives']['rotate']
        return self.compute_discrete_rotation(
            #shape_id,
            current_instance,
            snap_id,
            #current_transform,
            target_transform,
            #rotation_steps=rotate_space.n,
        )
    
    def compute_discrete_rotation(self,
        #shape_id,
        instance_id,
        snap_id,
        #current_transform,
        target_transform,
        #rotation_steps,
    ):
        #brick_shape_name = SHAPE_CLASS_NAMES[shape_id]
        #brick_shape = BrickShape(brick_shape_name)
        #snap_transform = brick_shape.snaps[snap_id].transform
        #inv_snap_transform = numpy.linalg.inv(snap_transform)
        #current_snap_transform = current_transform @ snap_transform
        
        target_r = unscale_transform(target_transform)[:3,:3]
        
        scene = self.env.components['scene'].brick_scene
        view_matrix = scene.get_view_matrix()
        camera_pose = numpy.linalg.inv(view_matrix)
        
        instance = scene.instances[instance_id]
        snap = instance.snaps[snap_id]
        
        pivot_a, pivot_b = projected_global_pivot(
            snap.transform, offset=camera_pose)
        
        os = orthogonal_orientations()
        candidates = [pivot_a @ o @ pivot_b @ instance.transform for o in os]
        ts = [surrogate_angle(target_transform, c) for c in candidates]
        best_i = numpy.argmax(ts)
        
        return best_i
        
        '''
        candidates = []
        for r in range(rotation_steps):
            c = math.cos(r * math.pi * 2 / rotation_steps)
            s = math.sin(r * math.pi * 2 / rotation_steps)
            ry = numpy.array([
                [ c, 0, s, 0],
                [ 0, 1, 0, 0],
                [-s, 0, c, 0],
                [ 0, 0, 0, 1],
            ])
            offset_transform = (
                current_snap_transform @
                ry @
                inv_snap_transform
            )
            offset_r = unscale_transform(offset_transform)[:3,:3]
            t = numpy.trace(offset_r.T @ target_r)
            candidates.append((t,r,offset_transform))
        
        snap_style = brick_shape.snaps[snap_id]
        if isinstance(snap_style, SnapFinger):
            flip_rotation = numpy.array([
                [-1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1],
            ])
            for r in range(rotation_steps):
                c = math.cos(r * math.pi * 2 / rotation_steps)
                s = math.sin(r * math.pi * 2 / rotation_steps)
                ry = numpy.array([
                    [ c, 0, s, 0],
                    [ 0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [ 0, 0, 0, 1],
                ])
                offset_transform = (
                    inv_connected_current_transform @
                    current_snap_transform @
                    ry @
                    flip_rotation @
                    inv_snap_transform
                )
                offset_r = unscale_transform(offset_transform)[:3,:3]
                t = numpy.trace(offset_r.T @ target_r)
                candidates.append((t,r+rotation_steps,offset_transform))
        '''
        
        #return max(candidates)[1]
    
    def pick_and_place_actions(self,
        observation,
        current_assembly,
        target_assembly,
        ct_matches,
        tc_matches,
        ct_disconnected,
        tc_disconnected,
    ):
        mode_space = self.env.action_space['action_primitives']['mode']
        pnp_index = mode_space.names.index('pick_and_place')
        actions = []
        for tgt_dis_i in tc_disconnected:
            missing_edges = matching_edges(target_assembly, tgt_dis_i)
            missing_edges = target_assembly['edges'][:,missing_edges]
            for _, tgt_con_i, dis_s, con_s in missing_edges.T:
                if tgt_con_i in tc_matches:
                    cur_con_i = tc_matches[tgt_con_i]
                    
                    for cur_dis_i in tc_disconnected[tgt_dis_i]:
                        # make an action connecting cur_dis_i, dis_s
                        # to cur_con_i, con_s
                        dis_loc = self.get_snap_locations(
                            observation, cur_dis_i, dis_s)
                        con_loc = self.get_snap_locations(
                            observation, cur_con_i, con_s)
                        
                        num_combos = min(
                            len(dis_loc),
                            len(con_loc),
                            self.max_instructions_per_cursor,
                        )
                        random.shuffle(dis_loc)
                        random.shuffle(con_loc)
                        dis_loc = dis_loc[:num_combos]
                        con_loc = con_loc[:num_combos]
                        
                        for d, c in zip(dis_loc, con_loc):
                            action = self.env.no_op_action()
                            action['action_primitives']['mode'] = pnp_index
                            action['action_primitives']['pick_and_place'] = 1
                            action['cursor']['button'] = d[0]
                            action['cursor']['click'] = numpy.array(
                                [d[1], d[2]])
                            action['cursor']['release'] = numpy.array(
                                [c[1], c[2]])
                            actions.append(action)
        
        return actions
    
    def get_snap_locations(self, observation, i, s):
        ny, nx = numpy.where(
            (observation['neg_snap_render'][:,:,0] == i) &
            (observation['neg_snap_render'][:,:,1] == s)
        )
        if len(ny):
            return [(0, y, x) for y, x in zip(ny, nx)]
        
        py, px = numpy.where(
            (observation['pos_snap_render'][:,:,0] == i) &
            (observation['pos_snap_render'][:,:,1] == s)
        )
        return [(1, y, x) for y, x in zip(py, px)]

