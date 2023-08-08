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

from supermecha.gym.supermecha_container import traceback_decorator

from ltron.matching import match_assemblies
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
from ltron.geometry.collision import build_collision_map
from ltron.matching import (
    matching_edges,
    find_matches_under_transform,
    compute_misaligned,
)

def wrapped_build_step_expert(env_name, **kwargs):
    return BuildStepExpert(make(env_name, **kwargs))

class BuildStepExpert(ObservationWrapper):
    @traceback_decorator
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
    
    @traceback_decorator
    def observation(self, observation):
    
        # get assemblies
        current_assembly = observation['assembly']
        target_assembly = observation['target_assembly']
        if 'initial_assembly' in observation:
            initial_assembly = observation['initial_assembly']
        
        # get the current matches (assumes current and target are aligned)
        #matches = find_matches_under_transform(
        #    current_assembly, target_assembly, numpy.eye(4))
        matches, matching_transform = match_assemblies(
            current_assembly, target_assembly, allow_rotations=False)
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
        
        '''
        HISTORIC, IGNORE THIS BLOCK
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
        #if len(fp) != 0 or len(fn) > 1 or num_misplaced > 1:
        #    actions = []
        '''
        
        # first compute whether or not the model has been assembled correctly
        # (no missing bricks, no extra bricks, no misplaced bricks)
        assembled_correctly = (
            len(fp) == 0 and
            len(fn) == 0 and
            num_misplaced == 0
        )
        
        # next compute whether or not we are done with an assembly step
        assemble_step = False
        action_primitives = self.env.action_space['action_primitives']
        instance_ids = numpy.where(current_assembly['shape'])[0]
        num_bricks = len(instance_ids)
        if 'assemble_step' in set(action_primitives.keys()):
            if observation['action_primitives']['phase'] == 0:
                prev_assembly = (
                    self.env.components['target_assembly'].observations[-1])
                prev_num_bricks = len(numpy.where(prev_assembly['shape'])[0])
                if num_bricks and num_bricks < prev_num_bricks:
                    assemble_step = True
                else:
                    assemble_step = False
            else:
                num_target_bricks = len(
                    numpy.where(target_assembly['shape'])[0])
                num_initial_bricks = len(
                    numpy.where(initial_assembly['shape'])[0])
                #print('t/i')
                #print(num_target_bricks)
                #print(num_initial_bricks)
                if (num_target_bricks != num_initial_bricks and
                    assembled_correctly
                ):
                    assemble_step = True
                else:
                    assemble_step = False
        
        too_hard = len(fn) > 1 or num_misplaced > 1 or (len(fn) and len(fp))
        
        if too_hard:
            #print('TOO HARD')
            actions = []
        elif assemble_step:
            #print('ASSEMBLE STEP')
            actions = self.assemble_step_actions()
        elif assembled_correctly:
            #print('DONE')
            actions = self.done_actions()
        
        ## first weed out cases that we can't handle
        ## this can only handle single-brick misalignments and
        ## can't handle both false positives and false negatives concurrently
        #if len(fn) > 1 or num_misplaced > 1 or (len(fn) and len(fp)):
        #    actions = []
        #
        #elif assemble_step:
        #    breakpoint()
        #    actions = self.assemble_step_actions()
        
        # if there are false positive bricks, remove them
        elif len(fp) != 0:
            #print('REMOVE')
            actions = self.remove_actions(
                observation,
                current_assembly,
                target_assembly,
                fp,
            )
        
        elif len(fn) == 1:
            #print('INSERT')
            actions = self.insert_actions(current_assembly, target_assembly, fn)
        
        #elif num_misplaced == 0:
        #    actions = self.done_actions()
        
        elif num_bricks == 1:
            #print('ROTATE_FIRST')
            instance_id = instance_ids[0]
            target_id = next(iter(ct_disconnected[instance_id]))
            target_transform = target_assembly['pose'][target_id]
            
            scene = self.env.components['scene'].brick_scene
            instance = scene.instances[instance_id]
            snap_ids = [snap.snap_id for snap in instance.snaps]
            actions = []
            for snap_id in snap_ids:
                click_loc = self.get_snap_locations(
                    observation, instance_id, snap_id)
                if (not(len(click_loc)) or
                    'rotate' not in self.env.no_op_action()['action_primitives']
                ):
                    continue
                
                r = self.compute_discrete_rotation(
                    instance_id,
                    snap_id,
                    target_transform,
                )
                
                for p, y, x in click_loc:
                    action = self.env.no_op_action()
                    mode_space = (
                        self.env.action_space['action_primitives']['mode'])
                    try:
                        rotate_index = mode_space.names.index('rotate')
                    except ValueError:
                        print('Warning: no "rotate" action primitive found')
                        return []
                    action['action_primitives']['mode'] = rotate_index
                    action['action_primitives']['rotate'] = r
                    action['cursor']['button'] = p
                    action['cursor']['click'] = numpy.array(
                        [y, x], dtype=numpy.int64)
                    actions.append(action)
        
        elif num_connected == 1:
            #print('ROTATE')
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
            #print('PICK AND PLACE')
            #breakpoint()
            #print('current:')
            #print(current_assembly['pose'][1:3])
            #print('target:')
            #print(target_assembly['pose'][1:3])
            #breakpoint()
            
            misplaced_current = next(iter(ct_disconnected.keys()))
            #misplaced_target = next(iter(ct_disconnected.keys()))
            misplaced_target = next(iter(ct_disconnected[misplaced_current]))
            
            current_transform = current_assembly['pose'][misplaced_current]
            target_transform = target_assembly['pose'][misplaced_target]
            correct_orientation = matrix_angle_close_enough(
                current_transform, target_transform, math.radians(30.))
            if correct_orientation:
                #print('!!!')
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
                #print('???')
                #print(current_transform)
                #print(target_transform)
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
        if 'done' in mode_space.names:
            done_index = mode_space.names.index('done')
            action['action_primitives']['mode'] = done_index
            action['action_primitives']['done'] = 1
        elif 'phase' in mode_space.names:
            phase_index = mode_space.names.index('phase')
            action['action_primitives']['mode'] = phase_index
            action['action_primitives']['phase'] = 1
        else:
            print('Warning: no "done" action primitive found')
            return []
        
        
        return [action]
    
    def assemble_step_actions(self):
        action = self.env.no_op_action()
        mode_space = self.env.action_space['action_primitives']['mode']
        assemble_step_index = mode_space.names.index('assemble_step')
        action['action_primitives']['mode'] = assemble_step_index
        action['action_primitives']['assemble_step'] = 1
        
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
    
    def remove_actions(self,
        observation,
        current_assembly,
        target_assembly,
        fn,
    ):
        # do a matching between the initial and current assembly
        # so that we can map instance in current assembly
        # to the collision map
        initial_assembly = observation['initial_assembly']
        matches, offset = match_assemblies(current_assembly, initial_assembly)
        current_to_initial = {c:i for c,i in matches}
        initial_to_current = {i:c for c,i in matches}
        
        # find what can be removed
        removable_clicks = []
        collision_map = self.env.components['initial_assembly'].collision_map
        for c, i in current_to_initial.items():
            if c not in fn:
                continue
            
            for (_,p,snaps), blocking in collision_map[i].items():
                for b in blocking:
                    if b in initial_to_current:
                        break
                else:
                    for s in snaps:
                        removable_clicks.append((c,s))
        
        # NEXT:
        # removable_clicks is a list of things that can be clicked on to
        # remove something, find the locations where they can be clicked from
        # the observed snap maps and send it off!
        mode_space = self.env.action_space['action_primitives']['mode']
        remove_index = mode_space.names.index('remove')
        actions = []
        for c,s in removable_clicks:
            con_loc = self.get_snap_locations(observation, c, s)
            num_loc = min(len(con_loc), self.max_instructions_per_cursor)
            random.shuffle(con_loc)
            con_loc = con_loc[:num_loc]
        
            for button, y, x in con_loc:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = remove_index
                action['action_primitives']['remove'] = 1
                action['cursor']['button'] = button
                action['cursor']['click'] = numpy.array([y,x])
                actions.append(action)
        
        return actions
    
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

