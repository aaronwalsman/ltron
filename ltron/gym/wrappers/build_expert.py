import math
from copy import deepcopy

import numpy

from scipy.spatial import cKDTree

from gymnasium import make, ObservationWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.spaces import Discrete

from steadfast.hierarchy import (
    stack_numpy_hierarchies,
    pad_numpy_hierarchy,
)

from ltron.constants import SHAPE_CLASS_LABELS
SHAPE_LABEL_NAMES = {v:k for k,v in SHAPE_CLASS_LABELS.items()}
from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import SnapFinger, UnsupportedSnap
from ltron.matching import (
    match_assemblies,
    find_matches_under_transform,
    compute_misaligned,
    matching_edges,
)
from ltron.geometry.utils import (
    unscale_transform,
    matrix_angle_close_enough,
    matrix_rotation_axis,
    vector_angle_close_enough,
)

def wrapped_build_expert(env_name, **kwargs):
    return BuildExpert(make(env_name, **kwargs))

class BuildExpert(ObservationWrapper):
    def __init__(self,
        env,
        max_instructions=2048,
        shuffle_instructions=True,
        always_add_viewpoint_actions=False,
        align_orientation=True,
    ):
        super().__init__(env)
        
        # store variables
        self.max_instructions = max_instructions
        self.shuffle_instructions = shuffle_instructions
        self.always_add_viewpoint_actions = always_add_viewpoint_actions
        self.align_orientation = align_orientation
        
        # build observation space
        observation_space = deepcopy(self.env.observation_space)
        observation_space['expert'] = batch_space(
            self.env.action_space, max_instructions)
        observation_space['num_expert_actions'] = Discrete(max_instructions)
        self.observation_space = observation_space
    
    def observation(self, observation):
        # get assemblies
        current_assembly = observation['assembly']
        target_assembly = observation['target_assembly']
        
        # compute the expert actions
        actions = self.expert_actions(
            observation, current_assembly, target_assembly)
        if not len(actions):
            actions = [self.env.no_op_action()]
        
        # convert the expert actions to a truncated list of instructions
        expert_actions = actions[:self.max_instructions]
        num_expert_actions = len(expert_actions)
        actions = stack_numpy_hierarchies(*actions)
        actions = pad_numpy_hierarchy(actions, self.max_instructions)
        observation['expert'] = actions
        observation['num_expert_actions'] = num_expert_actions
        
        return observation
    
    def expert_actions(self, observation, current_assembly, target_assembly):
        # match the current and target assemblies
        if self.align_orientation:
            instance_ids = numpy.where(current_assembly['shape'])[0]
            if len(instance_ids):
                first_instance = instance_ids[0]
                match_assembly = {
                    'shape' : current_assembly['shape'][:first_instance+1],
                    'color' : current_assembly['color'][:first_instance+1],
                    'pose' : current_assembly['pose'][:first_instance+1],
                }
            else:
                match_assembly = current_assembly
            kdtree = cKDTree(target_assembly['pose'][:,:3,3])
        else:
            match_assembly = current_assembly
            kdtree = None
        
        matches, offset = match_assemblies(current_assembly, target_assembly)
        
        if self.align_orientation:
            matches = find_matches_under_transform(
                current_assembly,
                target_assembly,
                offset,
                kdtree,
            )
        
        #current_instances = numpy.where(current_assembly['shape'] != 0)[0]
        #print(current_instances)
        
        current_to_target = dict(matches)
        target_to_current = {v:k for k,v in current_to_target.items()}
        
        # compute the bricks that are:
        #  - not in the correct location, but have a correct connection
        #    (misaligned_connected)
        #  - not in the correct location, but do not have a correct connection
        #    (misaligned_disconnected)
        #  - exist in the current assembly, but do not exist in the target
        #    (false positives)
        #  - exist in the target assembly, but do not exist in the current
        #    (false negatives)
        (current_to_target_misaligned_connected,
         target_to_current_misaligned_connected,
         current_to_target_misaligned_disconnected,
         target_to_current_misaligned_disconnected,
         false_positives,
         false_negatives) = compute_misaligned(
            current_assembly, target_assembly, matches)
        
        #print('HOLD UP')
        #breakpoint()
        #if hasattr(self.env.components['target_assembly'], 'collision_map'):
        #    collision_map = (
        #        self.env.components['target_assembly'].collision_map
        #    )
        #    
        #    # find any bricks blocking false positive from being removed
        #    blocking_bricks = []
        #    
        #    # wait, we need a collision map for current if we want to check
        #    # for blocking false positives... or do it manually?
        
        # if the current assembly matches the target assembly, then select done
        if not (
            len(current_to_target_misaligned_connected) or
            len(current_to_target_misaligned_disconnected) or
            len(false_positives) or
            len(false_negatives)
        ):
            return self.done(action_space, no_op_action)
        
        # if there are bricks that are incorrectly placed, but have a correct
        # connection, adjust the connection
        elif len(target_to_current_misaligned_connected):
            return self.adjust_connection(
                observation,
                current_to_target,
                target_to_current,
                target_to_current_misaligned_connected,
                current_assembly,
                target_assembly,
            )
        
        # if there are bricks that are incorrectly placed and do not have a
        # correct connection, make the connection
        elif len(target_to_current_misaligned_disconnected):
            actions = self.make_connection(
                current_to_target,
                target_to_current,
                list(target_to_current_misaligned_disconnected.keys()),
                current_assembly,
                target_assembly,
            )
        
        # if there false positives, remove them
        elif len(false_positives):
            actions = self.remove_false_positive(
                current_to_target,
                target_to_current,
                false_positives,
                current_assembly,
                target_assembly,
            )
        
        # if the current scene is empty, add the first brick
        elif not len(current_to_target):
            actions = self.add_first_brick(
                false_negatives,
                target_assembly,
                secondary_assemblies,
            )
        
        elif (
            len(current_to_target) == 1
            and self.align_orientation
            and not matrix_angle_close_enough(
                numpy.eye(4), offset, math.radians(5))
        ):
            actions = self.rotate_first_brick(
                current_assembly,
                offset,
            )
        
        # if the current scene is not empty, but is missing bricks, add a brick
        elif len(false_negatives):
            actions = self.make_connection(
                current_to_target,
                target_to_current,
                false_negatives,
                current_assembly,
                target_assembly,
                secondary_assemblies,
            )
        
        # return
        return actions

    def done(self):
        action = self.env.no_op_action()
        mode_switch = self.env.action_space['action_primitives']['mode']
        action['action_primitives']['mode'] = mode_switch.name_to_index('done')
        action['action_primitives']['done'] = 1
        
        return [action]
    
    def change_viewpoint(self):
        actions = []
        mode_switch = self.env.action_space['action_primitives']['mode']
        num_viewpoint_actions = (
            self.env.action_space['action_primitives']['viewpoint'].n)
        for i in range(1, num_viewpoint_actions):
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = mode_switch.name_to_index(
                'viewpoint')
            action['action_primitives']['viewpoint'] = i
            actions.append(action)
        
        return actions
    
    def insert_brick(self, shape, color):
        actions = []
        mode_switch = self.env.action_space['action_primitives']['mode']
        action = self.env.no_op_action()
        action['action_primitives']['mode'] = mode_switch.name_to_index(
            'insert')
        action['action_primitives']['insert'] = numpy.array(
            [shape, color], dtype=numpy.int64)
        
        return [action]
    
    def remove_false_positive(self,
        current_to_target,
        target_to_current,
        false_positives,
        current_assembly,
        target_assembly,
    ):
        actions = []
        
        # what shap/color combos are we trying to remove
        fp_shape_color_snaps = []
        breakpoint()
        
        return actions
    
    def get_snap_locations(observation, snaps):
        locations = []
        neg_snap_map = observation['neg_snap_render']
        pos_snap_map = observation['pos_snap_render']
        for i, s in snaps:
            for polarity, snap_map in enumerate((neg_snap_map, pos_snap_map)):
                click_map = (snap_map[0] == i) & (snap_map[1] == s)
                y, x = numpy.where(click_map)
                locations.extend([(polarity, yy, xx) for yy, xx in zip(y,x)])
        
        return locations

def rotate_first_brick(
    self,
    current_assembly,
    offset,
    rotation_steps=4,
):
    current_instance = numpy.where(current_assembly['shape'] != 0)[0][0]
    offset_axis = matrix_rotation_axis(offset)
    
    shape_id = current_assembly['shape'][current_instance]
    brick_shape_name = self.shape_names[shape_id]
    brick_shape = BrickShape(brick_shape_name)
    rotatable_snaps = []
    for i, snap in enumerate(brick_shape.snaps):
        snap_axis = snap.transform[:3,:3] @ [0,1,0]
        if vector_angle_close_enough(
            offset_axis, snap_axis, math.radians(5), allow_negative=True
        ):
            rotatable_snaps.append(i)
    
    # if one of these snaps is not already picked, pick one of them
    pick_n, pick_i, pick_s = self.env.get_pick_snap()
    if (pick_n != self.target_scene or
        pick_i != current_instance or
        pick_s not in rotatable_snaps
    ):
        pick_actions = []
        for rotatable_snap in rotatable_snaps:
            pick_actions.extend(self.env.actions_to_pick_snap(
                self.target_scene,
                current_instance,
                rotatable_snap,
            ))
        
        if self.always_add_viewpoint_actions or not len(pick_actions):
            view_actions = self.env.all_component_actions(
                self.target_scene + '_viewpoint', include_no_op=False)
            pick_actions.extend(view_actions)
        
        return pick_actions
    
    else:
        current_transform = current_assembly['pose'][current_instance]
        r = self.compute_discrete_rotation(
            shape_id,
            pick_s,
            offset,
            numpy.eye(4),
            rotation_steps=rotation_steps,
        )
        return [self.env.rotate_action(r)]
    
    def compute_attached_discrete_rotation(
        self,
        target_assembly,
        current_assembly,
        snap_id,
        target_instance,
        target_connected_instance,
        current_instance,
        current_connected_instance,
    ):
        
        target_transform = target_assembly['pose'][target_instance]
        connected_target_transform = target_assembly['pose'][
            target_connected_instance]
        inv_connected_target_transform = numpy.linalg.inv(
            connected_target_transform)
        target_offset = (
            inv_connected_target_transform @
            target_transform
        )

        current_transform = current_assembly['pose'][current_instance]
        connected_current_transform = current_assembly['pose'][
            current_connected_instance]
        
        shape_id = target_assembly['shape'][target_instance]
        target_transform = connected_current_transform @ target_offset
        
        return self.compute_discrete_rotation(
            shape_id,
            snap_id,
            current_transform,
            target_transform,
        )
    
    def compute_discrete_rotation(self,
        shape_id,
        snap_id,
        current_transform,
        target_transform,
    ):
        rotation_steps = (
            self.env.action_space['action_primitives']['rotation'].n)
        
        brick_shape_name = self.shape_names[shape_id]
        brick_shape = BrickShape(brick_shape_name)
        snap_transform = brick_shape.snaps[snap_id].transform
        inv_snap_transform = numpy.linalg.inv(snap_transform)
        current_snap_transform = current_transform @ snap_transform
        
        target_r = unscale_transform(target_transform)[:3,:3]
        
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
        
        #snap_style = brick_shape.snaps[snap_id]
        #if isinstance(snap_style, SnapFinger):
        #    flip_rotation = numpy.array([
        #        [-1, 0, 0, 0],
        #        [ 0,-1, 0, 0],
        #        [ 0, 0, 1, 0],
        #        [ 0, 0, 0, 1],
        #    ])
        #    for r in range(rotation_steps):
        #        c = math.cos(r * math.pi * 2 / rotation_steps)
        #        s = math.sin(r * math.pi * 2 / rotation_steps)
        #        ry = numpy.array([
        #            [ c, 0, s, 0],
        #            [ 0, 1, 0, 0],
        #            [-s, 0, c, 0],
        #            [ 0, 0, 0, 1],
        #        ])
        #        offset_transform = (
        #            inv_connected_current_transform @
        #            current_snap_transform @
        #            ry @
        #            flip_rotation @
        #            inv_snap_transform
        #        )
        #        offset_r = unscale_transform(offset_transform)[:3,:3]
        #        t = numpy.trace(offset_r.T @ target_r)
        #        candidates.append((t,r+rotation_steps,offset_transform))

        return max(candidates)[1]
    
    def adjust_connection(self,
        observation,
        current_to_target,
        target_to_current,
        targets_to_fix,
        current_assembly,
        target_assembly,
    ):
        actions = []
        
        # pickable maps (i,s) in current assembly to [(i,ci,cs)] in target
        pickable = {}
        for tgt_i, cur_set in targets_to_fix.items():
            for cur_i, cur_s, cur_con_i, cur_con_s in cur_set:
                fn_edge_indices = matching_edges(
                    target_assembly, i1=tgt_i, s1=cur_s, s2=cur_con_s)
                fn_edges = target_assembly['edges'][:,fn_edge_indices]
                if fn_edges.shape[1]:
                    # should only be one
                    tgt_con_i = fn_edges[1,0]
                    pickable[cur_i, cur_s] = [[tgt_i, tgt_con_i, cur_con_s]]
        
        #pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        #if (pick_n, pick_i, pick_s) not in pickable:
            
        #print('PICKABLE')
        #print(pickable)
        
        #pick_actions = []
        pick_locations = self.get_snap_locations(observation, pickable)
        
        if not len(pick_locations):
            return self.change_viewpoint()
        
        actions = []
        for (i, y, x) in pick_locations:
            r = self.compute_attached_discrete_rotation(
                target_assembly,
                current_assembly,
                pick_s,
                tgt_i,
                tgt_con_i,
                pick_i,
                target_to_current[tgt_con_i],
            )
            action = self.env.no_op_action()
            mode_space = self.env.action_space['action_primitives']['mode']
            action['action_primitives']['mode'] = mode_space.name_to_index(
                'rotate')
            action['action_primitives']['rotate'] = r
            actions.append(action)
        
        return actions
        
        ## it's already clicked, it's time to rotate!
        ##n, tgt_i, tgt_con_i, tgt_con_s = pickable[pick_n, pick_i, pick_s][0]
        #r = self.compute_attached_discrete_rotation(
        #    target_assembly,
        #    current_assembly,
        #    pick_s,
        #    tgt_i,
        #    tgt_con_i,
        #    pick_i,
        #    target_to_current[tgt_con_i],
        #)
        
        #rotate_actions = [self.action_component.rotate_action(r)]
        #rotate_actions = [self.env.rotate_action(r)]
        #if self.always_add_viewpoint_actions or not len(rotate_actions):
        #    rotate_actions.extend(
        #        self.action_component.all_component_actions(
        #            self.target_scene + '_viewpoint')
        #    )
        #return rotate_actions
    
    
    def add_first_brick(
        self,
        targets_to_fix,
        target_assembly,
        secondary_assemblies,
    ):
        actions = []
        fn_shape_colors = []
        for target_to_fix in targets_to_fix:
            target_pose = target_assembly['pose'][target_to_fix]
            shape = target_assembly['shape'][target_to_fix]
            color = target_assembly['color'][target_to_fix]
            
            shape_name = self.shape_names[shape]
            shape_type = BrickShape(shape_name)
            y_axis = target_pose[:3,1]
            scene = self.scene_components[self.target_scene].brick_scene
            for snap in shape_type.snaps:
                if isinstance(snap, UnsupportedSnap):
                    continue
                snap_y_axis = scene.upright @ snap.transform[:,1]
                snap_y_axis = snap_y_axis[:3]
                if numpy.dot(y_axis, snap_y_axis) > 0.99:
                    fn_shape_colors.append((shape, color, target_to_fix))
                    break
        
        pickable = set()
        for shape, color, target_index in fn_shape_colors:
            for name, secondary_assembly in secondary_assemblies.items():
                for i in range(secondary_assembly['shape'].shape[0]):
                    if name == self.target_scene:
                        continue
                    secondary_shape = secondary_assembly['shape'][i]
                    secondary_color = secondary_assembly['color'][i]
                    if secondary_shape == shape and secondary_color == color:
                        # what snaps?
                        # upright would be nice...
                        # anything with a connection would be sufficient...
                        # we could even do everything...
                        # but I don't know how many there are...
                        target_pose = target_assembly['pose'][target_index]
                        scene = self.scene_components[name].brick_scene
                        y_axis = target_pose[:3,1]
                        for snap in scene.instances[i].brick_shape.snaps:
                            if isinstance(snap, UnsupportedSnap):
                                continue
                            snap_y_axis = scene.upright @ snap.transform[:,1]
                            snap_y_axis = snap_y_axis[:3]
                            if numpy.dot(y_axis, snap_y_axis) > 0.99:
                                pickable.add((name, i, snap.snap_id))
        
        # if there are no pickable things, add the brick
        if not len(pickable):
            # TODO, need to filter this based on bricks that can be added next
            insert_actions = []
            for shape, color, i in fn_shape_colors:
                insert_actions.append(
                    self.env.actions_to_insert_brick_actions(shape, color))
            
            return insert_actions
        
        #pick_cursor = self.action_component.components['pick_cursor']
        #pick_cursor = self.env.components['pick_cursor']
        #pick_n, pick_i, pick_s = pick_cursor.get_selected_snap()
        #pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        # if a pickable thing is not picked, pick it
        #if (pick_n, pick_i, pick_s) not in pickable:
        pick_locations = []
        for n, i, s in pickable:
            pick_actions.extend(
                self.env.actions_to_pick_snap(n, i, s)
            )
        
        if not len(pick_locations):
            return self.change_viewpoint()
        
        #return pick_actions
        
        '''
        if place_n != self.target_scene:
            #place_actions = self.action_component.actions_to_deselect_place()
            place_actions = self.env.actions_to_deselect_place(
                self.target_scene)
            
            if self.always_add_viewpoint_actions:
                #print('supervising viewpoint(first2):')
                #import pdb
                #pdb.set_trace()
                # TODO
                pass
            
            return place_actions
        
        # clicks are correct, it's time to pick_and_place!
        #pnp_actions = [self.action_component.pick_and_place_action()]
        pnp_actions = [self.env.pick_and_place_action(2)]
        return pnp_actions
        '''
    
    def make_connection(self,
        current_to_target,
        target_to_current,
        targets_to_fix,
        current_assembly,
        target_assembly,
    ):
        actions = []
        
        # what shape/color combos are we looking for
        fn_shape_color_snaps = []
        for target_to_fix in targets_to_fix:
            shape = target_assembly['shape'][target_to_fix]
            color = target_assembly['color'][target_to_fix]
            
            fn_edges = matching_edges(target_assembly, target_to_fix)
            fn_edges = target_assembly['edges'][:,fn_edges]
            for _, tgt_con_i, tgt_s, tgt_con_s in fn_edges.T:
                if tgt_con_i in target_to_current:
                    fn_shape_color_snaps.append(
                        (shape, color, tgt_con_i, tgt_s, tgt_con_s))
        
        # is a pick already clicked on?
        pickable = {}
        for shape, color, tgt_con_i, tgt_s, tgt_con_s in fn_shape_color_snaps:
            #for name, secondary_assembly in secondary_assemblies.items():
            #for i in range(secondary_assembly['shape'].shape[0]):
            for i in range(current_assembly['shape'].shape[0]):
                # if this is already matched, do not consider moving it
                if i in current_to_target:
                    continue
                other_shape = current_assembly['shape'][i]
                other_color = current_assembly['color'][i]
                if other_shape == shape and other_color == color:
                    pickable.setdefault((i, tgt_s), [])
                    cur_con_i = target_to_current[tgt_con_i]
                    pickable[name, i, tgt_s].append((cur_con_i, tgt_con_s))
        
        # if there are no pickable things, add the brick
        if not len(pickable):
            # TODO, need to filter this based on bricks that can be added next
            insert_actions = []
            for shape, color, i, tgt_s, tgt_con_s in fn_shape_color_snaps:
                insert_actions.append(
                    self.insert_brick(shape, color))
            
            return insert_actions
        
        #pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        #if (pick_n, pick_i, pick_s) not in pickable:
        pick_actions = []
        pick_names = []
        for n, i, s in pickable:
            pick_actions.extend(
                self.env.actions_to_pick_snap(n, i, s)
            )
            pick_names.append(n)
        
        if self.always_add_viewpoint_actions or not len(pick_actions):
            #print('supervising viewpoint(fpnp):')
            #import pdb
            #pdb.set_trace()
            for n in set(pick_names):
                #view_actions = self.action_component.all_component_actions(
                view_actions = self.env.all_component_actions(
                        n + '_viewpoint', include_no_op=False)
                #print(view_actions)
                pick_actions.extend(view_actions)
            for n, i, s in pickable:
                a = self.env.actions_to_pick_snap(n, i, s)
        
        return pick_actions
        
        # is a place already clicked on?
        placeable = pickable[pick_n, pick_i, pick_s]
        #place_component = self.action_component.components['place_cursor']
        #place_n, place_i, place_s = place_component.get_selected_snap()
        place_n, place_i, place_s = self.env.get_place_snap()
        
        #current_i = current_to_target[place_i]
        
        if (place_n, place_i, place_s) not in placeable:
            place_actions = []
            place_names = []
            for n, i, s in placeable:
                #place_i = target_to_current[i]
                place_actions.extend(
                    #self.action_component.actions_to_place_snap(n, place_i, s)
                    #self.action_component.actions_to_place_snap(n, i, s)
                    self.env.actions_to_place_snap(n, i, s)
                )
                place_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(place_actions):
                #print('supervising viewpoint(fpnp2):')
                #import pdb
                #pdb.set_trace()
                for n in place_names:
                    #view_actions = self.action_component.all_component_actions(
                    view_actions = self.env.all_component_actions(
                            n + '_viewpoint', include_no_op=False)
                    #print(view_actions)
                    place_actions.extend(view_actions)
            return place_actions
        
        # they are both clicked, it's time to pick_and_place!
        #pnp_actions = [self.action_component.pick_and_place_action()]
        pnp_actions = [self.env.pick_and_place_action(1)]
        return pnp_actions
