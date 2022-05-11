import random
import math

import numpy

from gym.spaces import Box

from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import SnapFinger
from ltron.matching import match_assemblies, compute_misaligned, matching_edges
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.geometry.utils import unscale_transform

class BuildExpert(LtronGymComponent):
    def __init__(self,
        action_component,
        scene_components,
        target_assembly_component,
        current_assembly_components,
        target_scene,
        shape_names,
        max_instructions=128,
        always_add_viewpoint_actions=False,
    ):
        self.scene_components = scene_components
        self.target_scene = target_scene
        self.target_assembly_component = target_assembly_component
        self.current_assembly_components = current_assembly_components
        self.action_component = action_component
        self.shape_names = shape_names
        self.max_instructions = max_instructions
        self.always_add_viewpoint_actions = always_add_viewpoint_actions
        
        num_actions = action_component.action_space.n
        self.observation_space = Box(
            low=numpy.zeros(self.max_instructions, dtype=numpy.long),
            high=numpy.full(
                self.max_instructions, num_actions-1, dtype=numpy.long),
            shape=(self.max_instructions,),
            dtype=numpy.long,
        )
    
    def reset(self):
        return self.observe()
    
    def step(self, action):
        observation = self.observe()
        return observation, 0., False, {}
    
    def observe(self):
        current_assembly = (
            self.current_assembly_components[self.target_scene].observe())
        secondary_assemblies = {
            name : component.observe()
            for name, component in self.current_assembly_components.items()
            if name != self.target_scene
        }
        target_assembly = self.target_assembly_component.observe()
        
        actions = self.good_actions(
            current_assembly, target_assembly, secondary_assemblies)
        
        random.shuffle(actions)
        actions = actions[:self.max_instructions]
        self.observation = numpy.zeros(self.max_instructions, dtype=numpy.long)
        self.observation[:len(actions)] = actions
        
        return self.observation
    
    def good_actions(self,
        current_assembly,
        target_assembly,
        secondary_assemblies
    ):
        matches, offset = match_assemblies(
            current_assembly, target_assembly, self.shape_names)
        current_to_target = dict(matches)
        target_to_current = {v:k for k,v in current_to_target.items()}
        #misaligned, false_positives, false_negatives = compute_misaligned(
        #    matches, current_assembly, target_assembly)
        (current_to_target_misaligned_connected,
         target_to_current_misaligned_connected,
         current_to_target_misaligned_disconnected,
         target_to_current_misaligned_disconnected,
         false_positives,
         false_negatives) = compute_misaligned(
            current_assembly, target_assembly, matches)
        
        # Is everything fine?
        if not (
            len(current_to_target_misaligned_connected) or
            len(current_to_target_misaligned_disconnected) or
            len(false_positives) or
            len(false_negatives)
        ):
            # Yes: phase
            actions = [self.action_component.finish_action()] # first check
        
        # Are there connected misaligned
        elif len(target_to_current_misaligned_connected):
            actions = self.adjust_connection(
                current_to_target,
                target_to_current,
                target_to_current_misaligned_connected,
                #current_to_target_misaligned_connected,
                current_assembly,
                target_assembly,
            )
        
        # Are there disconnected misaligned
        elif len(target_to_current_misaligned_disconnected):
            actions = self.make_connection(
                current_to_target,
                target_to_current,
                list(target_to_current_misaligned_disconnected.keys()),
                current_assembly,
                target_assembly,
                {self.target_scene:current_assembly},
            )
        
        # Are there false positives
        elif len(false_positives):
            # IGNORE THIS FOR NOW
            print('This should not happen yet')
            raise Exception
        
        elif len(false_negatives):
            actions = self.make_connection(
                current_to_target,
                target_to_current,
                false_negatives,
                current_assembly,
                target_assembly,
                secondary_assemblies,
            )
        
        return actions
    
    def compute_discrete_rotation(
        self,
        target_assembly,
        current_assembly,
        snap,
        target_instance,
        target_connected_instance,
        current_instance,
        current_connected_instance,
        rotation_steps = 4,
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
        inv_connected_current_transform = numpy.linalg.inv(
            connected_current_transform)
        current_offset = (
            numpy.linalg.inv(connected_current_transform) @
            current_transform
        )
        
        shape_index = target_assembly['shape'][target_instance]
        brick_shape_name = self.shape_names[shape_index]
        brick_shape = BrickShape(brick_shape_name)
        snap_transform = brick_shape.snaps[snap].transform
        inv_snap_transform = numpy.linalg.inv(snap_transform)
        current_snap_transform = current_transform @ snap_transform

        target_r = unscale_transform(target_offset)
        
        offsets = []
        for r in range(rotation_steps):
            c = math.cos(r * math.pi * 2 / rotation_steps)
            s = math.sin(r * math.pi * 2 / rotation_steps)
            ry = numpy.array([
                [ c, 0, s, 0],
                [ 0, 1, 0, 0],
                [-s, 0, c, 0],
                [ 0, 0, 0, 1],
            ])
            offset = (
                inv_connected_current_transform @
                current_snap_transform @
                ry @
                inv_snap_transform
            )
            offset_r = unscale_transform(offset)
            t = numpy.trace(offset_r[:3,:3].T @ target_r[:3,:3])
            offsets.append((t,r,offset))

        snap_style = brick_shape.snaps[snap]
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
                offset = (
                    inv_connected_current_transform @
                    current_snap_transform @
                    ry @
                    flip_rotation @
                    inv_snap_transform
                )
                offset_r = unscale_transform(offset)
                t = numpy.trace(offset_r[:3,:3].T @ target_r[:3,:3])
                offsets.append((t,r+rotation_steps,offset))

        return max(offsets)[1]

    def adjust_connection(self,
        current_to_target,
        target_to_current,
        targets_to_fix,
        current_assembly,
        target_assembly,
    ):
        actions = []
        
        '''
        # what shape/color combos are we looking for
        fn_shape_color_snaps = []
        for tgt_i, cur_set in targets_to_fix.items():
            shape = target_assembly['shape'][tgt_i]
            color = target_assembly['color'][tgt_i]
            
            #fn_shape_colors.add((shape, color))
            fn_edges = matching_edges(target_assembly, tgt_i)
            fn_edges = target_assembly['edges'][:,fn_edges]
            # EITHER A
            for _, tgt_con_i, tgt_s, tgt_con_s in fn_edges.T:
                if tgt_con_i in target_to_current:
                    fn_shape_color_snaps.append(
                        (shape, color, tgt_i, tgt_con_i, tgt_s, tgt_con_s))
            
            # OR B
            for _, tgt_con_i, tgt_s, tgt_con_s in fn_edges.T:
                if tgt_con_i in target_to_current:
                    for cur_i, cur_s, cur_con_s in cur_set:
                        if (thing, cur_s, cur_con_s) == (thing, tgt_s, tgt_con_s):
                            fn_shape_color_snaps.append(
                                (shape, color, tgt_i, tgt_con_i, tgt_s, tgt_con_s))
        '''
        
        '''
        # is a pick already clicked on?
        pickable = {}
        for s, c, tgt_i, tgt_con_i, tgt_s, tgt_con_s in fn_shape_color_snaps:
            for i in range(current_assembly['shape'].shape[0]):
                if i in current_to_target:
                    continue
                secondary_shape = current_assembly['shape'][i]
                secondary_color = current_assembly['color'][i]
                if secondary_shape == s and secondary_color == c:
                    pickable.setdefault((self.target_scene, i, tgt_s), [])
                    pickable[self.target_scene, i, tgt_s].append(
                        (self.target_scene, tgt_i, tgt_con_i, tgt_con_s))
        '''
        
        # pickable maps (n,i,s) in current assembly to [(n,i,ci,cs)] in target
        pickable = {}
        for tgt_i, cur_set in targets_to_fix.items():
            for cur_i, cur_s, cur_con_s in cur_set:
                fn_edge_indices = matching_edges(
                    target_assembly, i1=tgt_i, s1=cur_s, s2=cur_con_s)
                fn_edges = target_assembly['edges'][:,fn_edge_indices]
                if fn_edges.shape[1]:
                    tgt_con_i = fn_edges[1,0]
                    pickable[self.target_scene, cur_i, cur_s] = [
                        [self.target_scene, tgt_i, tgt_con_i, cur_con_s]]
        
        pick_component = self.action_component.components['pick_cursor']
        pick_n, pick_i, pick_s = pick_component.get_selected_snap()
        
        if (pick_n, pick_i, pick_s) not in pickable:
            pick_actions = []
            pick_names = []
            for n, i, s in pickable:
                pick_actions.extend(
                    self.action_component.actions_to_pick_snap(n, i, s)
                )
                pick_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(pick_actions):
                for n in pick_names:
                    view_actions = self.action_component.all_component_actions(
                        n + '_viewpoint')
                    print('supervising viewpoint (fr):', view_actions)
                    pick_actions.extend(view_actions)
            return pick_actions
        
        # it's already clicked, it's time to rotate!
        n, tgt_i, tgt_con_i, tgt_con_s = pickable[pick_n, pick_i, pick_s][0]
        r = self.compute_discrete_rotation(
            target_assembly,
            current_assembly,
            pick_s,
            tgt_i,
            tgt_con_i,
            pick_i,
            target_to_current[tgt_con_i],
        )
        
        rotate_actions = [self.action_component.rotate_action(r)]
        #if self.always_add_viewpoint_actions or not len(rotate_actions):
        #    rotate_actions.extend(
        #        self.action_component.all_component_actions(
        #            self.target_scene + '_viewpoint')
        #    )
        return rotate_actions
    
    def make_connection(self,
        current_to_target,
        target_to_current,
        targets_to_fix,
        current_assembly,
        target_assembly,
        secondary_assemblies,
    ):
        actions = []
        
        # what shape/color combos are we looking for
        fn_shape_color_snaps = []
        for target_to_fix in targets_to_fix:
            shape = target_assembly['shape'][target_to_fix]
            color = target_assembly['color'][target_to_fix]
            
            #fn_shape_colors.add((shape, color))
            fn_edges = matching_edges(target_assembly, target_to_fix)
            fn_edges = target_assembly['edges'][:,fn_edges]
            for _, tgt_con_i, tgt_s, tgt_con_s in fn_edges.T:
                if tgt_con_i in target_to_current:
                    fn_shape_color_snaps.append(
                        (shape, color, tgt_con_i, tgt_s, tgt_con_s))
        
        # is a pick already clicked on?
        pickable = {}
        for shape, color, tgt_con_i, tgt_s, tgt_con_s in fn_shape_color_snaps:
            for name, secondary_assembly in secondary_assemblies.items():
                for i in range(secondary_assembly['shape'].shape[0]):
                    if name == self.target_scene and i in current_to_target:
                        continue
                    secondary_shape = secondary_assembly['shape'][i]
                    secondary_color = secondary_assembly['color'][i]
                    if secondary_shape == shape and secondary_color == color:
                        pickable.setdefault((name, i, tgt_s), [])
                        pickable[name, i, tgt_s].append(
                            (self.target_scene, tgt_con_i, tgt_con_s))
        
        pick_component = self.action_component.components['pick_cursor']
        pick_n, pick_i, pick_s = pick_component.get_selected_snap()
        
        if (pick_n, pick_i, pick_s) not in pickable:
            pick_actions = []
            pick_names = []
            for n, i, s in pickable:
                pick_actions.extend(
                    #self.action_component.actions_to_place_snap(n, i, s)
                    self.action_component.actions_to_pick_snap(n, i, s)
                )
                pick_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(pick_actions):
                for n in pick_names:
                    view_actions = self.action_component.all_component_actions(
                            n + '_viewpoint')
                    print('supervising viewpoint(fpnp):', view_actions)
                    pick_actions.extend(view_actions)
            
            return pick_actions
        
        # is a place already clicked on?
        placeable = pickable[pick_n, pick_i, pick_s]
        place_component = self.action_component.components['place_cursor']
        place_n, place_i, place_s = place_component.get_selected_snap()
        if (place_n, place_i, place_s) not in placeable:
            place_actions = []
            place_names = []
            for n, i, s in placeable:
                place_actions.extend(
                    self.action_component.actions_to_place_snap(n, i, s)
                )
                place_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(place_actions):
                for n in place_names:
                    view_actions = self.action_component.all_component_actions(
                            n + '_viewpoint')
                    print('supervisting viewpoint(fpnp2):', view_actions)
                    place_actions.extend(view_actions)
            return place_actions
        
        # they are both clicked, it's time to pick_and_place!
        pnp_actions = [self.action_component.pick_and_place_action()]
        return pnp_actions
        
    def off(self):
        # Is a snap selected that needs to rotate?
        (pick_screen,
         pick_i,
         pick_s) = self.action_component.get_selected_pick_snap()
        pick_misaligned = (
            pick_screen == self.target_screen
            and pick_i in current_to_target_misaligned
        )
        if pick_misaligned:
            
            # Is pick_i, pick_s connected to anything?
            picked_edges = numpy.where(
                current_assembly['edges'][0] == pick_i &
                current_assembly['edges'][2] == pick_s
            )[0]
            if len(picked_edges):
                
                # Is the connected brick part of the current matching?
                con_i, con_s = current_assembly['edges'][[1,3], picked_edges]
                if con_i in current_to_target:
                    
                    # Does the current connection map to the target assembly?
                    tgt_con_i = current_to_target[con_i]
                    tgt_i = current_to_target_misaligned[pick_i]
                    e = [tgt_i, tgt_con_i, pick_s, con_s]
                    tgt_edges = numpy.where(
                        target_assembly['edges'][0] == e[0] &
                        target_assembly['edges'][1] == e[1] &
                        target_assembly['edges'][2] == e[2] &
                        target_assembly['edges'][3] == e[3]
                    )[0]
                    if len(tgt_edges):
                        
                        # Rotate
                        return (
                            self.action_component.ROTATE_ACTIONS(...) +
                            camera_actions
                        )
        
        # Are two snaps selected that need to be pick-and-placed?
        # (either we are in the same screen and misplaced
        #  or we are not and pick is a false negative)
        (place_screen,
         place_i,
         place_s) = self.action_component.get_selected_place_snap()
        if place_screen == target_screen and place_i in current_to_target:
            if pick_misaligned:
                
                # Does pick_i, place_i, pick_s, place_s map to a target edge?
                tgt_i = current_to_target_misaligned[pick_i]
                tgt_con_i = current_to_target[place_i]
                e = [tgt_i, tgt_con_i, pick_s, place_s]
                tgt_edges = numpy.where(
                    target_assembly['edges'][0] == e[0] &
                    target_assembly['edges'][1] == e[1] &
                    target_assembly['edges'][2] == e[2] &
                    target_assembly['edges'][3] == e[3]
                )[0]
                if len(tgt_edges):
                
                    # Pick and place
                    return (
                        self.action_component.PICK_AND_PLACE(...) +
                        camera_actions,
                    )
                
            if pick_screen != target_screen:
                pick_shape = hand_assembly['shape'][pick_i]
                pick_color = hand_assembly['color'][pick_i]
                
                # Is there a matching shape/color in the false negatives?
                for tgt_i in targets_to_fix:
                    if (target_assembly['shape'][tgt_i] == pick_shape and
                        target_assembly['color'][tgt_i] == pick_color
                    ):
                        tgt_con_i = current_to_target[place_i]
                        e = [tgt_i, tgt_con_i, pick_s, place_s]
                        tgt_edges = numpy.where(
                            target_assembly['edges'][0] == e[0] &
                            target_assembly['edges'][1] == e[1] &
                            target_assembly['edges'][2] == e[2] &
                            target_assembly['edges'][3] == e[3]
                        )[0]
                        if len(tgt_edges):
                            
                            # Pick and place
                            return (
                                self.action_component.PICK_AND_PLACE(...) +
                                camera_actions,
                            )
        
        # Is something misaligned in the scene?
        if len(misaligned):
            possible_actions = []
            for cur_i, tgt_i in misaligned:
                misaligned_edges = numpy.where(
                    current_assembly['edges'][0] == cur_i
                )[0]
                misaligned_edges = current_assembly['edges'][:,misaligned_edges]
                for edge in misaligned_edges.T:
                    cur_con_i, cur_s, cur_con_s = edge[1:]
                    if cur_con_i in current_to_target:
                        tgt_con_i = current_to_target[cur_con_i]
                        e = [tgt_i, tgt_con_i, cur_s, cur_con_s]
                        tgt_edges = numpy.where(
                            target_assembly['edges'][0] == e[0] &
                            target_assembly['edges'][1] == e[1] &
                            target_assembly['edges'][2] == e[2] &
                            target_assembly['edges'][3] == e[3]
                        )[0]
                        if len(tgt_edges):
                            
                            # Move cursor
                            actions = self.action_component.MOVE_PICK_CURSOR(
                                ...)
                            if actions:
                                possible_actions.extend(actions)
            
            if possible_actions:
                return possible_actions + camera_actions
            
            # Do we need to pick-and-place in the target scene?
            for cur_i, tgt_i in misaligned:
                target_edges = numpy.where(
                    target_assembly['edges'][0] == tgt_i
                )[0]
                target_edges = target_assembly['edges'][:,target_edges]
                for edge in target_edges.T:
                    tgt_con_i, cur_s, cur_con_s = edge[1:]
                    if tgt_con_i in target_to_current:
                        e = None
        
        # We need to pick-and-place across screens
        if not len(misaligned):
            pass

