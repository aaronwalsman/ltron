import random
import math

import numpy

from gym.spaces import Box

from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import SnapFinger, UnsupportedSnap
from ltron.matching import match_assemblies, compute_misaligned, matching_edges
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.geometry.utils import unscale_transform

class BuildExpert(LtronGymComponent):
    def __init__(self,
        env,
        scene_components,
        target_assembly_component,
        current_assembly_components,
        target_scene,
        disassembly_scene,
        shape_ids,
        max_instructions=128,
        shuffle_instructions=True,
        always_add_viewpoint_actions=False,
        terminate_on_empty=False,
        max_actions=1000000,
    ):
        # store variables
        self.env = env
        self.scene_components = scene_components
        self.target_scene = target_scene
        self.disassembly_scene = disassembly_scene
        self.target_assembly_component = target_assembly_component
        self.current_assembly_components = current_assembly_components
        self.shape_names = {v:k for k,v in shape_ids.items()}
        self.max_instructions = max_instructions
        self.shuffle_instructions = shuffle_instructions
        self.always_add_viewpoint_actions = always_add_viewpoint_actions
        self.terminate_on_empty = terminate_on_empty
        
        # build observation space
        self.observation_space = Box(
            low=numpy.zeros(self.max_instructions, dtype=numpy.long),
            high=numpy.full(
                self.max_instructions, max_actions, dtype=numpy.long),
            shape=(self.max_instructions,),
            dtype=numpy.long,
        )
    
    def reset(self):
        return self.observe()
    
    def step(self, action):
        observation = self.observe()
        if self.terminate_on_empty:
            terminal = numpy.sum(observation) == 0
        else:
            terminal = False
        return observation, 0., terminal, {}
    
    def observe(self):
        # get assemblies
        current_assembly = (
            self.current_assembly_components[self.target_scene].observe())
        secondary_assemblies = {
            name : component.observe()
            for name, component in self.current_assembly_components.items()
            if name != self.target_scene
        }
        target_assembly = self.target_assembly_component.observe()
        
        # compute the expert actions
        actions = self.expert_actions(
            current_assembly, target_assembly, secondary_assemblies)
        
        # convert the expert actions to a truncated list of instructions
        if self.shuffle_instructions:
            random.shuffle(actions)
        actions = actions[:self.max_instructions]
        self.observation = numpy.zeros(self.max_instructions, dtype=numpy.long)
        self.observation[:len(actions)] = actions
        
        # return
        return self.observation
    
    def expert_actions(self,
        current_assembly,
        target_assembly,
        secondary_assemblies,
    ):
        
        # match the current and target assemblies
        matches, offset = match_assemblies(
            current_assembly, target_assembly, self.shape_names)
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
        
        # if the current assembly matches the target assembly,
        # then finish (switch phase)
        if not (
            len(current_to_target_misaligned_connected) or
            len(current_to_target_misaligned_disconnected) or
            len(false_positives) or
            len(false_negatives)
        ):
            actions = self.env.finish_actions() # first check
        
        # if there are bricks that are incorrectly placed, but have a correct
        # connection, adjust the connection
        elif len(target_to_current_misaligned_connected):
            actions = self.adjust_connection(
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
                {self.target_scene:current_assembly},
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
            return []
        
        # if the current scene is empty, add the first brick
        elif not len(current_to_target):
            actions = self.add_first_brick(
                false_negatives,
                target_assembly,
                secondary_assemblies,
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
        
        #pick_component = self.env.components['pick_cursor']
        #pick_n, pick_i, pick_s = pick_component.get_selected_snap()
        pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        if (pick_n, pick_i, pick_s) not in pickable:
            pick_actions = []
            pick_names = []
            for n, i, s in pickable:
                pick_actions.extend(
                    self.env.actions_to_pick_snap(n, i, s)
                )
                pick_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(pick_actions):
                print('supervising viewpoint (fr):')
                import pdb
                pdb.set_trace()
                for n in pick_names:
                    view_actions = self.env.all_component_actions(
                        n + '_viewpoint')
                    print(view_actions)
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
        
        #rotate_actions = [self.action_component.rotate_action(r)]
        rotate_actions = [self.env.rotate_action(r)]
        #if self.always_add_viewpoint_actions or not len(rotate_actions):
        #    rotate_actions.extend(
        #        self.action_component.all_component_actions(
        #            self.target_scene + '_viewpoint')
        #    )
        return rotate_actions
    
    
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
                    self.env.actions_to_insert_brick(shape, color))
            
            return insert_actions
        
        #pick_cursor = self.action_component.components['pick_cursor']
        #pick_cursor = self.env.components['pick_cursor']
        #pick_n, pick_i, pick_s = pick_cursor.get_selected_snap()
        pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        # if a pickable thing is not picked, pick it
        if (pick_n, pick_i, pick_s) not in pickable:
            pick_actions = []
            pick_names = []
            for n, i, s in pickable:
                pick_actions.extend(
                    self.env.actions_to_pick_snap(n, i, s)
                )
                pick_names.append(n)

            if self.always_add_viewpoint_actions or not len(pick_actions):
                print('supervising viewpoint(first):')
                import pdb
                pdb.set_trace()
                for n in pick_names:
                    #view_actions = self.action_component.all_component_actions(
                    view_actions = self.env.all_component_actions(
                            n + '_viewpoint')
                    print(view_actions)
                    pick_actions.extend(view_actions)

            return pick_actions
        
        #place_cursor = self.action_component.components['place_cursor']
        #place_n, place_i, place_s = place_cursor.get_selected_snap()
        place_n, place_i, place_s = self.env.get_place_snap()
        #place_n = 'hand'
        #place_i = 1
        #palce_s = 0
        
        # if anything is selected in the place cursor, deselect it
        #if place_i != 0 or place_n != self.target_scene:
        if place_n != self.target_scene:
            #place_actions = self.action_component.actions_to_deselect_place()
            place_actions = self.env.actions_to_deselect_place(
                self.target_scene)
            
            if self.always_add_viewpoint_actions:
                print('supervising viewpoint(first2):')
                import pdb
                pdb.set_trace()
                # TODO
            
            return place_actions
        
        # clicks are correct, it's time to pick_and_place!
        #pnp_actions = [self.action_component.pick_and_place_action()]
        pnp_actions = [self.env.pick_and_place_action(2)]
        return pnp_actions
    
    
    def remove_false_positive(self,
        current_to_target,
        target_to_current,
        false_positives,
        current_assembly,
        target_assembly,
    ):
        actions = []
        return actions
        
        # what shap/color combos are we trying to remove
        fp_shape_color_snaps = []
        
        import pdb
        pdb.set_trace()
        # CONTINUE HERE
    
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
                        cur_con_i = target_to_current[tgt_con_i]
                        pickable[name, i, tgt_s].append(
                            #(self.target_scene, tgt_con_i, tgt_con_s))
                            (self.target_scene, cur_con_i, tgt_con_s))
        
        # if there are no pickable things, add the brick
        if not len(pickable):
            # TODO, need to filter this based on bricks that can be added next
            insert_actions = []
            for shape, color, i, tgt_s, tgt_con_s in fn_shape_color_snaps:
                insert_actions.append(
                    self.env.actions_to_insert_brick(shape, color))
            
            return insert_actions
        
        #pick_component = self.action_component.components['pick_cursor']
        #pick_n, pick_i, pick_s = pick_component.get_selected_snap()
        pick_n, pick_i, pick_s = self.env.get_pick_snap()
        
        if (pick_n, pick_i, pick_s) not in pickable:
            pick_actions = []
            pick_names = []
            for n, i, s in pickable:
                pick_actions.extend(
                    #self.action_component.actions_to_place_snap(n, i, s)
                    #self.action_component.actions_to_pick_snap(n, i, s)
                    self.env.actions_to_pick_snap(n, i, s)
                )
                pick_names.append(n)
            
            if self.always_add_viewpoint_actions or not len(pick_actions):
                print('supervising viewpoint(fpnp):')
                import pdb
                pdb.set_trace()
                for n in pick_names:
                    #view_actions = self.action_component.all_component_actions(
                    view_actions = self.env.all_component_actions(
                            n + '_viewpoint')
                    print(view_actions)
                    pick_actions.extend(view_actions)
            
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
                print('supervising viewpoint(fpnp2):')
                import pdb
                pdb.set_trace()
                for n in place_names:
                    #view_actions = self.action_component.all_component_actions(
                    view_actions = self.env.all_component_actions(
                            n + '_viewpoint')
                    print(view_actions)
                    place_actions.extend(view_actions)
            return place_actions
        
        # they are both clicked, it's time to pick_and_place!
        #pnp_actions = [self.action_component.pick_and_place_action()]
        pnp_actions = [self.env.pick_and_place_action(1)]
        return pnp_actions
