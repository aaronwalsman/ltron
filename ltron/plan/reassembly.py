import random

import numpy

from PIL import Image

from ltron.bricks.brick_shape import BrickShape
from ltron.gym.reassembly_env import handspace_reassembly_template_action
from ltron.matching import match_configurations, match_lookup
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.visualization.drawing import stack_images_horizontal, write_text

class ExpertError(Exception):
    pass

class NothingRemovableError(ExpertError):
    pass

class UnfixableInstanceError(ExpertError):
    pass

class Off90RotationError(ExpertError):
    pass

class NoPlaceableTargetsError(ExpertError):
    pass

class NoUprightVisibleSnapsError(ExpertError):
    pass

class CantFindSnapError(ExpertError):
    pass

class TooManyRotatableSnapsError(ExpertError):
    pass

class ReassemblyExpert:
    def __init__(self, batch_size, shape_ids, color_ids):
        self.batch_size = batch_size
        self.shape_ids = shape_ids
        self.class_names = {value:key for key, value in shape_ids.items()}
        self.color_ids = color_ids
        self.color_names = {value:key for key, value in color_ids.items()}
        self.broken_seqs = {}
    
    def __call__(
        self,
        observations,
        check_collision=False,
        unfixable_mode='terminate',
        seq_ids=None,
        frame_ids=None
    ):
        num_observations = len_hierarchy(observations)
        actions = []
        statusses = []
        for i in range(num_observations):
            try:
                if seq_ids is None:
                    seq_id = None
                    frame_id = None
                else:
                    seq_id = seq_ids[i]
                    frame_id = frame_ids[i]
                action = self.act(
                    index_hierarchy(observations, i),
                    check_collision=check_collision,
                    unfixable_mode=unfixable_mode,
                    seq_id=seq_id,
                    frame_id=frame_id,
                )
                actions.append(action)
                statusses.append({'status':1})
            except ExpertError as e:
                action = handspace_reassembly_template_action()
                action['reassembly']['end'] = 1
                actions.append(action)
                statusses.append({'status':0})
                
                type_str = str(type(e))
                if type_str not in self.broken_seqs:
                    self.broken_seqs[type_str] = 0
                self.broken_seqs[type_str] += 1
                for c, n in self.broken_seqs.items():
                    print('%s: %i'%(str(c), n))
                
        return actions, statusses
    
    def act(
        self,
        observation,
        check_collision=False,
        unfixable_mode='terminate',
        seq_id=None,
        frame_id=None,
    ):
        observation['workspace_visualization'] = (
            observation['workspace_color_render'].copy())
        observation['handspace_visualization'] = (
            observation['handspace_color_render'].copy())
        
        try:
            print('='*80)
            print('act')
            if not observation['reassembly']['reassembling']:
                return self.disassembly_step(
                    observation,
                    check_collision=check_collision,  
                )
            else:
                return self.reassembly_step(
                    observation,
                    check_collision=check_collision,
                    unfixable_mode=unfixable_mode,
                )
        
        except ExpertError as e:
            observation['workspace_visualization'] = write_text(
                observation['workspace_visualization'], str(type(e)))
            raise
        
        finally:
            visualization = stack_images_horizontal(
                (observation['workspace_visualization'],
                 observation['handspace_visualization']),
                align='bottom')
            Image.fromarray(visualization).save(
                './vis_%i_%i.png'%(seq_id, frame_id))
    
    def disassembly_step(self, observation, check_collision=False):
        print('-- disassembly_step')
        
        # If there are still items in the workspace, pick one to remove and 
        workspace_config = observation['reassembly']['workspace_configuration']
        if numpy.any(workspace_config['class']):
            instance_to_remove = self.choose_instance_to_remove(observation)
            return self.disassemble_instance_action(
                observation, instance_to_remove)
        
        # If there is nothing left to remove, switch to reassembly
        else:
            return self.switch_to_reassembly_action()
    
    def choose_instance_to_remove(self, observation):
        print('---- choose instance to remove')
        
        # Figure out what can be removed.
        visible_instances = numpy.unique(
            observation['workspace_segmentation_render'])
        
        # TODO: collisions
        
        # TMP
        removable_instances = [i for i in visible_instances if i != 0]
        if not len(removable_instances):
            raise NothingRemovableError
        instance_to_remove = random.choice(removable_instances)
        
        return instance_to_remove
    
    def disassemble_instance_action(self, observation, instance_to_remove):
        print('---- disassemble instance action')
        
        # Initialize the action.
        action = handspace_reassembly_template_action()
        
        # Pick a snap corresponding to the instance.
        # TODO: collisions.
        pos_snaps = observation['workspace_pos_snap_render']
        neg_snaps = observation['workspace_neg_snap_render']
        pick_y, pick_x, polarity = self.select_from_pos_neg_maps(
            pos_snaps[:,:,0] == instance_to_remove,
            neg_snaps[:,:,0] == instance_to_remove,
        )
        
        # Fill in the action entries and return.
        action['disassembly']['activate'] = True
        action['disassembly']['polarity'] = polarity
        action['disassembly']['direction'] = 0 # TMP
        action['disassembly']['pick'] = numpy.array(
            (pick_y, pick_x), dtype=numpy.long)
        return action
    
    def switch_to_reassembly_action(self):
        print('---- switch to reassembly action')
        # Initialize the action, fill in the entries and return.
        action = handspace_reassembly_template_action()
        action['reassembly']['start'] = 1
        return action
    
    def reassembly_step(
        self,
        observation,
        check_collision=False,
        unfixable_mode='terminate',
    ):
        print('-- reassembly step')
        
        # Pull out the configurations.
        workspace_config = observation['reassembly']['workspace_configuration']
        handspace_config = observation['reassembly']['handspace_configuration']
        target_config = observation['reassembly']['target_configuration']
        
        # Compute matching between workspace and target.
        matching = match_configurations(workspace_config, target_config)
        
        '''
        # Compute what is missing from the target (false negatives)
        # and what is extra in the workspace (false positives).
        workspace_to_target = {a:b for a, b in matching}
        target_to_workspace = {b:a for a, b in matching}
        workspace_instances = numpy.where(workspace_config['class'] != 0)[0]
        misplaced_workspace_instances = [
            a for a in workspace_instances if a not in workspace_to_target]
        target_instances = numpy.where(target_config['class'] != 0)[0]
        unplaced_target_instances = [
            b for b in target_instances if b not in target_to_workspace]
        '''
        (workspace_to_target,
         target_to_workspace,
         misplaced_workspace_instances,
         unplaced_target_instances) = match_lookup(
            matching, workspace_config, target_config)
        
        # Compute which misplaced instances in the workspace are fixable
        # using low-level actions.
        fixable_instances = []
        unfixable_instances = []
        misplaced_rotatable_snaps = []
        misplaced_target_matches = []
        unplaced_classes = target_config['class'][
            list(unplaced_target_instances)]
        unplaced_colors = target_config['color'][
            list(unplaced_target_instances)]
        unplaced_class_colors = set(zip(unplaced_classes, unplaced_colors))
        for misplaced_instance in misplaced_workspace_instances:
            rotatable_snaps, target_match = self.find_misplaced_rotatable_snaps(
                misplaced_instance,
                workspace_config,
                target_config,
                workspace_to_target,
                target_to_workspace,
                unplaced_class_colors,
            )
            if len(rotatable_snaps):
                fixable_instances.append(misplaced_instance)
                misplaced_rotatable_snaps.append(rotatable_snaps)
                misplaced_target_matches.append(target_match)
            else:
                unfixable_instances.append(misplaced_instance)
        
        if len(unfixable_instances):
            
            # TODO: All of this basically
            #raise NotImplementedError('Unfixable instance, come back to this')
            raise UnfixableInstanceError
            
            if unfixable_mode == 'terminate':
                action = handspace_resassembly_template_action()
                action['reassembly']['end'] = True
                return action
            
            elif unfixable_mode == 'remove':
                removable_instances = []
                for unfixable_intance in unfixable_instances:
                    # TODO: collision detection
                    pass
                
                if len(removable_instances):
                    instance_to_remove = random.choice(removable_unfixable)
        
        if len(fixable_instances):
            r1 = random.randint(0, len(fixable_instances)-1)
            instance_to_fix = fixable_instances[r1]
            rotatable_snaps = misplaced_rotatable_snaps[r1]
            target_matches = misplaced_target_matches[r1]
            
            # TODO: visibility?
            r2 = random.randint(0, len(rotatable_snaps)-1)
            snap_to_rotate = rotatable_snaps[r2]
            misplaced_target_id, connected_target_id = target_matches[r2]
            connected_workspace_id = target_to_workspace[connected_target_id]
            
            misplaced_class = workspace_config['class'][misplaced_instance]
            brick_shape = BrickShape(self.class_names[misplaced_class])
            pose_to_fix = workspace_config['pose'][instance_to_fix]
            snap_transform = brick_shape.snaps[snap_to_rotate].transform
            inv_snap_transform = numpy.linalg.inv(snap_transform)
            instance_snap_transform = pose_to_fix @ snap_transform
            
            connected_transform = workspace_config['pose'][
                connected_workspace_id]
            inv_connected_transform = numpy.linalg.inv(connected_transform)
            
            ry0 = numpy.array([
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            ry1 = numpy.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            ry2 = numpy.array([
                [-1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0,-1, 0],
                [ 0, 0, 0, 1],
            ])
            offset_to_r0 = (
                inv_connected_transform @
                instance_snap_transform @
                ry0 @
                inv_snap_transform
            )
            offset_to_r1 = (
                inv_connected_transform @
                instance_snap_transform @
                ry1 @
                inv_snap_transform
            )
            offset_to_r2 = (
                inv_connected_transform @
                instance_snap_transform @
                ry2 @
                inv_snap_transform
            )
            
            connected_target_pose = target_config['pose'][connected_target_id]
            target_offset = (
                numpy.linalg.inv(connected_target_pose) @
                target_config['pose'][misplaced_target_id]
            )
            
            if numpy.allclose(offset_to_r0, target_offset):
                direction = 1
            elif numpy.allclose(offset_to_r1, target_offset):
                direction = 0
            elif numpy.allclose(offset_to_r2, target_offset):
                direction = random.randint(0,1)
            else:
                #assert False, 'this should never happen (yet)'
                raise Off90RotationError
            
            pos_snaps = observation['workspace_pos_snap_render']
            neg_snaps = observation['workspace_neg_snap_render']
            
            pick_y, pick_x, pick_p = self.select_from_pos_neg_maps(
                (pos_snaps[:,:,0] == misplaced_instance) &
                (pos_snaps[:,:,1] == snap_to_rotate),
                (neg_snaps[:,:,0] == misplaced_instance) &
                (neg_snaps[:,:,1] == snap_to_rotate),
            )
            
            action = handspace_reassembly_template_action()
            action['rotate']['activate'] = True
            action['rotate']['pick'] = numpy.array(
                (pick_y, pick_x), dtype=numpy.long)
            action['rotate']['polarity'] = pick_p
            action['rotate']['direction'] = direction
            return action
        
        # If there are no misplaced bricks, and no unplaced bricks then the
        # model is complete and we can terminate the sequence.
        if len(unplaced_target_instances) == 0:
            
            print('end')
            
            action = handspace_reassembly_template_action()
            action['reassembly']['end'] = True
            return action
        
        # Is there anything in the hand?
        handspace_class = handspace_config['class'][1]
        print('handspace class')
        print(handspace_class)
        if handspace_class != 0:
            # Can we place it?
            # TODO: FIGURE OUT IF WE CAN PLACE THE THING IN THE HAND
            if numpy.any(workspace_config['class']):
                return self.nth_brick_placement_action(
                    observation,
                    unplaced_target_instances,  
                    target_to_workspace,
                )
            else:
                return self.first_brick_placement_action(observation)
        
        # If there's nothing in the hand, or the hand brick is wrong,
        # figure out what to pick up next.
        if numpy.any(workspace_config['class']):
            placeable_target_instances = []
            for unplaced_instance in unplaced_target_instances:
                target_edges = target_config['edges']
                unplaced_edges = numpy.where(
                    target_edges[0] == unplaced_instance)[0]
                connected_instances = target_edges[1,unplaced_edges]
                connections_in_workspace = [
                    i in target_to_workspace for i in connected_instances]
                if any(connections_in_workspace):
                    if check_collision:
                        # TODO: check collisions
                        pass
                    else:
                        placeable_target_instances.append(unplaced_instance)
        
        # If there's nothing in the workspace yet, anything with upright snaps
        # is ok to start with.
        else:
            all_target_instances = numpy.where(target_config['class'] != 0)[0]
            placeable_target_instances = []
            brick_shapes = {}
            for target_instance in all_target_instances:
                instance_class = target_config['class'][target_instance]
                instance_class_name = self.class_names[instance_class]
                if instance_class_name not in brick_shapes:
                    brick_shapes[instance_class_name] = BrickShape(
                        instance_class_name)
                brick_shape = brick_shapes[instance_class_name]
                pose = target_config['pose'][target_instance]
                upright_snaps = self.upright_snaps(brick_shape, pose)
                if len(upright_snaps):
                    placeable_target_instances.append(target_instance)
            
        #assert len(placeable_target_instances), (
        #    'No placeable instances, there may be no upright target instances')
        if not len(placeable_target_instances):
            raise NoPlaceableTargetsError
        
        return self.pick_target_instance(
            target_config, placeable_target_instances)
    
    def find_misplaced_rotatable_snaps(
        self,
        misplaced_instance,
        workspace_config,
        target_config,
        workspace_to_target,
        target_to_workspace,
        unplaced_class_colors,
    ):
        print('---- find misplaced rotatable snaps')
        
        # If it's not even a good class/color combo, then it's not fixable.
        misplaced_class = workspace_config['class'][misplaced_instance]
        misplaced_color = workspace_config['color'][misplaced_instance]
        if (misplaced_class, misplaced_color) not in unplaced_class_colors:
            #unfixable_instances.append(misplaced_instance)
            #continue
            return []
        
        # Can the instance be rotated into place?
        misplaced_edges = workspace_config['edges'][0] == misplaced_instance
        connected_instances = numpy.unique(
            workspace_config['edges'][1,misplaced_edges])
        rotation_snaps = []
        target_match = []
        for i, connected_instance in enumerate(connected_instances):
            # If the connected instance is not a true positive, we should
            # not consider rotating about a shared connection point between
            # them.
            if connected_instance not in workspace_to_target:
                continue
            
            # Does the target configuration have any connections between
            # the connected instance and a class matching our misplaced
            # brick about the matching snaps?
            # TODO: SYMMETRY
            connected_target = workspace_to_target[connected_instance]
            connected_target_edges = (
                target_config['edges'][0] == connected_target)
            candidate_edges = (
                target_config['edges'][:,connected_target_edges])
            candidate_classes = (
                target_config['class'][candidate_edges[1]])
            
            instance_matching_snaps = []
            instance_target_matches = []
            for e, c in zip(candidate_edges.T, candidate_classes):
                if c != misplaced_class:
                    continue
                
                # Skip this brick if it is already accounted for in the current
                # matching.
                if e[1] in target_to_workspace:
                    continue
                
                workspace_edge_start = connected_instance
                workspace_edge_end = misplaced_instance
                candidate_snaps = e[2:]
                # TODO: SYMMETRY
                
                matching_workspace_edges = numpy.where(
                    (workspace_config['edges'][0] == connected_instance) &
                    (workspace_config['edges'][1] == misplaced_instance) &
                    (workspace_config['edges'][2] == e[2]) &
                    (workspace_config['edges'][3] == e[3])
                )[0]
                matching_workspace_snaps = (
                    workspace_config['edges'][3, matching_workspace_edges])
                instance_matching_snaps.extend(matching_workspace_snaps)
                if matching_workspace_snaps.shape[0]:
                    instance_target_matches.append((e[1], e[0]))
            
            # There should only be one correct snap between the misplaced
            # and candidate bricks, otherwise, the misplaced brick wouldn't
            # be misplaced.
            #try:
            #    assert len(set(instance_matching_snaps)) < 2
            #except:
            #    import pdb
            #    pdb.set_trace()
            if len(set(instance_matching_snaps)) >= 2:
                raise TooManyRotatableSnapsError
            
            rotation_snaps.extend(instance_matching_snaps)
            target_match.extend(instance_target_matches)
            
        return rotation_snaps, target_match
    
    def pick_target_instance(
        self,
        target_config,
        placeable_target_instances,
    ):
        print('---- pick target instance')
        instance_to_pick = random.choice(placeable_target_instances)
        class_to_pick = target_config['class'][instance_to_pick]
        color_to_pick = target_config['color'][instance_to_pick]
        action = handspace_reassembly_template_action()
        action['insert_brick']['shape_id'] = class_to_pick
        action['insert_brick']['color_id'] = color_to_pick
        return action
    
    def first_brick_placement_action(self, observation):
        print('---- first brick placement action')
        
        # Initialize the action.
        action = handspace_reassembly_template_action()
        
        # Pull out configurations.
        target_config = observation['reassembly']['target_configuration']
        handspace_config = observation['reassembly']['handspace_configuration']
        
        # Find the snaps that are aligned with the brick's up direction when
        # placed in the scene.
        brick_class = handspace_config['class'][1]
        class_name = self.class_names[brick_class]
        brick_shape = BrickShape(class_name)
        matching_target_instances = numpy.where(
            target_config['class'] == brick_class)[0]
        target_poses = target_config['pose'][matching_target_instances]
        potential_instances = []
        for instance, pose in zip(matching_target_instances, target_poses):
            upright_snaps = self.upright_snaps(brick_shape, pose)
            if len(upright_snaps):
                potential_instances.append((instance, upright_snaps))
        
        # Pick an upright visible snap
        picked_instance, upright_snaps = random.choice(potential_instances)
        pos_snaps = observation['handspace_pos_snap_render']
        neg_snaps = observation['handspace_neg_snap_render']
        visible_snaps = (
            set(numpy.unique(pos_snaps[:,:,1])) |
            set(numpy.unique(neg_snaps[:,:,1]))
        )
        upright_visible_snaps = (visible_snaps & set(upright_snaps)) - set([0])
        #assert len(upright_visible_snaps), (
        #    'Somehow none of the upright snaps are visible')
        if not len(upright_visible_snaps):
            raise NoUprightVisibleSnapsError
        picked_snap = random.choice(list(upright_visible_snaps))
        
        # Pick one of the snap pixels.
        pick_y, pick_x, pick_p = self.select_from_pos_neg_maps(
            (pos_snaps[:,:,0] != 0) & (pos_snaps[:,:,1] == picked_snap),
            (neg_snaps[:,:,0] != 0) & (neg_snaps[:,:,1] == picked_snap),
        )
        
        # Fill in the action entries and return.
        action['pick_and_place']['activate'] = True
        action['pick_and_place']['polarity'] = pick_p
        action['pick_and_place']['pick'] = numpy.array(
            (pick_y, pick_x), dtype=numpy.long)
        action['pick_and_place']['place_at_origin'] = True
        return action
    
    def nth_brick_placement_action(
        self,
        observation,
        unplaced_target_instances,
        target_to_workspace,
    ):
        print('---- nth brick placement action')
        
        # TODO:
        # need some combination of workspace_to_target, target_to_workspace,
        # unplaced_target_instances, etc.
        
        # Initialize the action.
        action = handspace_reassembly_template_action()
        
        # Pull out configurations.
        target_config = observation['reassembly']['target_configuration']
        handspace_config = observation['reassembly']['handspace_configuration']
        workspace_config = observation['reassembly']['workspace_configuration']
        
        # Find a place for the brick in the hand by assigning it to
        # an unplaced target brick.
        brick_class = handspace_config['class'][1]
        candidate_target_instances = numpy.array([
            t for t in unplaced_target_instances
            if target_config['class'][t] == brick_class
        ])
        
        expanded_edges = target_config['edges'][0].reshape(-1,1)
        edge_eq_candidate = (
            expanded_edges == candidate_target_instances.reshape(1,-1))
        edge_ids, candidate_ids = numpy.where(edge_eq_candidate)
        candidate_ids = candidate_target_instances[candidate_ids]
        target_ids = target_config['edges'][1, edge_ids]
        handspace_snaps = target_config['edges'][2, edge_ids]
        workspace_snaps = target_config['edges'][3, edge_ids]
        
        assert len(candidate_ids), (
            'No potential connections between the hand and the workspace')
        
        potential_workspace_connections = [
            (c_id, target_to_workspace[t_id], h_snap, w_snap)
            for c_id, t_id, h_snap, w_snap
            in zip(candidate_ids, target_ids, handspace_snaps, workspace_snaps)
            if t_id in target_to_workspace
        ]
        
        # TODO: Filter for visible snaps.
        
        # At this point potential_workspace_connections has the structure:
        # [(candidate_id, workspace_id, handspace_snap, workspace_snap), ...]
        # where each candidate_id is a target instance index corresponding to
        # one location where we might place the brick in the hand, and
        # workspace_id is the brick we might connect it to using handspace_snap
        # and workspace_snap as connection points.
        
        c_id, w_id, h_snap, w_snap = random.choice(
            potential_workspace_connections)
        
        pos_handspace_snaps = observation['handspace_pos_snap_render']
        neg_handspace_snaps = observation['handspace_neg_snap_render']
        pos_workspace_snaps = observation['workspace_pos_snap_render']
        neg_workspace_snaps = observation['workspace_neg_snap_render']
        
        # Sample a pick location.
        pos_handspace_map = (
            (pos_handspace_snaps[:,:,0] == 1) &
            (pos_handspace_snaps[:,:,1] == h_snap)
        )
        neg_handspace_map = (
            (neg_handspace_snaps[:,:,0] == 1) &
            (neg_handspace_snaps[:,:,1] == h_snap)
        )
        pick_y, pick_x, pick_p = self.select_from_pos_neg_maps(
            pos_handspace_map, neg_handspace_map)
        
        # Sample a place location.
        pos_workspace_map = (
            (pos_workspace_snaps[:,:,0] == w_id) &
            (pos_workspace_snaps[:,:,1] == w_snap)
        )
        neg_workspace_map = (
            (neg_workspace_snaps[:,:,0] == w_id) &
            (neg_workspace_snaps[:,:,1] == w_snap)
        )
        
        place_y, place_x, place_p = self.select_from_pos_neg_maps(
            pos_workspace_map, neg_workspace_map)
        
        # Fill in the action entries and return.
        action = handspace_reassembly_template_action()
        action['pick_and_place']['activate'] = True
        action['pick_and_place']['polarity'] = pick_p
        action['pick_and_place']['pick'] = numpy.array((pick_y, pick_x))
        action['pick_and_place']['place'] = numpy.array((place_y, place_x))
        
        return action
    
    def upright_snaps(self, brick_shape, pose):
        # Get the up (+y) direction for each snap and compare it against the
        # up (-y) direction of the pose.  Any of these 
        snap_ys = [snap.transform[:3,1] for snap in brick_shape.snaps]
        # I think this is wrong
        # pose_y = pose[:3,1]
        # I think this is right
        pose_y = pose[1,:3]
        snap_ys = [snap.transform[1,:3] for snap in brick_shape.snaps]
        alignment = [-snap_y @ pose_y for snap_y in snap_ys]
        upright_snaps = [i for i, a in enumerate(alignment) if a > 0.99]
        
        return upright_snaps
    
    def select_from_pos_neg_maps(self, pos_map, neg_map):
        pos_y, pos_x = numpy.where(pos_map)
        neg_y, neg_x = numpy.where(neg_map)
        y = numpy.concatenate((pos_y, neg_y))
        x = numpy.concatenate((pos_x, neg_x))
        p = numpy.concatenate((
            numpy.ones(pos_y.shape, dtype=numpy.long),
            numpy.zeros(neg_y.shape, dtype=numpy.long),
        ))
        
        try:
            r = random.randint(0, y.shape[0]-1)
        except:
            raise CantFindSnapError
        
        pick_y = y[r]
        pick_x = x[r]
        pick_p = p[r]
        return pick_y, pick_x, pick_p
