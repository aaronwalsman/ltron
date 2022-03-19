import random
import math
import copy
from bisect import insort

import numpy

from splendor.image import save_image

from ltron.bricks.snap import SnapFinger
from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.brick_instance import BrickInstance
from ltron.exceptions import LtronException
from ltron.geometry.utils import default_allclose, unscale_transform

class NoMatchingRotationError(LtronException):
    pass

class PickAndPlaceFailure(LtronException):
    pass

class RotationFailure(LtronException):
    pass

class DisassemblyFailure(LtronException):
    pass

class NoVisibleSnapsFailure(LtronException):
    pass

class DoubleSnapConnectionFailure(LtronException):
    pass

# utilities ====================================================================

def snap_finder_condition(
    instances,
    snaps,
    pos_component,
    neg_component,
    mask_component=None,
):
    # build a condition function which takes an observation and finds where
    # the specified instances and snaps are visible in the current scene
    def condition(observation):
        matching_yxpis = []
        for pp, render_name in enumerate((neg_component, pos_component)):
            for ii, ss in zip(instances, snaps):
                render = observation[render_name]
                if mask_component is not None:
                    mask_render = observation[mask_component]
                    y, x = numpy.where(
                        (render[:,:,0] == ii) &
                        (render[:,:,1] == ss) &
                        (mask_render[:,:] == ii)
                    )
                else:
                    y, x = numpy.where(
                        (render[:,:,0] == ii) & (render[:,:,1] == ss))
                p = numpy.ones(y.shape[0], dtype=numpy.long) * pp
                i = numpy.ones(y.shape[0], dtype=numpy.long) * ii
                s = numpy.ones(y.shape[0], dtype=numpy.long) * ss
                yxpis = numpy.stack((y, x, p, i, s), axis=0)
                matching_yxpis.append(yxpis)
        
        matching_yxpis = numpy.concatenate(matching_yxpis, axis=1)
        
        success = bool(matching_yxpis.shape[1])
        return success, matching_yxpis
    return condition

def modular_distance(a,b,m):
    d0 = abs(a-b)
    d1 = abs(a+m-b)
    d2 = abs(b+m-a)
    return min(d0, d1, d2)

def search_camera_space(env, component_name, state, condition, max_steps):
        
    # BFS
    component = env.components[component_name]
    current_position = tuple(component.position) + (0,)
    explored = set()
    explored.add(current_position)
    
    frontier = [(0,) + current_position]
    min_position = (0,0,0,0)
    max_position = (
        component.azimuth_steps-1,
        component.elevation_steps-1,
        component.distance_steps-1,
        1,
    )
    
    while frontier:
        distance, *position = frontier.pop(0)
        position = tuple(position)
        
        test_result, test_info = test_camera_position(
            env, position, component_name, state, condition
        )
        if test_result:
            # reset state
            env.set_state(state)
            return position, test_info
        
        for i in range(4):
            for direction in (-1, 1):
                offset = numpy.zeros(4, dtype=numpy.long)
                offset[i] += direction
                new_position = position + offset
                new_position = numpy.concatenate((
                    new_position[[0]] % component.azimuth_steps,
                    new_position[1:]
                ))
                new_distance = numpy.sum(
                    numpy.abs(new_position[1:] - current_position[1:]))
                new_distance += modular_distance(
                    new_position[0],
                    current_position[0],
                    component.azimuth_steps,
                )
                if (numpy.all(new_position >= min_position) and
                    numpy.all(new_position <= max_position) and
                    new_distance <= max_steps and
                    tuple(new_position) not in explored
                ):
                    new_position = tuple(new_position)
                    explored.add(new_position)
                    new_distance_position = (new_distance,) + new_position
                    insort(frontier, new_distance_position)
    
    # reset state
    env.set_state(state)
    return None, None

def test_camera_position(env, position, component_name, state, condition):
    new_state = copy.deepcopy(state)
    replace_camera_in_state(env, new_state, component_name, position)
    observation = env.set_state(new_state)
    return condition(observation)

def replace_camera_in_state(env, state, component_name, position):
    state[component_name]['position'] = position[:3]
    if position[-1]:
        component = env.components[component_name]
        center = component.compute_center()
        state[component_name]['center'] = center

def plan_camera_to_see_snaps(
    env,
    state,
    instances,
    snaps,
    pos_snap_component,
    neg_snap_component,
    viewpoint_component,
    mask_component=None,
):
    condition = snap_finder_condition(
        instances,
        snaps,
        pos_snap_component,
        neg_snap_component,
        mask_component=mask_component,
    )
    start_camera_position = (
        tuple(state[viewpoint_component]['position']) + (0,))
    end_camera_position, visible_snaps = search_camera_space(
        env, viewpoint_component, state, condition, float('inf'))
    if end_camera_position is None or visible_snaps is None:
        return None, None, None
    
    # convert the camera motion to camera actions
    camera_actions = compute_camera_actions(
        env, viewpoint_component, start_camera_position, end_camera_position)
    
    return camera_actions, end_camera_position, visible_snaps

def compute_camera_actions(
    env,
    viewpoint_component,
    start_position,
    end_position,
):
    actions = []
    
    a = env.components[viewpoint_component].azimuth_steps
    azimuth_a = end_position[0] + a - start_position[0]
    azimuth_b = end_position[0] - start_position[0]
    azimuth_c = end_position[0] - a - start_position[0]
    if abs(azimuth_a) < abs(azimuth_b) and abs(azimuth_a) < abs(azimuth_c):
        azimuth_steps = azimuth_a
    elif abs(azimuth_b) < abs(azimuth_c):
        azimuth_steps = azimuth_b
    else:
        azimuth_steps = azimuth_c
    
    other_steps = numpy.array(end_position)[1:] - start_position[1:]
    steps = numpy.concatenate([[azimuth_steps], other_steps])
    
    frame_scene = steps[3]
    if frame_scene:
        action = env.no_op_action()
        #action[viewpoint_component]['frame'] = True
        action[viewpoint_component] = 7
        actions.append(action)
    
    for directions, offset in zip(((1,2),(3,4),(5,6)), steps[:3]):
        repeat = abs(offset)
        if offset < 0:
            direction = directions[0]
        else:
            direction = directions[1]
        for step in range(repeat):
            action = env.no_op_action()
            #action[viewpoint_component]['direction'] = direction
            action[viewpoint_component] = direction
            actions.append(action)
    
    return actions

def get_new_to_wip_connections(
    goal_assembly,
    instance,
    goal_to_wip,
    debug = False,
):
    # get connections
    connections = goal_assembly['edges'][0] == instance
    
    # fitler for what actually exists in the scene
    extant_connections = numpy.array([
        goal_assembly['edges'][1,i] in goal_to_wip
        for i in range(goal_assembly['edges'].shape[1])])
    connections = connections & extant_connections
    
    # get the connected instances
    connected_instances = goal_assembly['edges'][1][connections]
    wip_instances = [goal_to_wip[i] for i in connected_instances]
    
    # get the connected snaps
    instance_snaps = goal_assembly['edges'][2][connections]
    wip_snaps = goal_assembly['edges'][3][connections]
    
    # make the new_to_wip and wip_to_new lookups
    new_to_wip = {
        (1,iss):(wii,wss) for wii, iss, wss in
        zip(wip_instances, instance_snaps, wip_snaps)
    }
    wip_to_new = {v:k for k,v in new_to_wip.items()}
    
    if debug:
        import pdb
        pdb.set_trace()
    
    return new_to_wip, wip_to_new, wip_instances, wip_snaps

def get_wip_to_wip_connections(
    wip_assembly,
    instance,
):
    connections = wip_assembly['edges'][0] == instance
    connected_instances = wip_assembly['edges'][1][connections]
    instance_snaps = wip_assembly['edges'][2][connections]
    connected_snaps = wip_assembly['edges'][3][connections]
    instance_to_wip = {
        (instance, iss):(wii,wss) for wii, iss, wss in
        zip(connected_instances, instance_snaps, connected_snaps)
    }
    wip_to_instance = {v:k for k,v in instance_to_wip.items()}
    
    return (
        instance_to_wip,
        wip_to_instance,
        connected_instances,
        connected_snaps,
        instance_snaps)

def compute_discrete_rotation(
    goal_assembly,
    wip_assembly,
    snap,
    goal_instance,
    goal_connected_instance,
    wip_instance,
    wip_connected_instance,
    brick_shape_name,
    rotation_steps = 4,
    allow_snap_flip = False,
):
    
    goal_transform = goal_assembly['pose'][goal_instance]
    connected_goal_transform = goal_assembly['pose'][goal_connected_instance]
    inv_connected_goal_transform = numpy.linalg.inv(connected_goal_transform)
    goal_offset = (
        inv_connected_goal_transform @
        goal_transform
    )
    
    wip_transform = wip_assembly['pose'][wip_instance]
    connected_wip_transform = wip_assembly['pose'][wip_connected_instance]
    inv_connected_wip_transform = numpy.linalg.inv(connected_wip_transform)
    wip_offset = (
        numpy.linalg.inv(connected_wip_transform) @
        wip_transform
    )
    
    brick_shape = BrickShape(brick_shape_name)
    snap_transform = brick_shape.snaps[snap].transform
    inv_snap_transform = numpy.linalg.inv(snap_transform)
    wip_snap_transform = wip_transform @ snap_transform
    
    goal_r = unscale_transform(goal_offset)
    
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
            inv_connected_wip_transform @
            wip_snap_transform @
            ry @
            inv_snap_transform
        )
        offset_r = unscale_transform(offset)
        t = numpy.trace(offset_r[:3,:3].T @ goal_r[:3,:3])
        offsets.append((t,r,offset))
    
    snap_style = brick_shape.snaps[snap]
    if allow_snap_flip and isinstance(snap_style, SnapFinger):
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
                inv_connected_wip_transform @
                wip_snap_transform @
                ry @
                flip_rotation @
                inv_snap_transform
            )
            offset_r = unscale_transform(offset)
            t = numpy.trace(offset_r[:3,:3].T @ goal_r[:3,:3])
            offsets.append((t,r+rotation_steps,offset))
    
    return max(offsets)[1]

# action builders ==============================================================

def make_insert_action(env, brick_shape, brick_color):
    insert_action = env.no_op_action()
    insert_action['insert_brick'] = {
        'shape' : brick_shape,
        'color' : brick_color,
    }
    
    return insert_action


# planners =====================================================================

def plan_add_first_brick(
    env,
    goal_assembly,
    instance,
    observation,
    goal_to_wip,
    shape_id_to_brick_shape,
    split_cursor_actions=False,
    debug=False
):
    
    if debug:
        vis_obs(observation, 1, 0)
    
    # initialization -----------------------------------------------------------
    # initialize the action squence
    observation_seq = [observation]
    action_seq = []
    reward_seq = []
    
    # pull shape, color and transform information out of the goal assembly
    shape_index = goal_assembly['shape'][instance]
    color_index = goal_assembly['color'][instance]
    brick_transform = goal_assembly['pose'][instance]
    
    # find the upright snaps ---------------------------------------------------
    brick_shape = BrickShape(shape_id_to_brick_shape[shape_index])
    brick_instance = BrickInstance(0, brick_shape, color_index, brick_transform)
    upright_snaps = brick_instance.get_upright_snaps()
    
    # if there are no upright snaps this brick cannot be added as a first brick
    if not len(upright_snaps):
        return None

    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(env, shape_index, color_index)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    observation_seq.append(observation)
    reward_seq.append(reward)
    
    if debug:
        vis_obs(observation, 1, 1)
    
    # hand camera --------------------------------------------------------------
    # compute the hand camera motion
    state = env.get_state()
    (hand_camera_actions,
     hand_camera_position,
     new_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        #numpy.ones(len(upright_snap_ids), dtype=numpy.long),
        numpy.ones(len(upright_snaps), dtype=numpy.long),
        #upright_snap_ids,
        [int(snap.snap_style) for snap in upright_snaps],
        'hand_pos_snap_render',
        'hand_neg_snap_render',
        'hand_viewpoint',
    )
    if new_visible_snaps is None:
        return None
    
    # update the state
    if hand_camera_actions:
        action_seq.extend(hand_camera_actions)
        #_ = env.set_state(state) # no longer necessary
        for action in hand_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
            reward_seq.append(reward)
        #replace_camera_in_state(
        #    env, state, 'hand_viewpoint', hand_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, 1, 2)
    
    # make the pick and place action -------------------------------------------
    nr = random.randint(0, new_visible_snaps.shape[1]-1)
    y, x, p, i, s = new_visible_snaps[:,nr]
    
    if split_cursor_actions:
        # hand cursor
        hand_cursor_action = env.no_op_action()
        hand_cursor_action['hand_cursor'] = {
            'activate':True,
            'position':numpy.array([y,x]),
            'polarity':p,
        }
        
        action_seq.append(hand_cursor_action)
        observation, reward, terminal, info = env.step(hand_cursor_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        # pick and place
        pick_and_place_action = env.no_op_action()
        pick_and_place_action['pick_and_place'] = 2
        
        action_seq.append(pick_and_place_action)
        observation, reward, terminal, info = env.step(pick_and_place_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
    
    else:
        pick_and_place_action = env.no_op_action()
        pick_and_place_action['hand_cursor'] = {
            'activate':True,
            'position':numpy.array([y,x]),
            'polarity':p,
        }
        pick_and_place_action['pick_and_place'] = 2
        action_seq.append(pick_and_place_action)
        observation, reward, terminal, info = env.step(pick_and_place_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
    
    if debug:
        vis_obs(observation, 1, 3)
    
    if debug:
        env.components['table_scene'].brick_scene.export_ldraw(
            './add_1.mpd')
    
    return observation_seq, action_seq, reward_seq

def plan_add_nth_brick(
    env,
    goal_assembly,
    instance,
    observation,
    goal_to_wip,
    shape_id_to_brick_shape,
    split_cursor_actions=False,
    allow_snap_flip=False,
    debug=False,
):
    
    # initialization -----------------------------------------------------------
    wip_to_goal = {v:k for k,v in goal_to_wip.items()}
    new_wip_instance = max(goal_to_wip.values())+1
    
    # initialize the action sequence
    observation_seq = [observation]
    action_seq = []
    reward_seq = []
    
    # pull shape, color and transform information out of the goal assembly
    brick_shape = goal_assembly['shape'][instance]
    brick_color = goal_assembly['color'][instance]
    brick_transform = goal_assembly['pose'][instance]
    
    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(env, brick_shape, brick_color)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    observation_seq.append(observation)
    reward_seq.append(reward)
    
    if debug:
        vis_obs(observation, new_wip_instance, 0)
    
    # table camera -------------------------------------------------------------
    # find all connections between the new instance and the wip bricks
    (new_to_wip,
     wip_to_new,
     wip_instances,
     wip_snaps) = get_new_to_wip_connections(
        goal_assembly, instance, goal_to_wip)
    
    # compute the table camera motion
    state = env.get_state()
    (table_camera_actions,
     table_camera_position,
     wip_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        wip_instances,
        wip_snaps,
        'table_pos_snap_render',
        'table_neg_snap_render',
        'table_viewpoint',
        'table_mask_render',
    )
    if wip_visible_snaps is None:
        return None, None
    
    # update the state
    if table_camera_actions:
        action_seq.extend(table_camera_actions)
        #_ = env.set_state(state) # no longer necessary
        for action in table_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
            reward_seq.append(reward)
        #replace_camera_in_state(
        #    env, state, 'table_viewpoint', table_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, new_wip_instance, 1)
    
    # hand camera --------------------------------------------------------------
    # find the new snaps that are connected to the wip visible snaps
    wy, wx, wp, wi, ws = wip_visible_snaps
    new_instances = numpy.ones(ws.shape[0], dtype=numpy.long)
    try:
        new_snaps = [wip_to_new[i,s][1] for i,s in zip(wi, ws)]
    except:
        # this is caused by two snaps being connected to one
        raise DoubleSnapConnectionFailure
    
    # compute the hand camera motion
    state = env.get_state()
    (hand_camera_actions,
     hand_camera_position,
     new_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        new_instances,
        new_snaps,
        'hand_pos_snap_render',
        'hand_neg_snap_render',
        'hand_viewpoint',
    )
    if new_visible_snaps is None:
        return None, None
    
    # update the state
    if hand_camera_actions:
        action_seq.extend(hand_camera_actions)
        #_ = env.set_state(state) # no longer necessary
        for action in hand_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
            reward_seq.append(reward)
        #replace_camera_in_state(
        #    env, state, 'hand_viewpoint', hand_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, new_wip_instance, 2)
    
    # make the pick and place action -------------------------------------------
    # pick one of the new visible snaps randomly
    nr = random.randint(0, new_visible_snaps.shape[1]-1)
    nyy, nxx, npp, nii, nss = new_visible_snaps[:,nr]
    
    # find the table visible snap that is connected to the chosen new snap
    wip_i, wip_s = new_to_wip[nii, nss]
    table_locations = numpy.where(
        (wi == wip_i) & (ws == wip_s))[0]
    wr = random.choice(table_locations)
    wyy, wxx, wpp, wii, wss = wip_visible_snaps[:,wr]

    # make the pick and place action
    if split_cursor_actions:
        # hand cursor
        hand_cursor_action = env.no_op_action()
        hand_cursor_action['hand_cursor'] = {
            'activate':True,
            'position':numpy.array([nyy, nxx]),
            'polarity':npp,
        }
        
        action_seq.append(hand_cursor_action)
        observation, reward, terminal, info = env.step(hand_cursor_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        # table cursor
        table_cursor_action = env.no_op_action()
        table_cursor_action['table_cursor'] = {
            'activate':True,
            'position':numpy.array([wyy, wxx]),
            'polarity':wpp,
        }
        
        action_seq.append(table_cursor_action)
        observation, reward, terminal, info = env.step(table_cursor_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        # pick and place
        pick_and_place_action = env.no_op_action()
        pick_and_place_action['pick_and_place'] = 1
        
        action_seq.append(pick_and_place_action)
        observation, reward, terminal, info = env.step(pick_and_place_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        if not observation['pick_and_place']['success']:
            raise PickAndPlaceFailure
    
    else:
        pick_and_place_action = env.no_op_action()
        pick_and_place_action['table_cursor'] = {
            'activate':True,
            'position':numpy.array([wyy, wxx]),
            'polarity':wpp,
        }
        pick_and_place_action['hand_cursor'] = {
            'activate':True,
            'position':numpy.array([nyy, nxx]),
            'polarity':npp,
        }
        #pick_and_place_action['pick_and_place'] = {
        #    'activate':True,
        #    'place_at_origin':False,
        #}
        pick_and_place_action['pick_and_place'] = 1
        action_seq.append(pick_and_place_action)
        
        # take the action to generate the latest observation
        observation, reward, terminal, info = env.step(pick_and_place_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        if not observation['pick_and_place']['success']:
            raise PickAndPlaceFailure
    
    if debug:
        vis_obs(observation, new_wip_instance, 3)
    
    # make the rotation action -------------------------------------------------
    
    # figure out if we need to rotate the new instance
    discrete_rotation = compute_discrete_rotation(
        goal_assembly,
        observation['table_assembly'],
        nss,
        instance,
        wip_to_goal[wii],
        new_wip_instance,
        wii,
        shape_id_to_brick_shape[brick_shape],
        rotation_steps = 4,
        allow_snap_flip = allow_snap_flip,
    )
    
    if discrete_rotation:
        # compute the camera motion to make sure the snap to rotate is visible
        state = env.get_state()
        (table_camera_actions,
         table_camera_position,
         new_visible_snaps) = plan_camera_to_see_snaps(
            env,
            state,
            [new_wip_instance],
            [nss],
            'table_pos_snap_render',
            'table_neg_snap_render',
            'table_viewpoint',
            'table_mask_render',
        )
        if new_visible_snaps is None:
            return None, None
        
        # update the state
        if table_camera_actions:
            action_seq.extend(table_camera_actions)
            #_ = env.set_state(state) # no longer necessary
            for action in table_camera_actions:
                observation, reward, terminal, info = env.step(action)
                observation_seq.append(observation)
                reward_seq.append(reward)
            #replace_camera_in_state(
            #    env, state, 'table_viewpoint', table_camera_position)
            #observation = env.set_state(state)
            
            if debug:
                vis_obs(observation, new_wip_instance, 4)
        
        r = random.randint(0, new_visible_snaps.shape[1]-1)
        y, x, p, i, s = new_visible_snaps[:,r]
        if split_cursor_actions:
            # cursor
            table_cursor_action = env.no_op_action()
            table_cursor_action['table_cursor'] = {
                'activate' : True,
                'position' : numpy.array([y,x]),
                'polarity' : p,
            }
            
            action_seq.append(table_cursor_action)
            observation, reward, terminal, info = env.step(table_cursor_action)
            observation_seq.append(observation)
            reward_seq.append(reward)
            
            # rotation
            rotation_action = env.no_op_action()
            rotation_action['rotate'] = discrete_rotation
            
            action_seq.append(rotation_action)
            observation, reward, terminal, info = env.step(rotation_action)
            observation_seq.append(observation)
            reward_seq.append(reward)
           
        else:
            # cursor/rotate action
            rotation_action = env.no_op_action()
            rotation_action['table_cursor'] = {
                'activate' : True,
                'position' : numpy.array([y,x]),
                'polarity' : p,
            }
            rotation_action['rotate'] = discrete_rotation
            
            action_seq.append(rotation_action)
            observation, reward, terminal, info = env.step(rotation_action)
            observation_seq.append(observation)
            reward_seq.append(reward)
        
        if debug:
            vis_obs(observation, new_wip_instance, 5)
    
    if debug:
        env.components['table_scene'].brick_scene.export_ldraw(
            './add_%i.mpd'%new_wip_instance)
    
    #env.components['table_scene'].brick_scene.export_ldraw(
    #    './export_step_%i_b.mpd'%instance)
    
    return observation_seq, action_seq, reward_seq

def plan_remove_nth_brick(
    env,
    goal_assembly,
    instance,
    observation,
    false_positive_to_wip,
    shape_id_to_brick_shape,
    use_mask=True,
    split_cursor_actions=False,
    debug=False,
):
    
    # intialization ------------------------------------------------------------
    wip_assembly = observation['table_assembly']
    wip_instance = false_positive_to_wip[instance]
    
    if debug:
        debug_index = (
            wip_assembly['shape'].shape[0] -
            numpy.sum(wip_assembly['shape'] != 0))
    
    # initialize the action sequence
    observation_seq = [observation]
    action_seq = []
    reward_seq = []
    
    # table camera -------------------------------------------------------------
    # find all connections between the instance and the rest of the scene
    # if there are any connections, we need to remove the brick using the
    # connected snaps, otherwise we can use any snaps
    # (Note to future self: this seems a little strange, this is used as a
    # second check after we have used a collision map to figure out a good
    # ordering for removing bricks.  That collision map reasons in terms of
    # groups of snaps with the same polarity that point in the same direction,
    # but here, we just find all the connected snaps, and search for the first
    # one we can find.  This seemed wrong because maybe there are more than
    # one group of connected snaps, and maybe we can remove along one direction
    # but not another.  This objection kind of goes away if we only allow
    # bricks to be removed if they are only connected via one "group" of snaps,
    # which is maybe not strictly enforced (but could be) but is loosely
    # enforced via the collision checking.  The second objection is that this
    # is only looking at connected snaps, but maybe there are visible snaps
    # FROM THE SAME GROUP that we don't need to move the camera to see and that
    # we could use to disassemble the object.  But this also only buys
    # us anything for disassembly, not reassembly, since we have to find
    # connected snaps for reassembly.
    
    (instance_to_wip,
     wip_to_instance,
     wip_instances,
     wip_snaps,
     instance_snaps) = get_wip_to_wip_connections(wip_assembly, wip_instance)
    
    if not len(wip_instances):
        brick_shape = wip_assembly['shape'][wip_instance]
        brick_shape = BrickShape(shape_id_to_brick_shape[brick_shape])
        instance_snaps = numpy.array(range(len(brick_shape.snaps)))
    
    # compute the table camera motion
    state = env.get_state()
    (table_camera_actions,
     table_camera_position,
     visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        numpy.ones(instance_snaps.shape, dtype=numpy.long) * instance,
        instance_snaps,
        'table_pos_snap_render',
        'table_neg_snap_render',
        'table_viewpoint',
        'table_mask_render' if use_mask else None,
    )
    if visible_snaps is None:
        raise NoVisibleSnapsFailure
    
    # update the state
    if table_camera_actions:
        action_seq.extend(table_camera_actions)
        #_ = env.set_state(state) # no longer necessary
        for action in table_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
            reward_seq.append(reward)
        #replace_camera_in_state(
        #    env, state, 'table_viewpoint', table_camera_position)
        #observation = env.set_state(state)
        #env.components['table_scene'].brick_scene.export_ldraw(
        #    './fail_%i.mpd'%instance)
        if debug:
            vis_obs(observation, debug_index, 1, label='disassemble')
    
    # make the removal action --------------------------------------------------
    # pick one of the visible snaps randomly
    wr = random.randint(0, visible_snaps.shape[1]-1)
    wyy, wxx, wpp, wii, wss = visible_snaps[:,wr]
    
    # make the pick and place action
    if split_cursor_actions:
        # cursor action
        table_cursor_action = env.no_op_action()
        table_cursor_action['table_cursor'] = {
            'activate':True,
            'position':numpy.array([wyy, wxx]),
            'polarity':wpp,
        }
        
        action_seq.append(table_cursor_action)
        observation, reward, terminal, info = env.step(table_cursor_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        # disassembly action
        disassembly_action = env.no_op_action()
        disassembly_action['disassembly'] = 1
        
        action_seq.append(disassembly_action)
        observation, reward, terminal, info = env.step(disassembly_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        if not observation['disassembly']['success']:
            raise DisassemblyFailure
    
    else:
        disassembly_action = env.no_op_action()
        disassembly_action['table_cursor'] = {
            'activate':True,
            'position':numpy.array([wyy, wxx]),
            'polarity':wpp,
        }
        #disassembly_action['disassembly'] = {
        #    'activate':True,
        #}
        disassembly_action['disassembly'] = 1
        action_seq.append(disassembly_action)
        observation, reward, terminal, info = env.step(disassembly_action)
        observation_seq.append(observation)
        reward_seq.append(reward)
        
        if not observation['disassembly']['success']:
            raise DisassemblyFailure
    
    if debug:
        vis_obs(observation, debug_index, 2, label='disassemble')
        env.components['table_scene'].brick_scene.export_ldraw(
            './disassembly_%i.mpd'%instance)
    
    return observation_seq, action_seq, reward_seq

def vis_obs(obs, i, j, label='tmp'):
    from splendor.image import save_image
    from ltron.visualization.drawing import (
        map_overlay, stack_images_horizontal)
    table_opacity = numpy.zeros((64,64,1))
    table_cursor = numpy.zeros((64,64,3), dtype=numpy.uint8)
    y, x = obs['table_cursor']['position']
    p = obs['table_cursor']['polarity']
    k = obs['table_cursor']['instance_id']
    s = obs['table_cursor']['snap_id']
    table_opacity[y,x] = 1.
    if p == 0:
        table_cursor[y,x] = [255,0,0]
    else:
        table_cursor[y,x] = [0,0,255]
    table_image = map_overlay(
        obs['table_color_render'],
        table_cursor,
        table_opacity,
    )

    hand_opacity = numpy.zeros((24,24,1))
    hand_cursor = numpy.zeros((24,24,3), dtype=numpy.uint8)
    y, x = obs['hand_cursor']['position']
    p = obs['hand_cursor']['polarity']
    k = obs['hand_cursor']['instance_id']
    s = obs['hand_cursor']['snap_id']
    hand_opacity[y,x] = 1.
    if p == 0:
        hand_cursor[y,x] = [255,0,0]
    else:
        hand_cursor[y,x] = [0,0,255]
    hand_image = map_overlay(
        obs['hand_color_render'],
        hand_cursor,
        hand_opacity,
    )

    image = stack_images_horizontal(
        (table_image, hand_image), align='bottom')
    path = './%s_%04i_%04i.png'%(label, i, j)
    save_image(image, path)
