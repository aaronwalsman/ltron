import random
import math
import copy
from bisect import insort

import numpy

from splendor.image import save_image

from ltron.bricks.brick_type import BrickType
from ltron.bricks.brick_instance import BrickInstance
from ltron.gym.envs.reassembly_env import reassembly_template_action
from ltron.exceptions import LtronException

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
    
    return None, None

def test_camera_position(env, position, component_name, state, condition):
    new_state = copy.deepcopy(state)
    replace_camera_in_state(env, new_state, component_name, position)
    observation = env.set_state(new_state)
    #save_image(observation['workspace_color_render'],
    #    './cam_%i_%i_%i_%i.png'%tuple(position))
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
    
    # convert the handspace camera motion to camera actions
    camera_actions = compute_camera_actions(
        env, viewpoint_component, start_camera_position, end_camera_position)
    
    return camera_actions, end_camera_position, visible_snaps

def compute_camera_actions(
    env,
    viewpoint_component,
    start_position,
    end_position
):
    actions = []
    steps = numpy.array(end_position) - start_position
    frame_scene = steps[3]
    if frame_scene:
        action = reassembly_template_action()
        action[viewpoint_component]['frame'] = True
        actions.append(action)
    
    for directions, offset in zip(((1,2),(3,4),(5,6)), steps[:3]):
        steps = abs(offset)
        if offset < 0:
            direction = directions[0]
        else:
            direction = directions[1]
        for step in range(steps):
            action = reassembly_template_action()
            action[viewpoint_component]['direction'] = direction
            actions.append(action)
    
    return actions

def get_new_to_wip_connections(
    goal_config,
    instance,
    goal_to_wip,
):
    # get connections
    connections = goal_config['edges'][0] == instance
    
    # fitler for what actually exists in the scene
    extant_connections = numpy.array([
        goal_config['edges'][1,i] in goal_to_wip
        for i in range(goal_config['edges'].shape[1])])
    connections = connections & extant_connections
    
    # get the connected instances
    connected_instances = goal_config['edges'][1][connections]
    wip_instances = [goal_to_wip[i] for i in connected_instances]
    
    # get the connected snaps
    instance_snaps = goal_config['edges'][2][connections]
    wip_snaps = goal_config['edges'][3][connections]
    
    # make the new_to_wip and wip_to_new lookups
    new_to_wip = {
        (1,iss):(wii,wss) for wii, iss, wss in
        zip(wip_instances, instance_snaps, wip_snaps)
    }
    wip_to_new = {v:k for k,v in new_to_wip.items()}
    
    return new_to_wip, wip_to_new, wip_instances, wip_snaps

def get_wip_to_wip_connections(
    wip_config,
    instance,
):
    connections = wip_config['edges'][0] == instance
    connected_instances = wip_config['edges'][1][connections]
    instance_snaps = wip_config['edges'][2][connections]
    connected_snaps = wip_config['edges'][3][connections]
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
    goal_config,
    wip_config,
    snap,
    goal_instance,
    goal_connected_instance,
    wip_instance,
    wip_connected_instance,
    brick_type_name,
    rotation_steps = 4,
):
    
    goal_transform = goal_config['pose'][goal_instance]
    connected_goal_transform = goal_config['pose'][goal_connected_instance]
    inv_connected_goal_transform = numpy.linalg.inv(connected_goal_transform)
    goal_offset = (
        inv_connected_goal_transform @
        goal_transform
    )
    
    wip_transform = wip_config['pose'][wip_instance]
    connected_wip_transform = wip_config['pose'][wip_connected_instance]
    inv_connected_wip_transform = numpy.linalg.inv(connected_wip_transform)
    wip_offset = (
        numpy.linalg.inv(connected_wip_transform) @
        wip_transform
    )
    
    brick_type = BrickType(brick_type_name)
    snap_transform = brick_type.snaps[snap].transform
    inv_snap_transform = numpy.linalg.inv(snap_transform)
    wip_snap_transform = wip_transform @ snap_transform
    
    if numpy.allclose(goal_offset, wip_offset):
        return 0
    
    offsets = []
    for r in range(1, rotation_steps):
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
        offsets.append(offset)
        if numpy.allclose(offset, goal_offset):
            return r
    
    raise NoMatchingRotationError

# action builders ==============================================================

def make_insert_action(brick_class, brick_color):
    insert_action = reassembly_template_action()
    insert_action['insert_brick'] = {
        'class_id' : brick_class,
        'color_id' : brick_color,
    }
    
    return insert_action


# planners =====================================================================

def plan_add_first_brick(
    env,
    goal_config,
    instance,
    observation,
    goal_to_wip,
    class_to_brick_type,
    debug=False
):
    
    if debug:
        vis_obs(observation, 1, 0)
    
    # initialization -----------------------------------------------------------
    # initialize the action squence
    observation_seq = [observation]
    action_seq = []
    
    # pull class, color and transform information out of the goal configuration
    brick_class = goal_config['class'][instance]
    brick_color = goal_config['color'][instance]
    brick_transform = goal_config['pose'][instance]
    
    # find the upright snaps ---------------------------------------------------
    brick_type = BrickType(class_to_brick_type[brick_class])
    brick_instance = BrickInstance(0, brick_type, brick_color, brick_transform)
    upright_snaps, upright_snap_ids = brick_instance.get_upright_snaps()
    
    # if there are no upright snaps this brick cannot be added as a first brick
    if not len(upright_snaps):
        return None

    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(brick_class, brick_color)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    observation_seq.append(observation)
    
    if debug:
        vis_obs(observation, 1, 1)
    
    # handspace camera ---------------------------------------------------------
    # compute the handspace camera motion
    state = env.get_state()
    (handspace_camera_actions,
     handspace_camera_position,
     new_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        numpy.ones(len(upright_snap_ids), dtype=numpy.long),
        upright_snap_ids,
        'handspace_pos_snap_render',
        'handspace_neg_snap_render',
        'handspace_viewpoint',
    )
    if new_visible_snaps is None:
        return None
    
    # update the state
    if handspace_camera_actions:
        action_seq.extend(handspace_camera_actions)
        _ = env.set_state(state)
        for action in handspace_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
        #replace_camera_in_state(
        #    env, state, 'handspace_viewpoint', handspace_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, 1, 2)
    
    # make the pick and place action -------------------------------------------
    nr = random.randint(0, new_visible_snaps.shape[1]-1)
    y, x, p, i, s = new_visible_snaps[:,nr]
    
    pick_and_place_action = reassembly_template_action()
    pick_and_place_action['handspace_cursor'] = {
        'activate':True,
        'position':numpy.array([y,x]),
        'polarity':p,
    }
    pick_and_place_action['pick_and_place'] = {
        'activate':True,
        'place_at_origin':True,
    }
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(pick_and_place_action)
    observation_seq.append(observation)
    
    if debug:
        vis_obs(observation, 1, 3)
    
    if debug:
        env.components['workspace_scene'].brick_scene.export_ldraw(
            './add_1.mpd')
    
    return observation_seq, action_seq

def plan_add_nth_brick(
    env,
    goal_config,
    instance,
    observation,
    goal_to_wip,
    class_to_brick_type,
    debug=False,
):
    
    # initialization -----------------------------------------------------------
    wip_to_goal = {v:k for k,v in goal_to_wip.items()}
    new_wip_instance = max(goal_to_wip.values())+1
    
    # initialize the action sequence
    observation_seq = [observation]
    action_seq = []
    
    # pull class, color and transform information out of the goal configuration
    brick_class = goal_config['class'][instance]
    brick_color = goal_config['color'][instance]
    brick_transform = goal_config['pose'][instance]
    
    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(brick_class, brick_color)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    observation_seq.append(observation)
    
    if debug:
        vis_obs(observation, new_wip_instance, 0)
    
    # workspace camera ---------------------------------------------------------
    # find all connections between the new instance and the wip bricks
    (new_to_wip,
     wip_to_new,
     wip_instances,
     wip_snaps) = get_new_to_wip_connections(goal_config, instance, goal_to_wip)
    
    # compute the workspace camera motion
    state = env.get_state()
    (workspace_camera_actions,
     workspace_camera_position,
     wip_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        wip_instances,
        wip_snaps,
        'workspace_pos_snap_render',
        'workspace_neg_snap_render',
        'workspace_viewpoint',
        'workspace_mask_render',
    )
    if wip_visible_snaps is None:
        return None
    
    # update the state
    if workspace_camera_actions:
        action_seq.extend(workspace_camera_actions)
        _ = env.set_state(state)
        for action in workspace_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
        replace_camera_in_state(
            env, state, 'workspace_viewpoint', workspace_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, new_wip_instance, 1)
    
    # handspace camera ---------------------------------------------------------
    # find the new snaps that are connected to the wip visible snaps
    wy, wx, wp, wi, ws = wip_visible_snaps
    new_instances = numpy.ones(ws.shape[0], dtype=numpy.long)
    try:
        new_snaps = [wip_to_new[i,s][1] for i,s in zip(wi, ws)]
    except:
        import pdb
        pdb.set_trace()
    
    # compute the handspace camera motion
    (handspace_camera_actions,
     handspace_camera_position,
     new_visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        new_instances,
        new_snaps,
        'handspace_pos_snap_render',
        'handspace_neg_snap_render',
        'handspace_viewpoint',
    )
    if new_visible_snaps is None:
        return None
    
    # update the state
    if handspace_camera_actions:
        action_seq.extend(handspace_camera_actions)
        _ = env.set_state(state)
        for action in handspace_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
        #replace_camera_in_state(
        #    env, state, 'handspace_viewpoint', handspace_camera_position)
        #observation = env.set_state(state)
        
        if debug:
            vis_obs(observation, new_wip_instance, 2)
    
    # make the pick and place action -------------------------------------------
    # pick one of the new visible snaps randomly
    nr = random.randint(0, new_visible_snaps.shape[1]-1)
    nyy, nxx, npp, nii, nss = new_visible_snaps[:,nr]
    
    # find the workspace visible snap that is connected to the chosen new snap
    wip_i, wip_s = new_to_wip[nii, nss]
    workspace_locations = numpy.where(
        (wi == wip_i) & (ws == wip_s))[0]
    wr = random.choice(workspace_locations)
    wyy, wxx, wpp, wii, wss = wip_visible_snaps[:,wr]

    # make the pick and place action
    pick_and_place_action = reassembly_template_action()
    pick_and_place_action['workspace_cursor'] = {
        'activate':True,
        'position':numpy.array([wyy, wxx]),
        'polarity':wpp,
    }
    pick_and_place_action['handspace_cursor'] = {
        'activate':True,
        'position':numpy.array([nyy, nxx]),
        'polarity':npp,
    }
    pick_and_place_action['pick_and_place'] = {
        'activate':True,
        'place_at_origin':False,
    }
    action_seq.append(pick_and_place_action)
    
    # take the action to generate the latest observation
    observation, reward, terminal, info = env.step(pick_and_place_action)
    observation_seq.append(observation)
    state = env.get_state()
    
    if not observation['pick_and_place']['success']:
        raise PickAndPlaceFailure
    
    if debug:
        vis_obs(observation, new_wip_instance, 3)

    # make the rotation action -------------------------------------------------
    
    # figure out if we need to rotate the new instance
    discrete_rotation = compute_discrete_rotation(
        goal_config,
        observation['workspace_config']['config'],
        nss,
        instance,
        wip_to_goal[wii],
        new_wip_instance,
        wii,
        class_to_brick_type[brick_class],
        rotation_steps = 4,
    )
    
    if discrete_rotation:
        # compute the camera motion to make sure the snap to rotate is visible
        (workspace_camera_actions,
         workspace_camera_position,
         new_visible_snaps) = plan_camera_to_see_snaps(
            env,
            state,
            [new_wip_instance],
            [nss],
            'workspace_pos_snap_render',
            'workspace_neg_snap_render',
            'workspace_viewpoint',
            'workspace_mask_render',
        )
        if new_visible_snaps is None:
            return None
        
        # update the state
        if workspace_camera_actions:
            action_seq.extend(workspace_camera_actions)
            _ = env.set_state(state)
            for action in workspace_camera_actions:
                observation, reward, terminal, info = env.step(action)
                observation_seq.append(observation)
            #replace_camera_in_state(
            #    env, state, 'workspace_viewpoint', workspace_camera_position)
            #observation = env.set_state(state)
            
            if debug:
                vis_obs(observation, new_wip_instance, 4)
        
        rotation_action = reassembly_template_action()
        r = random.randint(0, new_visible_snaps.shape[1]-1)
        y, x, p, i, s = new_visible_snaps[:,r]
        rotation_action['workspace_cursor'] = {
            'activate' : True,
            'position' : numpy.array([y,x]),
            'polarity' : p
        }
        rotation_action['rotate'] = discrete_rotation
        action_seq.append(rotation_action)
        observation, reward, terminal, info = env.step(rotation_action)
        observation_seq.append(observation)
        
        if debug:
            vis_obs(observation, new_wip_instance, 5)
    
    if debug:
        env.components['workspace_scene'].brick_scene.export_ldraw(
            './add_%i.mpd'%new_wip_instance)
    
    #env.components['workspace_scene'].brick_scene.export_ldraw(
    #    './export_step_%i_b.mpd'%instance)
    
    return observation_seq, action_seq

def plan_remove_nth_brick(
    env,
    goal_config,
    instance,
    observation,
    false_positive_to_wip,
    class_to_brick_type,
    debug=False
):
    
    # intialization ------------------------------------------------------------
    wip_config = observation['workspace_config']['config']
    wip_instance = false_positive_to_wip[instance]
    
    if debug:
        debug_index = (
            wip_config['class'].shape[0] - numpy.sum(wip_config['class'] != 0))
    
    # initialize the action sequence
    observation_seq = [observation]
    action_seq = []
    
    # workspace camera ---------------------------------------------------------
    # find all connections between the instance and the rest of the scene
    # if there are any connections, we need to remove the brick using the
    # connected snaps, otherwise we can use any snaps
    (instance_to_wip,
     wip_to_instance,
     wip_instances,
     wip_snaps,
     instance_snaps) = get_wip_to_wip_connections(wip_config, wip_instance)
    
    if not len(wip_instances):
        brick_class = wip_config['class'][wip_instance]
        brick_type = BrickType(class_to_brick_type[brick_class])
        instance_snaps = numpy.array(range(len(brick_type.snaps)))
    
    # compute the workspace camera motion
    state = env.get_state()
    (workspace_camera_actions,
     workspace_camera_position,
     visible_snaps) = plan_camera_to_see_snaps(
        env,
        state,
        numpy.ones(instance_snaps.shape, dtype=numpy.long) * instance,
        instance_snaps,
        'workspace_pos_snap_render',
        'workspace_neg_snap_render',
        'workspace_viewpoint',
        'workspace_mask_render',
    )
    if visible_snaps is None:
        raise NoVisibleSnapsFailure
    
    # update the state
    if workspace_camera_actions:
        action_seq.extend(workspace_camera_actions)
        _ = env.set_state(state)
        for action in workspace_camera_actions:
            observation, reward, terminal, info = env.step(action)
            observation_seq.append(observation)
        #replace_camera_in_state(
        #    env, state, 'workspace_viewpoint', workspace_camera_position)
        #observation = env.set_state(state)
        #env.components['workspace_scene'].brick_scene.export_ldraw(
        #    './fail_%i.mpd'%instance)
        if debug:
            vis_obs(observation, debug_index, 1, label='disassemble')
    
    # make the removal action --------------------------------------------------
    # pick one of the visible snaps randomly
    wr = random.randint(0, visible_snaps.shape[1]-1)
    wyy, wxx, wpp, wii, wss = visible_snaps[:,wr]
    
    # make the pick and place action
    disassembly_action = reassembly_template_action()
    disassembly_action['workspace_cursor'] = {
        'activate':True,
        'position':numpy.array([wyy, wxx]),
        'polarity':wpp,
    }
    disassembly_action['disassembly'] = {
        'activate':True,
    }
    action_seq.append(disassembly_action)
    observation, reward, terminal, info = env.step(disassembly_action)
    observation_seq.append(observation)
    
    if not observation['disassembly']['success']:
        raise DisassemblyFailure
    
    if debug:
        vis_obs(observation, debug_index, 2, label='disassemble')
        env.components['workspace_scene'].brick_scene.export_ldraw(
            './disassembly_%i.mpd'%instance)
    
    return observation_seq, action_seq

def vis_obs(obs, i, j, label='tmp'):
    from splendor.image import save_image
    from ltron.visualization.drawing import (
        map_overlay, stack_images_horizontal)
    workspace_opacity = numpy.zeros((64,64,1))
    workspace_cursor = numpy.zeros((64,64,3), dtype=numpy.uint8)
    y, x, p, k, s = obs['workspace_cursor']
    workspace_opacity[y,x] = 1.
    if p == 0:
        workspace_cursor[y,x] = [255,0,0]
    else:
        workspace_cursor[y,x] = [0,0,255]
    workspace_image = map_overlay(
        obs['workspace_color_render'],
        workspace_cursor,
        workspace_opacity,
    )

    handspace_opacity = numpy.zeros((24,24,1))
    handspace_cursor = numpy.zeros((24,24,3), dtype=numpy.uint8)
    y, x, p, k, s = obs['handspace_cursor']
    handspace_opacity[y,x] = 1.
    if p == 0:
        handspace_cursor[y,x] = [255,0,0]
    else:
        handspace_cursor[y,x] = [0,0,255]
    handspace_image = map_overlay(
        obs['handspace_color_render'],
        handspace_cursor,
        handspace_opacity,
    )

    image = stack_images_horizontal(
        (workspace_image, handspace_image), align='bottom')
    path = './%s_%04i_%04i.png'%(label, i, j)
    save_image(image, path)
