import random
import copy

import numpy

from ltron.bricks.brick_type import BrickType
from ltron.bricks.brick_instance import BrickInstance
from ltron.gym.envs.reassembly_env import reassembly_template_action


# utilities ====================================================================

def snap_finder_condition(instances, snaps, pos_component, neg_component):
    # build a condition function which takes an observation and finds where
    # the specified instances and snaps are visible in the current scene
    def condition(observation):
        matching_yxpis = []
        for pp, render_name in enumerate((neg_component, pos_component)):
            for ii, ss in zip(instances, snaps):
                render = observation[render_name]
                y, x = numpy.where(
                    (render[:,:,0] == ii) & (render[:,:,1] == ss))
                p = numpy.ones(y.shape[0], dtype=numpy.long) * pp
                i = numpy.ones(y.shape[0], dtype=numpy.long) * ii
                s = numpy.ones(y.shape[0], dtype=numpy.long) * ss
                yxpis = numpy.stack((y, x, p, i, s), axis=0)
                matching_yxpis.append(yxpis)
        
        matching_yxpis = numpy.concatenate(matching_yxpis, axis=1)
        
        success = bool(matching_yxpis.shape[1])
        if success is False:
            import pdb
            pdb.set_trace()
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
                    numpy.abs(new_position[1:] - position[1:]))
                new_distance += modular_distance(
                    new_position[0], position[0], component.azimuth_steps)
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
):
    condition = snap_finder_condition(
        instances, snaps, pos_snap_component, neg_snap_component)
    start_camera_position = (
        tuple(state[viewpoint_component]['position']) + (0,))
    end_camera_position, visible_snaps = search_camera_space(
        env, viewpoint_component, state, condition, float('inf'))
    
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

def get_wip_connections(
    goal_config,
    new_instance,
    goal_to_start_lookup,
):
    connections = goal_config['edges'][0] == new_instance
    extant_connections = numpy.array([
        goal_config['edges'][1,i] in goal_to_start_lookup
        for i in range(goal_config['edges'].shape[1])])
    connections = connections & extant_connections
    connected_instances = goal_config['edges'][1][connections]
    wip_instances = [goal_to_start_lookup[i] for i in connected_instances]
    instance_snaps = goal_config['edges'][2][connections]
    wip_snaps = goal_config['edges'][3][connections]
    new_to_wip = {
        (i,s):(1,cs) for i, s, cs in
        zip(wip_instances, instance_snaps, wip_snaps)
    }
    wip_to_new = {v:k for k,v in new_to_wip.items()}
    
    return new_to_wip, wip_to_new, wip_instances, wip_snaps


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
    goal_to_start_lookup,
    class_to_brick_type,
):
    # initialization -----------------------------------------------------------
    # initialize the action squence
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
        return False

    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(brick_class, brick_color)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    
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
    action_seq.extend(handspace_camera_actions)
    
    # update the state
    if handspace_camera_actions:
        replace_camera_in_state(
            env, state, 'handspace_viewpoint', handspace_camera_position)
        observation = env.set_state(state)
    
    '''
    snap_y = []
    snap_x = []
    snap_p = []
    for snap, i in zip(upright_snaps, upright_snap_ids):
        if snap.polarity == '+':
            snap_map = observation['handspace_pos_snap_render']
        else:
            snap_map = observation['handspace_neg_snap_render']
        y, x = numpy.where((snap_map[:,:,0] == 1) & (snap_map[:,:,1] == i))
        snap_y.append(y)
        snap_x.append(x)
        snap_p.append(numpy.ones(len(y)) * (snap.polarity == '+'))

    snap_y = numpy.concatenate(snap_y)
    snap_x = numpy.concatenate(snap_x)
    snap_p = numpy.concatenate(snap_p)

    # TODO: search camera space for upright snaps
    if not len(snap_y):
        # TODO test other camera locations
        return False
    
    i = random.randint(0, len(snap_y)-1)
    y = snap_y[i]
    x = snap_x[i]
    p = snap_p[i]
    '''

    # make the pick and place action -------------------------------------------
    nr = random.randint(0, new_visible_snaps.shape[1]-1)
    y, x, p, i, s = new_visible_snaps[:,nr]
    
    pick_and_place_action = reassembly_template_action()
    pick_and_place_action['handspace_cursor'] = {
        'activate':True,
        'position':[y,x],
        'polarity':p,
    }
    pick_and_place_action['pick_and_place'] = {
        'activate':True,
        'place_at_origin':True,
    }
    action_seq.append(insert_action)
    observation = env.step(pick_and_place_action)

    return action_seq

def plan_add_nth_brick(
    env,
    goal_config,
    instance,
    observation,
    goal_to_wip_lookup
):
    
    # initialization -----------------------------------------------------------
    
    vis_obs(observation, 0)
    
    # initialize the action sequence
    action_seq = []
    
    # pull class, color and transform information out of the goal configuration
    brick_class = goal_config['class'][instance]
    brick_color = goal_config['color'][instance]
    brick_transform = goal_config['pose'][instance]
    
    # make the insert action ---------------------------------------------------
    insert_action = make_insert_action(brick_class, brick_color)
    action_seq.append(insert_action)
    observation, reward, terminal, info = env.step(insert_action)
    
    vis_obs(observation, 1)
    
    # workspace camera ---------------------------------------------------------
    # find all connections between the new instance and the wip bricks
    new_to_wip, wip_to_new, wip_instances, wip_snaps = get_wip_connections(
        goal_config, instance, goal_to_wip_lookup)
    
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
    )
    action_seq.extend(workspace_camera_actions)
    
    # update the state
    if workspace_camera_actions:
        replace_camera_in_state(
            env, state, 'workspace_viewpoint', workspace_camera_position)
        observation = env.set_state(state)
    
    vis_obs(observation, 2)
    
    # handspace camera ---------------------------------------------------------
    # find the new snaps that are connected to the wip visible snaps
    wy, wx, wp, wi, ws = wip_visible_snaps
    new_instances = numpy.ones(ws.shape[0], dtype=numpy.long)
    new_snaps = [wip_to_new[i,s][1] for i,s in zip(wi, ws)]
    
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
    action_seq.extend(handspace_camera_actions)
    
    # update the state
    if handspace_camera_actions:
        replace_camera_in_state(
            env, state, 'handspace_viewpoint', handspace_camera_position)
        observation = env.set_state(state)
    
    vis_obs(observation, 3)
    
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
        'position':[wyy, wxx],
        'polarity':wpp,
    }
    pick_and_place_action['handspace_cursor'] = {
        'activate':True,
        'position':[nyy, nxx],
        'polarity':npp,
    }
    pick_and_place_action['pick_and_place'] = {
        'activate':True,
        'place_at_origin':False,
    }
    action_seq.append(pick_and_place_action)
    
    # take the action to generate the latest observation
    observation, reward, terminal, info = env.step(pick_and_place_action)

    vis_obs(observation, 4)

    # make the rotation action -------------------------------------------------

    import pdb
    pdb.set_trace()
    
    return action_seq


def vis_obs(obs, i):
    from splendor.image import save_image
    from ltron.visualization.drawing import (
        map_overlay, stack_images_horizontal)
    workspace_opacity = numpy.zeros((64,64,1))
    workspace_cursor = numpy.zeros((64,64,3), dtype=numpy.uint8)
    y, x, p, j, s = obs['workspace_cursor']
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
    y, x, p, j, s = obs['handspace_cursor']
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
    path = './tmp_%i.png'%i
    save_image(image, path)

# DETRITUS =====================================================================

'''
def plan_add_nth_brick_workspace_camera(env, state, instances, snaps):
    # compute the necessary workspace camera motion
    condition = snap_finder_condition(
        instances,
        snaps,
        'workspace_pos_snap_render',
        'workspace_neg_snap_render',
    )
    start_workspace_camera_position = (
        tuple(state['workspace_viewpoint']['position']) + (0,))
    workspace_camera_position, workspace_snaps = self.search_camera_space(
        'workspace_viewpoint', state, condition, float('inf'))
    
    # convert the workspace camera motion to camera actions
    workspace_camera_actions = self.compute_camera_actions(
        'workspace',
        start_workspace_camera_position,
        workspace_camera_position,
    )
    
    return workspace_camera_actions, workspace_camera_position

def plan_add_nth_brick_workspace_camera(env, state, instances, snaps):
    condition = snap_finder_condition(
        instances,
        snaps,
        'handspace_pos_snap_render',
        'handspace_neg_snap_render',
    )
    start_handspace_camera_position = (
        tuple(state['handspace_viewpoint']['position']) + (0,))
    handspace_camera_position, handspace_snaps = self.search_camera_space(
        'handspace_viewpoint', state, condition, float('inf'))
    
    # convert the handspace camera motion to camera actions
    handspace_camera_actions = self.compute_camera_actions(
        'handspace',
        start_handspace_camera_position,
        handspace_camera_position,
    )
'''
