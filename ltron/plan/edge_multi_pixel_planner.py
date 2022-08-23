def plan_add_nth_brick(
    env,
    goal_assembly,
    instance,
    observation,
    goal_to_wip,
    shape_id_to_brick_shape,
    split_cursor_actions=True,
):
    
    # initialization -----------------------------------------------------------
    wip_to_goal = {v:k for k,v in goal_to_wip.items()}
    new_wip_instance = max(goal_to_wip.values())+1
    
    # initialize the action sequence
    observation_seq = [obervation]
    click_masks = []
    action_seq = []
    reward_seq = []
    
    # pull shape, color and transform information out of the goal assembly
    brick_shape = goal_assembly['shape'][instance]
    brick_color = goal_assembly['color'][instance]
    brick_transform = goal_assembly['pose'][instance]
    
    # make the insert action ---------------------------------------------------
    hand_assembly = observation['hand_assembly']
    hand_shape = hand_assembly['shape'][1]
    hand_color = hand_assembly['color'][1]
    if hand_shape != brick_shape or hand_color != brick_color:
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
        'table_instance_render',
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
    
    
