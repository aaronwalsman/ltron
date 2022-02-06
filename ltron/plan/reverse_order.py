import random

import numpy

from ltron.geometry.collision import build_collision_map
from ltron.plan.edge_planner import plan_remove_nth_brick

def get_visible_snaps(observation):
    
    #visible_instances = observation['table_mask_render'].reshape(-1)
    
    v = set()
    for polarity in 'pos', 'neg':
        snaps = observation['table_%s_snap_render'%polarity].reshape(-1,2)
        nonzero = numpy.where(snaps[:,0])
        nonzero_snaps = snaps[nonzero]
        #vi = visible_instances[nonzero]
        #v = v | {(i, s) for (i, s),j in zip(nonzero_snaps, vi) if i == j}
        v = v | {(i,s) for i,s in nonzero_snaps}

    return v

def build_connectivity_graph(node, edges):
    connectivity = {}
    for i in range(edges.shape[1]):
        if edges[0,i] in node and edges[1,i] in node:
            connectivity.setdefault(edges[0,i], set())
            connectivity[edges[0,i]].add(edges[1,i])
            connectivity.setdefault(edges[1,i], set())
            connectivity[edges[1,i]].add(edges[0,i])
    
    return connectivity

def brick_splits_graph(i, graph):
    assert i in graph
    if len(graph) <= 2:
        return False
    
    if len(graph[i]) == 1:
        return False
    
    connected = {i}
    frontier = [next(iter(graph[i]))]
    while frontier:
        brick = frontier.pop()
        if brick in connected:
            continue
        
        connected.add(brick)
        frontier.extend(graph[brick])
    
    return connected != graph.keys()

def get_collision_free_snaps(i, map_subset, collision_map):
    collision_free_snaps = set()
    for a, p, snap_group in collision_map[i]:
        colliding_bricks = collision_map[i][a, p, snap_group]
        colliding_bricks = frozenset(colliding_bricks)
        if not len(colliding_bricks & map_subset):
            #collision_free_snaps.extend(snap_group)
            collision_free_snaps |= {(i,s) for s in snap_group}
    
    return collision_free_snaps

def break_plan(env, first_observation, shape_ids, color_ids):
    shape_names = {value:key for key, value in shape_ids.items()}
    table_scene = env.components['table_scene'].brick_scene
    collision_map = build_collision_map(table_scene)
    
    start_assembly = first_observation['table_assembly']
    edges = start_assembly['edges']
    connected_snaps = (
        {(i_a, s_a) for i_a, i_b, s_a, s_b in edges.T} |
        {(i_b, s_b) for i_a, i_b, s_a, s_b in edges.T}
    )
        
    start_membership = frozenset(int(i) for i in table_scene.instances)
    
    observation = first_observation
    visible_snaps = get_visible_snaps(observation)
    current_node = start_membership
    current_path = (start_membership,)
    observation_seq = []
    action_seq = []
    while len(current_node):
        
        print(current_node)
        
        # build a connectivity graph
        graph = build_connectivity_graph(current_node, edges)
        
        for i in current_node:
            # can I remove this brick without splitting the model into
            # multiple islands?
            if len(current_node) > 1 and brick_splits_graph(i, graph):
                continue
            
            # can I remove this brick without causing a collision?
            collision_free_snaps = get_collision_free_snaps(
                i, current_node, collision_map)
            if not len(collision_free_snaps):
                continue
            
            still_connected = {
                (ii,s) for ii, s in connected_snaps if ii in current_node}
            
            # is there a collision-free snap that's connected and visible?
            removable_snaps = (
                visible_snaps & still_connected & collision_free_snaps)
            print('hl', i, removable_snaps,
                len(visible_snaps),
                len(still_connected),
                len(collision_free_snaps),
                len(still_connected & collision_free_snaps),
                len(visible_snaps & collision_free_snaps),
            )
            #print('  cs', {(ii,s) for ii,s in connected_snaps if ii == i})
            if not len(removable_snaps):
                continue
            
            # maybe do the biggest one instead?
            i, s = random.choice(list(removable_snaps))
            obs, act = plan_remove_nth_brick(
                env,
                None,
                i,
                observation,
                {i:i},
                shape_names,
                use_mask=False,
            )
            observation_seq.append(obs)
            action_seq.append(act)
            
            observation = obs[-1]
            visible_snaps = get_visible_snaps(observation)
            current_node = current_node - {i}
            current_path = current_path + (current_node,)
            break
        
        else:
            import pdb
            pdb.set_trace()
    
    aa = sum(action_seq, [])
    cc = [a['table_viewpoint'] for a in aa]
    print(cc)
    import pdb
    pdb.set_trace()
