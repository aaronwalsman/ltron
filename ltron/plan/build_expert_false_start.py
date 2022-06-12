from ltron.matching import match_assemblies
from ltron.exceptions import LtronException

class ExpertException(LtronException):
    pass

class ExpertPickException(ExpertException):
    pass

class ExpertPlaceException(ExpertException):
    pass

def build_expert(
    actor,
    current_assembly,
    target_assembly,
    secondary_assemblies,
    shape_names,
):
    
    # compute the closest matching between the current and target assemblies
    matches, offset = match_assemblies(
        current_assembly, target_assembly, shape_names)
    current_to_target = dict(matches)
    target_to_current = {v:k for k,v in current_to_target.items()}
    
    # compute the best alignment between disconnected components
    (current_to_target_misaligned_connected,
     target_to_current_misaligned_connected,
     current_to_target_misaligned_disconnected,
     target_to_current_misaligned_disconnected,
     false_positives,
     false_negatives) = compute_misaligned(
        current_assembly, target_assembly, matches
    )
    
    # if there are no problems, return the finish action
    if not (
        len(current_to_target_misaligned_connected) or
            len(current_to_target_misaligned_disconnected) or
            len(false_positives) or
            len(false_negatives)
        ):
            actions = [actor.finish_action()]
    
    # if there are connected misaligned bricks, adjust the misaligned brick
    elif len(target_to_current_misaligned_connected):
        actions = adjust_connection(
            current_to_target,
            target_to_current,
            target_to_current_misaligned_connected,
            current_assembly,
            target_assembly,
        )
    
    # if there are disconnected misaligned bricks, make a connection
    elif len(target_to_current_misaligned_disconnected):
        actions = make_connection(
            current_to_target,
            target_to_current,
            list(target_to_current_misaligned_disconnected.keys()),
            current_assembly,
            target_assembly,
            {...},
        )
    
    # if there are false positives, remove them
    elif len(false_positives):
        raise NotImplementedError
    
    # if the current scene is empty, place the first brick
    elif not len(current_to_target):
        actions = add_first_brick(
            false_negatives,
            target_assembly,
            secondary_assemblies,
        )
    
    # if a brick is missing, add it
    elif len(false_negatives):
        actions = add_nth_brick(
            SOMETHING,
        )
    
    return actions

def adjust_connection(
    current_to_target,
    target_to_current,
    targets_to_fix,
    current_assembly,
    target_assembly,
    actor,
):
    actions = []
    
    pickable = {}
    for tgt_i, cur_set in targets_to_fix.items():
        for cur_i, cur_s, cur_con_s in cur_set:
            fn_edge_indices = matching_edges(
                target_assembly, i1=tgt_i, s1=cur_s, s2=cur_con_s)
            fn_edges = target_assembly['edges'][:,fn_edge_indices]
            if fn_edges.shape[1]:
                tgt_con_i = fn_edges[1,0]
                pickable[TARGET_SCENE, cur_i, cur_s] = [
                    [TARGET_SCENE, tgt_i, tgt_con_i, cur_con_s]]
    
    pick_n, pick_i, pick_s = actor.get_picked_snap()
    
    if (pick_n, pick_i, pick_s) not in pickable:
        pick_actions = []
        pick_names = []
        for n, i, s in pickable:
            pick_actions.extend(actor.actions_to_pick_snap(n, i, s))
            pick_names.append(n)
        
        if always_include_viewpoint_actions or not len(pick_actions):
            for pick_name in pick_names:
                pick_actions.extend(actor.get_viewpoint_actions(pick_name))
        
        if not pick_actions:
            raise ExpertPickException
        
        return pick_actions
