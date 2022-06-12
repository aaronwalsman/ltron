def build_expert(
    current_assembly,
    target_assembly,
    part_names,
    camera_moves_fn,
):
    
    raise Exception('DEPRECATED')
    
    # compute the best alignment and find misaligned bricks
    matches, offset = match_assemblies(
        current_assembly, target_assembly, self.part_names)
    misaligned, false_positives, false_negatives = compute_unmatched(
        matches, current_assembly, target_assembly)
    
    possible_actions = []
    
    # If there are misaligned bricks in the scene, first look for a way to
    # rotate or translate them into place.
    if len(misaligned):
        for current_i, target_i in misaligned:
            # do any of the misaligned bricks have a correct connection
            for current_i, target_i in misaligned:
                target_edges = target_assembly['edges'][0] == target_i
                target_edges = target_assembly['edges'][:,target_edges]
                
                current_edges = current_assembly['edges'][0] == current_i
                current_edges = current_assembly['edges'][:,current_edges]
                
                for tgt_i, tgt_con_i, tgt_s, tgt_con_s in target_edges.T:
                    if tgt_i == 0:
                        break
                    for cur_i, cur_con_i, cur_s, cur_con_s in current_edges.T:
                        if cur_i == 0:
                            break
                        if ((cur_con_i, tgt_con_i) in matches and
                            tgt_s == cur_s and tgt_con_s == cur_con_s
                        ):
                            actions = ADJUST_ABOUT_SNAP_ACTIONS()
                            possible_actions.extend(actions)
            
            if possible_actions:
                return possible_actions + camera_moves_fn()
            
            # is it possible to drag-and-drop the misaligned brick into place
            for current_i, target_i in misaligned:
                target_edges = target_assembly['edges'][0] == target_i
                target_edges = target_assembly['edges'][:,target_edges]
                
                for tgt_i, tgt_con_i, tgt_s, tgt_con_s in target_edges.T:
                    if tgt_i == 0:
                        break
                    if tgt_con_i in [m[1] for m in matches]:
                        actions = DRAG_AND_DROP_ACTIONS()
                        possible_actions.extend(actions)
            
            if possible_actions:
                return possible_actions + camera_moves_fn()
            
            # if it's not possible to rotate or translate the brick into place
            # can we delete it?
            for current_i, target_i in misaligned:
                actions = REMOVE_BRICK()
                possible_actions.extend(actions)
        
        # If there are misaligned bricks, but we can't find a way to rotate,
        # translate, or remove them, try a different camera angle
        return camera_moves_fn()
    
    # If there are false positives in the scene, remove them.
    if len(false_positives):
        for false_positive in false_positives:
            actions = REMOVE_BRICK()
            possible_actions.extend(actions)
        
        return possible_actions + camera_moves_fn()
    
    # If there are false negatives, remove them from the scene.
    if len(false_negatives):
        for false_negative in false_negatives:
            # Need to check collision map
            actions = ADD_BRICK()
            possible_actions.extend(actions)
        
        return possible_actions + camera_moves_fn()

