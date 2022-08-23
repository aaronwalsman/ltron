from ltron.matching import match_assemblies, compute_unmatched
from ltron.geometry.collision import build_collision_map

def brick_expert(
    env,
    goal_assembly,
    observation,
    goal_to_wip,
    shape_id_to_brick_shape,
    split_cursor_actions=False,
    allow_snap_flip=False,
    debug=False,
):
    
class BrickExpert:
    def __init__(self, goal_configuration, shape_ids, color_ids):
        self.goal_configuration = goal_configuration
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        
        temp_scene = BrickScene(renderable=True, track_snaps=True)
        temp_scene.import_assembly(
            self.goal_configuration, self.shape_ids, self.color_ids)
        self.collision_map = build_collision_map(temp_scene)
    
    def label(self, observation):
        # figure out what in the table assembly corresponds to the goal assembly
        best_matches, best_offset = match_assemblies(
            observation['table_assembly'],
            self.goal_assembly,
            self.shape_ids,
        )
        misaligned_matches, unmatched_a, unmatched_b = compute_unmatched(
            best_matches, observation['table_assembly'], goal_assembly)
        
        mode_mask = numpy.zeros(NUM_MODES, dtype=numpy.bool)
        table_mask = numpy.zeros((th, tw, 2), dtype=numpy.bool)
        hand_mask = numpy.zeros((hh, hw, 2), dtype=numpy.bool)
        
        # if there are no false positives or false negatives, we are done
        if not len(missed_table) and not len(missed_goal):
            mode_mask[SWITCH_ACTION] = 1
        
        # if there are no misplaced bricks
        if len(misaligned_matches) == 0):
            
            # is the thing in the hand is on the path of righteousness
            if righteousness:
                # pick and place
                pass
            
            else:
                # pick up a good thing OOF, NO WAY TO MASK THIS THOUGH
            
        
        # if there's a single misplaced brick
        elif len(misaligned_matches) == 1:
            pass
        
        # if there's multiple misplaced bricks
        elif len(misaligned_matches) > 1:
            # for now end early
            return None
        
