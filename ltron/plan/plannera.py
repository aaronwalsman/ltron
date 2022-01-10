# configuration space:
# a brick configuration:
# list of classes, colors, poses, (and edges that comes along for the ride)
# camera position

from ltron.bricks.brick_scene import BrickScene
from ltron.matching import match_configurations, match_lookup
from ltron.geometry.collision import build_collision_map

class 

def planner(
    start_config,
    goal_config,
    shape_ids,
    color_ids,
    goal_collision_map = None,
):
    
    # setup
    # do a matching
    
    # OK, here's what we've figured out so far
    # a NODE in our planning graph is an entire configuration
    # the EDGES are the add/remove actions that get us between them.
    
    # Actually, even better:
    # For the high-level, a NODE is a sorted tuple of instance ids
    # (relative to the goal configuration)
    # the EDGES are add/remove operations
    
    # iterate until goal is reached
    frontier = [(0, start_config)]
    while frontier:
        config = frontier.pop()
        visited.add(config) # this wont work, config needs to be hashable

def planner_old(
    start_config,
    goal_config,
    shape_ids,
    color_ids,
    goal_collision_map = None,
):
    
    if goal_collision_map is None:
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_configuration(goal_config, shape_ids, color_ids)
        
        goal_collision_map = build_collision_map(goal_scene)
    
    finished = False
    while not finished:
        matching = match_configurations(start_config, goal_config)
        (start_to_goal,
         goal_to_start,
         misplaced_start,
         unplaced_goal) = match_lookup(matching, start_config, goal_config)
        
        # TODO: this part needs to be fleshed out
        if len(misplaced_start):
            success, remove_brick = pick_brick_to_remove()
            if not success:
                block_edge()
            success = low_level_remove(remove_brick)
            if not success:
                block_edge()
        
        elif blockage_detected:
            # TODO: disassemble until you unblock the blockage
            pass
        
        elif len(unplaced_goal):
            pick_brick_to_add(current_config, )
            #successs, add_brick = pick_brick_to_add()
            #if not success:
                # TODO: need to figure out how to unblock a brick
                #pass
                #block_edge()
            success = low_level_add(add_brick)
            if not success:
                block_edge()
        
        else:
            finished = True
