import time

from ltron.exceptions import LTronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map
from ltron.matching import match_configurations, match_lookup

class PlanningException(LTronException):
    pass

class PathNotFoundError(PlanningException):
    pass

class RoadMap:
    def __init__(self, class_ids, color_ids):
        self.nodes = set()
        self.edges = {}
        self.class_ids = class_ids
        self.color_ids = color_ids
    
    def plan(self,
        start_config,
        goal_config,
        max_cost=float('inf'),
        timeout=float('inf')
    ):
        t_start = time.time()
        
        # compute a matching
        matching, offset = match_configurations(start_config, goal_config)
        (start_to_goal_lookup,
         goal_to_start_lookup,
         false_positives,
         false_negatives) = match_lookup(matching, start_config, goal_config)
        
        # build scenes and collision maps
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_configuration(
            start_config, self.class_ids, self.color_ids)
        start_collision_map = build_collision_map(start_scene)
        
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_configuration(
            goal_config, self.class_ids, self.color_ids)
        goal_collision_map = build_collision_map(goal_scene)
        
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            # rollout a trajectory
            high_level_path = self.high_level_rollout(start, goal)
            
            # check the edge connectivity everywhere
    
    def high_level_rollout(start, goal, visit_stats):
        current = start
        path = [start]
        
        while current != goal:
            # if we are at an unexplored node, then find the neighbors
            if current not in visit_stats:
                self.expand_high_level_node(current)
            
            # sample a high level node based on visit counts
            current = self.sample_next(...)
            path.append(current)
        
        return path
