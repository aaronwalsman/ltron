import time
import math

import numpy

from ltron.exceptions import LTronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map
from ltron.geometry.epsilon_array import EpsilonArray
from ltron.matching import match_configurations, match_lookup

class PlanningException(LTronException):
    pass

class PathNotFoundError(PlanningException):
    pass

'''
def configuration_to_state(config, global_offset=None):
    class_ids = tuple(config['class'])
    color_ids = tuple(config['color'])
    poses = config['pose']
    if global_offset is not None:
        poses = global_offset @ poses
    poses = tuple(EpsilonArray(pose) for pose in poses)
    return frozenset(
        (class_id, color_id, pose)
        for class_id, color_id, pose in zip(class_ids, color_ids, poses)
        if class_id != 0
    )
'''

def node_addable(
    new_brick,
    existing_bricks,
    goal_state,
    collision_map,
    goal_configuration,
):
    if not len(existing_bricks):
        for axis, polarity, snaps in collision_map[new_brick]:
            if (polarity and numpy.allclose(axis, (0,-1,0))) or (
                not polarity and numpy.allclose(axis, (0,1,0))
            ):
                return True
        
        else:
            return False
    
    else:
        # there must be a connection to a new brick
        new_edges = numpy.where(goal_configuration['edges'][0] == new_brick)[0]
        connected_bricks = goal_configuration['edges'][1, new_edges]
        connected_bricks = frozenset(connected_bricks)
        if not len(connected_bricks & existing_bricks):
            return False
        
        updated_bricks = existing_bricks | frozenset((new_brick,))

        unadded_bricks = goal_state - updated_bricks
        for unadded_brick in unadded_bricks:
            for axis, polarity, snap_group in collision_map[unadded_brick]:
                colliding_bricks = collision_map[unadded_brick][
                    axis, polarity, snap_group]
                colliding_bricks = frozenset(colliding_bricks)
                if not len(colliding_bricks & updated_bricks):
                    break
            
            else:
                return False
        
        return True

'''
class GraphEdge:
    def __init__(self, collision_free, viewpoint_change, feasible):
        self.collision_free = collision_free
        self.viewpoint_change = viewpoint_change
        self.feasible = feasible
'''
class Planner:
    def __init__(self, roadmap, start_config):
        
        # compute a matching
        matching, offset = match_configurations(
            start_config, roadmap.goal_config)
        (start_to_goal_lookup,
         goal_to_start_lookup,
         false_positives,
         false_negatives) = match_lookup(
            matching, start_config, roadmap.goal_config)
        
        self.roadmap = roadmap
        self.start_config = start_config
        self.start_state = frozenset(goal_to_start_lookup.keys())
        self.roadmap.nodes.add(self.start_state)
        
        self.visits = {}
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_configuration(
            start_config, self.roadmap.class_ids, self.roadmap.color_ids)
        self.start_collision_map = build_collision_map(start_scene)
        
    def plan(self, max_cost=float('inf'), timeout=float('inf')):
        t_start = time.time()
        w = 0
        n = 0
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            # plan a path
            candidate_path, goal_found = self.plan_collision_free()
            if goal_found:
                q = 1
            else:
                q = -1
            w += q
            n += 1
            print(w/n)
            
            self.update_path_visits(candidate_path, q)
    
    def plan_collision_free(self):
        current_state = self.start_state
        path = [self.start_state]
        goal_found = True
        
        while current_state != self.roadmap.goal_state:
            # add the current state to the road map nodes
            if current_state not in self.roadmap.successors:
                self.expand_successors(current_state)
            # add the current state to the visit statistics
            if current_state not in self.visits:
                self.expand_visits(current_state)
            
            # sample a high level node based on visit counts
            current_state = self.sample_collision_free(current_state)
            if current_state is None:
                goal_found = False
                break
            else:
                path.append(current_state)
        
        return path, goal_found
    
    def expand_successors(self, state):
        
        # compute false positives and false negatives
        false_positives = state - self.roadmap.goal_state
        false_negatives = self.roadmap.goal_state - state
        
        # compute possible successors
        successors = []
        
        # if there are any false positives, remove them first
        if false_positives:
            for false_positive in false_positives:
                successor = state - frozenset((false_positive,))
                successors.append(successor)
        
        # if there are no false positives, but false negatives, add them
        elif false_negatives:
            for false_negative in false_negatives:
                # check if false_negative can be added
                if node_addable(
                    false_negative,
                    state,
                    self.roadmap.goal_state,
                    self.roadmap.goal_collision_map,
                    self.roadmap.goal_config,
                ):
                    successor = state | frozenset((false_negative,))
                    successors.append(successor)
        
        # add the succesors to the roadmap
        self.roadmap.successors[state] = successors
        
        # add the edges and successors to the roadmap
        for successor in successors:
            self.roadmap.edges[state, successor] = {
                'collision_free':True,
                'viewpoint_change':None,
                'feasible':None,
            }
            self.roadmap.nodes.add(successor)
    
    def expand_visits(self, state):
        self.visits[state] = {'n':1}
        for successor in self.roadmap.successors[state]:
            self.visits[state, successor] = {'n':0, 'q':0, 'w':0}
    
    def update_path_visits(self, path, q):
        for a, b in zip(path[:-1], path[1:]):
            self.visits[a,b]['w'] += q
            self.visits[a,b]['n'] += 1
            self.visits[a,b]['q'] = self.visits[a,b]['w']/self.visits[a,b]['n']
    
    def sample_collision_free(self, state):
        successors = self.roadmap.successors[state]
        if len(successors):
            u = [
                upper_confidence_bound(
                    q=self.visits[state, successor]['q'],
                    n_action=self.visits[state, successor]['n'],
                    n_state=self.visits[state]['n'],
                )
                for successor in successors]
            best_u, best_successor = max(zip(u, successors))
            return best_successor
        else:
            return None

class RoadMap:
    def __init__(self, goal_config, class_ids, color_ids):
        self.goal_config = goal_config
        #self.goal_state = configuration_to_state(self.goal_config)
        self.goal_state = frozenset(numpy.where(goal_config['class'])[0])
        self.nodes = set()
        self.nodes.add(self.goal_state)
        self.edges = {}
        self.successors = {}
        self.class_ids = class_ids
        self.color_ids = color_ids
        
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_configuration(
            self.goal_config, self.class_ids, self.color_ids)
        self.goal_collision_map = build_collision_map(goal_scene)
    
    def new_planner(self, start_config):
        return Planner(self, start_config)
    
    def plan(self,
        start_config,
        max_cost=float('inf'),
        timeout=float('inf')
    ):
        t_start = time.time()
        
        # compute a matching
        matching, offset = match_configurations(start_config, self.goal_config)
        (start_to_goal_lookup,
         goal_to_start_lookup,
         false_positives,
         false_negatives) = match_lookup(
            matching, start_config, self.goal_config)
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_configuration(
            start_config, self.class_ids, self.color_ids)
        start_collision_map = build_collision_map(start_scene)
        
        #start_state = configuration_to_state(start_config, offset)
        start_state = frozenset(goal_to_start_lookup.keys())
        self.nodes.add(start_state)
        
        visit_stats = {}
        
        W = 0
        N = 0
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            # plan a path
            high_level_path, goal_found = self.high_level_plan(
                start_state, visit_stats, start_collision_map)
            
            if goal_found:
                # check the edge connectivity everywhere
                Q = 1
            else:
                Q = -1
            
            W += Q
            N += 1
            print(W/N)
            
            for a, b in zip(high_level_path[:-1], high_level_path[1:]):
                visit_stats[a,b]['W'] += Q
                visit_stats[a,b]['N'] += 1
                visit_stats[a,b]['Q'] = (
                    visit_stats[a,b]['W'] / visit_stats[a,b]['N'])
            
    def high_level_plan(self, start_state, visit_stats, start_collision_map):
        current_state = start_state
        path = [start_state]
        goal_found = True
        
        while current_state != self.goal_state:
            # if we are at an unexplored node, then find the neighbors
            if current_state not in visit_stats:
                self.expand_high_level_state(
                    current_state, visit_stats, start_collision_map)
            
            # sample a high level node based on visit counts
            current_state = self.sample_next(current_state, visit_stats)
            if current_state is None:
                goal_found = False
                break
            else:
                path.append(current_state)
        
        return path, goal_found
    
    def expand_high_level_state(self, state, visit_stats, start_collision_map):
        false_positives = state - self.goal_state
        false_negatives = self.goal_state - state
        
        successors = []
        if false_positives:
            for false_positive in false_positives:
                successor = state - frozenset((false_positive,))
                successors.append(successor)
        
        elif false_negatives:
            for false_negative in false_negatives:
                # check if false_negative can be added
                # (does will it block anything else that hasn't been added yet?)
                if node_addable(
                    false_negative,
                    state,
                    self.goal_state,
                    #start_collision_map,
                    self.goal_collision_map,
                    self.goal_config,
                ):
                    successor = state | frozenset((false_negative,))
                    successors.append(successor)
        
        self.successors[state] = successors
        visit_stats[state] = {'N':1}
        
        for successor in successors:
            self.edges[state, successor] = GraphEdge(
                collision_free=True,
                viewpoint_change=None,
                feasible=None,
            )
            self.nodes.add(successor)
            visit_stats[state, successor] = {'N':0, 'Q':0, 'W':0}
    
    def sample_next(self, state, visit_stats):
        successors = self.successors[state]
        if len(successors):
            U = [
                upper_confidence_bound(
                    q=visit_stats[state, successor]['Q'],
                    n_action=visit_stats[state, successor]['N'],
                    n_state=visit_stats[state]['N'],
                )
                for successor in successors]
            best_u, best_successor = max(zip(U, successors))
            return best_successor
        else:
            return None

def upper_confidence_bound(q, n_action, n_state, c=2**0.5):
    return q + c * (math.log(n_state+1)/(n_action+1))**0.5
