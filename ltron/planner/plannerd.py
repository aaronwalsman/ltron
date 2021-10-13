import time
import math

import numpy

from ltron.exceptions import LTronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map
from ltron.geometry.epsilon_array import EpsilonArray
from ltron.matching import match_configurations, match_lookup
from ltron.bricks.brick_instance import BrickInstance
from ltron.bricks.brick_type import BrickType

class PlanningException(LTronException):
    pass

class PathNotFoundError(PlanningException):
    pass

def brick_type_has_upright_snaps(brick_type):
    for snap in brick_type.snaps:
        

def node_connected_collision_free(
    new_brick,
    new_brick_type_name,
    existing_bricks,
    goal_state,
    collision_map,
    goal_configuration,
):
    if not len(existing_bricks):
        # This is actually not a good check, the collision map only contains
        # connected snaps.
        #for axis, polarity, snaps in collision_map[new_brick]:
        #    if (polarity and numpy.allclose(axis, (0,-1,0))) or (
        #        not polarity and numpy.allclose(axis, (0,1,0))
        #    ):
        #        return True
        #
        #else:
        #    return False
        brick_type = BrickType(new_brick_type_name)
        brick_transform = goal_configuration['pose'][new_brick]
        
        return bool(len(brick_type.get_upright_snaps()))
    
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

class Roadmap:
    def __init__(self, env, goal_config, class_ids, color_ids):
        self.env = env
        self.goal_config = goal_config
        self.goal_state = frozenset(numpy.where(goal_config['class'])[0])
        self.nodes = set()
        self.nodes.add(self.goal_state)
        self.edges = {}
        self.successors = {}
        self.env_states = {}
        self.class_ids = class_ids
        self.id_classes = {value:key for key, value in self.class_ids.items()}
        self.color_ids = color_ids
        
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_configuration(
            self.goal_config, self.class_ids, self.color_ids)
        self.goal_collision_map = build_collision_map(goal_scene)

class RoadmapPlanner:
    def __init__(self, roadmap, start_env_state):
        
        # init
        self.roadmap = roadmap
        self.edge_checker = EdgeChecker(self.roadmap)
        
        # set the env to the start state
        observation = self.roadmap.env.set_state(start_env_state)
        self.start_config = observation['workspace_scene']['config']
        
        # compute a matching
        matching, offset = match_configurations(
            self.start_config, roadmap.goal_config)
        (start_to_goal_lookup,
         goal_to_start_lookup,
         false_positives,
         false_negatives) = match_lookup(
            matching, self.start_config, roadmap.goal_config)
        
        self.start_state = frozenset(goal_to_start_lookup.keys())
        self.roadmap.nodes.add(self.start_state)
        assert self.start_state not in self.roadmap.env_states
        self.roadmap.env_states[self.start_state] = start_env_state
        
        self.visits = {}
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_configuration(
            self.start_config, self.roadmap.class_ids, self.roadmap.color_ids)
        self.start_collision_map = build_collision_map(start_scene)
    
    def greedy_path(self):
        current_state = self.start_state
        path = [self.start_state]
        while current_state != self.roadmap.goal_state:
            successors = self.roadmap.successors[current_state]
            edges = [(current_state, successor) for successor in successors]
            best_q, current_state = max(zip(
                (self.visits[edge]['q'] for edge in edges),
                successors))
            path.append(current_state)
        
        return path
    
    def plan(self, max_cost=float('inf'), timeout=float('inf')):
        t_start = time.time()
        #w = 0
        #n = 0
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            # plan a path
            candidate_path, goal_found = self.plan_collision_free()
            truncated_path, goal_feasible = self.edge_checker.check_path(
                candidate_path)
            
            #if goal_feasible:
            #    q = 1
            #else:
            #    q = -1
            #w += q
            #n += 1
            #print(w/n)
            
            self.update_path_visits(truncated_path, q)
    
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
                if node_connected_collision_free(
                    false_negative,
                    self.roadmap.id_classes[false_negative],
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

class EdgeChecker:
    def __init__(self, roadmap):
        self.roadmap = roadmap
    
    def check_path(self, candidate_path):
        truncated_path = [candidate_path[0]]
        for a, b in zip(candidate_path[:-1], candidate_path[1:]):
            edge = self.roadmap.edges[a,b]
            if b not in self.roadmap.env_states:
                try:
                    self.check_edge(a, b)
                    truncated_path.append(b)
                    last_state = self.roadmap.env.get_state()
                    self.roadmap.env_states[b] = last_state
                except EdgeNotConnectableError:
                    return truncated_path, False
            else:
                truncated_path.append(b)
        else:
            return truncated_path, True
    
    def check_edge(self, a, b):
        env_state = self.roadmap.env_states[a]
        observation = self.roadmap.env.set_state(env_state)
        
        assert abs(len(a) - len(b)) == 1
        
        if len(a) < len(b):
            # add a brick
            if len(a) == 0:
                # add the first brick
                action_seq = self.plan_add_first_brick(next(iter(b)))
            
            else:
                # add the nth brick
                action_seq = self.plan_add_nth_brick(next(iter(b-a)))
        
        elif len(b) < len(a):
            # remove a brick
            action_seq = self.plan_remove_nth_brick(next(iter(a-b)))
    
    def plan_add_first_brick(self, instance):
        brick_class = self.roadmap.goal_config['class'][instance]
        brick_color = self.roadmap.goal_config['color'][instance]
        brick_type = BrickType(self.roadmap.id_classes[brick_class])
        if not len(brick_type.get_upright_snaps):
            
        
        # the only thing we need to check for is non-upright snaps,
        # if that check passes, everything else should just work
        # actually, this is already checked at the high level... oh, but we
        # still need to find the snap that is upright.

    def plan_add_nth_brick(self, instance):
        pass
    
    def plan_remove_brick(self, instance):
        pass

class EdgePlanner:
    def __init__(self, edge_checker):
        self.edge_checker = edge_checker
        self.visits = {}
    
    def plan(self):
        # what's the right stopping criteria?
        # 1. number of trajectories?
        # 2. timeout?
        # 3. definitely break if we find a perfect (no camera motion) path
        while True:
            edge_trajectory = self.rollout_trajectory()
            self.update_visit_stats(edge_trajectory)
            
    
    def observation_to_candidate_actions(self, observation):
        raise NotImplementedError
    
    

def upper_confidence_bound(q, n_action, n_state, c=2**0.5):
    return q + c * (math.log(n_state+1)/(n_action+1))**0.5
