import random
import time
import math
import copy
from bisect import insort

import tqdm

import numpy

from ltron.exceptions import LtronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map
from ltron.matching import match_assemblies, match_lookup
from ltron.bricks.brick_instance import BrickInstance
from ltron.bricks.brick_shape import BrickShape

from ltron.plan.edge_planner import (
    plan_add_first_brick,
    plan_add_nth_brick,
    plan_remove_nth_brick,
)

class PlanningException(LtronException):
    pass

class PathNotFoundError(PlanningException):
    pass

class FrontierError(PlanningException):
    pass

def node_connected_collision_free(
    new_brick,
    new_brick_shape_name,
    existing_bricks,
    goal_state,
    collision_map,
    goal_assembly,
):
    if not len(existing_bricks):
        brick_shape = BrickShape(new_brick_shape_name)
        brick_transform = goal_assembly['pose'][new_brick]
        brick_instance = BrickInstance(0, brick_shape, 0, brick_transform)
        upright_snaps = brick_instance.get_upright_snaps()
        return bool(len(upright_snaps))
    
    else:
        # there must be a connection to a new brick
        new_edges = numpy.where(goal_assembly['edges'][0] == new_brick)[0]
        connected_bricks = goal_assembly['edges'][1, new_edges]
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

def node_removable_collision_free(
    remove_brick,
    remove_brick_shape_name,
    existing_bricks,
    collision_map,
    start_assembly,
    maintain_connectivity=False,
):
    if maintain_connectivity and len(existing_bricks) > 1:
        # this is not efficient, but I don't care yet
        edges = start_assembly['edges']
        connectivity = {}
        for i in range(edges.shape[1]):
            if edges[0,i] == 0:
                continue
            
            if edges[0,i] not in existing_bricks:
                continue
            
            if edges[0,i] == remove_brick:
                continue
            
            if edges[1,i] in existing_bricks:
                connectivity.setdefault(edges[0,i], set())
                if edges[1,i] == remove_brick:
                    continue
                connectivity[edges[0,i]].add(edges[1,i])
        
        connected = set()
        try:
            frontier = [list(connectivity.keys())[0]]
        except:
            raise FrontierError
        while frontier:
            node = frontier.pop()
            if node in connected:
                continue
            
            connected.add(node)
            frontier.extend(connectivity[node])
        
        if len((existing_bricks - {remove_brick}) - connected):
            return False
    
    for axis, polarity, snap_group in collision_map[remove_brick]:
        colliding_bricks = collision_map[remove_brick][
            axis, polarity, snap_group]
        colliding_bricks = frozenset(colliding_bricks)
        if not len(colliding_bricks & existing_bricks):
            break
    else:
        return False
    
    return True

class Roadmap:
    def __init__(self, env, goal_assembly, shape_ids, color_ids):
        self.env = env
        self.goal_assembly = goal_assembly
        self.goal_state = frozenset(numpy.where(goal_assembly['shape'])[0])
        # removing nodes, they are not used for anything
        #self.nodes = set()
        #self.nodes.add(self.goal_state)
        self.false_positive_labels = set()
        self.edges = {}
        self.successors = {}
        self.env_states = {}
        self.brick_shape_to_shape_id = shape_ids
        self.shape_id_to_brick_shape = {
            value:key for key, value in self.brick_shape_to_shape_id.items()}
        self.color_ids = color_ids
        
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_assembly(
            self.goal_assembly, self.brick_shape_to_shape_id, self.color_ids)
        self.goal_collision_map = build_collision_map(goal_scene)
    
    def get_observation_action_seq(self, path):
        observation_seq = []
        action_seq = []
        for a,b in zip(path[:-1], path[1:]):
            observation_seq.extend(self.edges[a,b]['observation_seq'][:-1])
            action_seq.extend(self.edges[a,b]['action_seq'])
        
        observation_seq.append(self.edges[a,b]['observation_seq'][-1])
        
        return observation_seq, action_seq

class RoadmapPlanner:
    def __init__(self, roadmap, start_env_state):
        
        # initialize
        self.roadmap = roadmap
        self.edge_checker = EdgeChecker(self, self.roadmap)
        
        # set the env to the start state
        observation = self.roadmap.env.set_state(start_env_state)
        self.start_assembly = observation['table_assembly']
        
        # compute a matching
        matching, offset = match_assemblies(
            self.start_assembly,
            roadmap.goal_assembly,
            roadmap.shape_id_to_brick_shape,
        )
        self.wip_to_goal, self.goal_to_wip, fp, fn = match_lookup(
            matching, self.start_assembly, roadmap.goal_assembly)
        
        self.false_positive_lookup = self.make_false_positive_labels(fp)
        
        self.start_state = (
            frozenset(self.goal_to_wip.keys()) |
            frozenset(self.false_positive_lookup.keys())
        )
        assert self.start_state not in self.roadmap.env_states
        self.roadmap.env_states[self.start_state] = start_env_state
        
        self.visits = {}
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_assembly(
            self.start_assembly,
            self.roadmap.brick_shape_to_shape_id,
            self.roadmap.color_ids,
        )
        self.start_collision_map = build_collision_map(start_scene)
    
    def make_false_positive_labels(self, fp):
        max_fp = max(self.roadmap.false_positive_labels, default=0)
        new_labels = {f + max_fp:f for f in fp}
        self.roadmap.false_positive_labels |= new_labels.keys()
        
        return new_labels
    
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
        found_path = False
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            # plan a path
            candidate_path, goal_found = self.plan_collision_free()
            if goal_found:
                (candidate_path,
                 viewpoint_changes,
                 goal_feasible) = self.edge_checker.check_path(candidate_path)
                if goal_feasible:
                    found_path = True
                    q = -0.05 * viewpoint_changes
                    cost = -q
                    if cost < max_cost:
                        return found_path
                else:
                    q = -1
            else:
                q = -1
            
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
        successors = set()
        
        # if there are any false positives, remove them first
        if false_positives:
            for false_positive in false_positives:
                false_positive_shape = (
                    self.start_assembly['shape'][false_positive])
                if node_removable_collision_free(
                    false_positive,
                    self.roadmap.shape_id_to_brick_shape[false_positive_shape],
                    state,
                    self.start_collision_map,
                    self.start_assembly,
                    maintain_connectivity=True,
                ):
                    successor = state - frozenset((false_positive,))
                    successors.add(successor)
        
        # if there are no false positives, but false negatives, add them
        elif false_negatives:
            for false_negative in false_negatives:
                # check if false_negative can be added
                false_negative_shape = self.roadmap.goal_assembly['shape'][
                    false_negative]
                if node_connected_collision_free(
                    false_negative,
                    self.roadmap.shape_id_to_brick_shape[false_negative_shape],
                    state,
                    self.roadmap.goal_state,
                    self.roadmap.goal_collision_map,
                    self.roadmap.goal_assembly,
                ):
                    successor = state | frozenset((false_negative,))
                    successors.add(successor)
        
        # add the succesors to the roadmap
        self.roadmap.successors[state] = successors
        
        # add the edges and successors to the roadmap
        for successor in successors:
            self.roadmap.edges[state, successor] = {
                'collision_free':True,
                'viewpoint_changes':None,
                'feasible':None,
                'action_seq':[],
                'observation_seq':[],
            }
            #self.roadmap.nodes.add(successor)
    
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
        successors = list(self.roadmap.successors[state])
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
    def __init__(self, planner, roadmap):
        self.planner = planner
        self.roadmap = roadmap
    
    def check_path(self, candidate_path):
        successful_path = [candidate_path[0]]
        goal_to_wip = copy.deepcopy(self.planner.goal_to_wip)
        total_viewpoint_changes = 0
        
        silent = True
        iterate = zip(candidate_path[:-1], candidate_path[1:])
        if not silent:
            iterate = tqdm.tqdm(iterate, total=len(candidate_path)-1)
        for a, b in iterate:
            edge = self.roadmap.edges[a,b]
            if b not in self.roadmap.env_states:
                observation_seq, action_seq = self.check_edge(a, b, goal_to_wip)
                self.roadmap.edges[a,b]['observation_seq'] = observation_seq
                self.roadmap.edges[a,b]['action_seq'] = action_seq
                if action_seq is None:
                    self.roadmap.successors[a].remove(b)
                    del self.roadmap.edges[a,b]
                    return successful_path, total_viewpoint_changes, False
                successful_path.append(b)
                last_state = self.roadmap.env.get_state()
                self.roadmap.env_states[b] = last_state
                viewpoint_changes = len([
                    a for a in action_seq if (
                        a['table_viewpoint'] != 0 or
                        a['hand_viewpoint'] != 0
                    )
                ])
                edge['viewpoint_changes'] = viewpoint_changes
                edge['feasible'] = True
                total_viewpoint_changes += viewpoint_changes
            else:
                successful_path.append(b)
        
        return successful_path, total_viewpoint_changes, True
    
    def check_edge(self, a, b, goal_to_wip):
        env_state = self.roadmap.env_states[a]
        observation = self.roadmap.env.set_state(env_state)
        if len(goal_to_wip):
            next_instance = max(goal_to_wip.values())+1
        else:
            next_instance = 1
        
        assert abs(len(a) - len(b)) == 1
        
        if len(a) < len(b):
            # add a brick
            if len(a) == 0:
                # add the first brick
                instance = next(iter(b))
                observation_seq, action_seq = plan_add_first_brick(
                    self.roadmap.env,
                    self.roadmap.goal_assembly,
                    instance,
                    observation,
                    goal_to_wip,
                    self.roadmap.shape_id_to_brick_shape,
                    debug=False,
                )
                goal_to_wip[instance] = next_instance
                return observation_seq, action_seq
            
            else:
                # add the nth brick
                instance = next(iter(b-a))
                observation_seq, action_seq = plan_add_nth_brick(
                    self.roadmap.env,
                    self.roadmap.goal_assembly,
                    instance,
                    observation,
                    goal_to_wip,
                    self.roadmap.shape_id_to_brick_shape,
                    debug=False,
                )
                goal_to_wip[instance] = next_instance
                return observation_seq, action_seq
        
        elif len(b) < len(a):
            # remove a brick
            instance = next(iter(a-b))
            return plan_remove_nth_brick(
                self.roadmap.env,
                self.roadmap.goal_assembly,
                instance,
                observation,
                #goal_to_wip,
                self.planner.false_positive_lookup,
                self.roadmap.shape_id_to_brick_shape,
            )

def upper_confidence_bound(q, n_action, n_state, c=2**0.5):
    return q + c * (math.log(n_state+1)/(n_action+1))**0.5
