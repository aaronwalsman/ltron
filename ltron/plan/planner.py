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
    goal_membership,
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

        unadded_bricks = goal_membership - updated_bricks
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
            frontier = [next(iter(connectivity.keys()))]
        except StopIteration:
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
    '''
    Stores the data necessary for the RoadmapPlanner
    Planning is done over membership states, which are frozensets of indices
    corresponding to brick instances that exist in the scene.  The indices of
    brick instances is drawn from the indices of the goal_assembly, with
    extra instances having successive indices after that.
    '''
    def __init__(self, env, goal_assembly, shape_ids, color_ids):
        self.env = env
        self.goal_assembly = goal_assembly
        self.goal_membership = frozenset(numpy.where(goal_assembly['shape'])[0])
        self.false_positive_labels = set()
        # edges : (path, path) -> 
        #self.edges = {}
        # successors : (path) -> 
        #self.successors = {}
        self.paths = {}
        #self.env_states = {}
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
        #for a,b in zip(path[:-1], path[1:]):
        #    observation_seq.extend(self.edges[a,b]['observation_seq'][:-1])
        #    action_seq.extend(self.edges[a,b]['action_seq'])
        for i in range(len(path)):
            sub_path = path[:i+1]
            observation_seq.extend(self.paths[sub_path]['observation_seq'][:-1])
            action_seq.extend(self.paths[sub_path]['action_seq'])
        
        #try:
        #    observation_seq.append(self.edges[a,b]['observation_seq'][-1])
        #except:
        #    import pdb
        #    pdb.set_trace()
        
        observation_seq.append(self.paths[path]['observation_seq'][-1])
        
        return observation_seq, action_seq

class RoadmapPlanner:
    def __init__(self, roadmap, start_env_state, failure_penalty=-3):
        
        # initialize
        self.roadmap = roadmap
        self.edge_checker = EdgeChecker(self, self.roadmap)
        self.failure_penalty = failure_penalty
        
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
        
        self.start_membership = (
            frozenset(self.goal_to_wip.keys()) |
            frozenset(self.false_positive_lookup.keys())
        )
        #assert self.start_membership not in self.roadmap.env_states
        #self.roadmap.env_states[self.start_membership] = start_env_state
        #assert (self.start_membership,) not in self.roadmap.paths
        #PROBABLY_BOOT_STRAP_THIS_IN
        
        #self.visits = {}
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_assembly(
            self.start_assembly,
            self.roadmap.brick_shape_to_shape_id,
            self.roadmap.color_ids,
        )
        self.start_collision_map = build_collision_map(start_scene)
        
        #self.expand_successors((self.start_membership,))
        first_path = (self.start_membership,)
        self.initialize_path(first_path)
        self.roadmap.paths[first_path]['env_state'] = start_env_state
    
    def make_false_positive_labels(self, fp):
        max_fp = max(self.roadmap.false_positive_labels, default=0)
        new_labels = {f + max_fp:f for f in fp}
        self.roadmap.false_positive_labels |= new_labels.keys()
        
        return new_labels
    
    #def greedy_path(self):
    #    current_path = (self.start_membership,)
    #    total_q = 0
    #    # THIS IS TOTALLY WRONG BY THE WAY... MAYBE THE HIGHEST Q VALUE ISN'T
    #    # EVEN FEASIBLE BECAUSE THE FAILURE PENALTY ISN'T HIGH ENOUGH
    #    # MAKING FEASIBILITY PART OF THE Q FUNCTION IS JUST BAD
    #    while current_path[-1] != self.roadmap.goal_membership:
    #        #successors = self.roadmap.successors[current_path]
    #        #edges = [(current_path, successor) for successor in successors]
    #        best_q, current_path = max(zip(
    #            (self.visits[edge]['q'] for edge in edges),
    #            successors))
    #        total_q += best_q
    #    
    #    return path, total_q
    
    def plan(self, max_cost=float('inf'), timeout=float('inf')):
        print('plan')
        t_start = time.time()
        found_path = False
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                #raise PathNotFoundError
                break
            
            # plan a path
            candidate_path, goal_found = self.plan_collision_free()
            print(goal_found)
            if goal_found:
                (candidate_path,
                 viewpoint_changes,
                 goal_feasible) = self.edge_checker.check_path(candidate_path)
                cost = viewpoint_changes
                print('cost:', cost)
                q = -cost
                if goal_feasible:
                    found_path = True
                    if cost <= max_cost:
                        return found_path
                else:
                    q += self.failure_penalty
            else:
                q = self.failure_penalty
            
            self.update_path_visits(candidate_path, q)
        
        return found_path
    
    def plan_collision_free(self):
        print('plan_collision_free')
        
        current_path = (self.start_membership,)
        #path = [self.start_membership]
        goal_found = True
        
        while current_path[-1] != self.roadmap.goal_membership:
            print('pcf step', current_path[-1])
            # add the current state to the road map nodes
            #if current_path not in self.roadmap.paths:
            if self.roadmap.paths[current_path]['successors'] is None:
                self.expand_successors(current_path)
            # add the current state to the visit statistics
            #if current_path not in self.visits:
            #    self.expand_visits(current_path)
            
            # sample a high level node based on visit counts
            next_membership = self.sample_collision_free(current_path)
            if next_membership is None:
                goal_found = False
                break
            else:
                #path.append(current_state)
                current_path = current_path + (next_membership,)
        
        return current_path, goal_found
    
    def expand_successors(self, path):
        print('expand', path)
        # compute false positives and false negatives
        false_positives = path[-1] - self.roadmap.goal_membership
        false_negatives = self.roadmap.goal_membership - path[-1]
        
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
                    path[-1],
                    self.start_collision_map,
                    self.start_assembly,
                    maintain_connectivity=True,
                ):
                    successor = path[-1] - frozenset((false_positive,))
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
                    path,
                    self.roadmap.goal_membership,
                    self.roadmap.goal_collision_map,
                    self.roadmap.goal_assembly,
                ):
                    successor = path[-1] | frozenset((false_negative,))
                    successors.add(successor)
        
        # add the succesors to the roadmap
        self.roadmap.paths[path]['successors'] = successors
        
        # add the successor paths roadmap
        for successor in successors:
            self.initialize_path(path + (successor,))
    
    def initialize_path(self, path):
        self.roadmap.paths[path] = {
            'collision_free':True,
            'viewpoint_changes':None,
            'env_state':None,
            #'feasible':None,
            'action_seq':[],
            'observation_seq':[],
            'successors':None,
            'n':0,
            'w':0,
            'q':0,
        }
    
    #def expand_visits(self, state):
    #    self.visits[state] = {'n':1}
    #    for successor in self.roadmap.successors[state]:
    #        self.visits[state, successor] = {'n':0, 'q':0, 'w':0}
    
    def update_path_visits(self, path, q):
        for i in range(len(path)):
            sub_path = path[:i+1]
            sub_path_data = self.roadmap.paths[sub_path]
            sub_path_data['w'] += q
            sub_path_data['n'] += 1
            sub_path_data['q'] = sub_path_data['w'] / sub_path_data['n']
        #for a, b in zip(path[:-1], path[1:]):
        #    self.visits[a,b]['w'] += q
        #    self.visits[a,b]['n'] += 1
        #    self.visits[a,b]['q'] = self.visits[a,b]['w']/self.visits[a,b]['n']
    
    def sample_collision_free(self, path):
        print('sample_collision_free')
        #successors = list(self.roadmap.successors[path])
        successors = list(self.roadmap.paths[path]['successors'])
        if len(successors):
            u = [
                ucb(
                    #q=self.visits[path, successor]['q'],
                    #n_action=self.visits[path, successor]['n'],
                    #n_state=self.visits[path]['n'],
                    q=self.roadmap.paths[path + (successor,)]['q'],
                    n_action=self.roadmap.paths[path + (successor,)]['n'],
                    n_state=self.roadmap.paths[path]['n'],
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
        print('checking path')
        goal_to_wip = copy.deepcopy(self.planner.goal_to_wip)
        total_viewpoint_changes = 0
        
        silent = True
        #iterate = zip(candidate_path[:-1], candidate_path[1:])
        #if not silent:
        #    iterate = tqdm.tqdm(iterate, total=len(candidate_path)-1)
        #for a, b in iterate:
        iterate = range(1, len(candidate_path))
        if not silent:
            iterate = tqdm.tqdm(iterate)
        for i in iterate:
            a_path = candidate_path[:i]
            b_path = candidate_path[:i+1]
            print('   ', a_path, '->', b_path)
            a_path_data = self.roadmap.paths[a_path]
            b_path_data = self.roadmap.paths[b_path]
            #edge = self.roadmap.edges[a,b]
            if b_path_data['env_state'] is None:
                #observation_seq, action_seq = self.check_edge(
                #    a, b, goal_to_wip)
                #self.roadmap.edges[a,b]['observation_seq'] = observation_seq
                #self.roadmap.edges[a,b]['action_seq'] = action_seq
                observation_seq, action_seq = self.check_edge(
                    b_path, goal_to_wip)
                b_path_data['observation_seq'] = observation_seq
                b_path_data['action_seq'] = action_seq
                if action_seq is None:
                    b = b_path[-1]
                    self.roadmap.paths[a_path]['successors'].remove(b)
                    for j in range(i, len(candidate_path)):
                        post_feasible_path = candidate_path[:j+1]
                        del(self.roadmap.paths[post_feasible_path])
                    #self.roadmap.successors[a].remove(b)
                    #del self.roadmap.edges[a,b]
                    #print('fail')
                    
                    return a_path, total_viewpoint_changes, False
                
                last_state = self.roadmap.env.get_state()
                #self.roadmap.env_states[b] = last_state
                b_path_data['env_state'] = last_state
                viewpoint_changes = len([
                    a for a in action_seq if (
                        a['table_viewpoint'] != 0 or
                        a['hand_viewpoint'] != 0
                    )
                ])
                b_path_data['viewpoint_changes'] = viewpoint_changes
                #edge['viewpoint_changes'] = viewpoint_changes
                #edge['feasible'] = True
                total_viewpoint_changes += viewpoint_changes
                print('    step ok', candidate_path[:i+1])
            #else:
                #successful_path.append(b)
        
        print('success')
        #return successful_path, total_viewpoint_changes, True
        return candidate_path, total_viewpoint_changes, True
    
    #def check_edge(self, a, b, goal_to_wip):
        #env_state = self.roadmap.env_states[a]
    def check_edge(self, path, goal_to_wip):
        prev_path = path[:-1]
        env_state = self.roadmap.paths[prev_path]['env_state']
        observation = self.roadmap.env.set_state(env_state)
        if len(goal_to_wip):
            next_instance = max(goal_to_wip.values())+1
        else:
            next_instance = 1
        
        a, b = path[-2:]
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

def ucb(q, n_action, n_state, c=2**0.5):
    return q + c * (math.log(n_state+1)/(n_action+1))**0.5

def pucb(q, p, n_action, n_state, c=2**0.5):
    return q + c * p * (n_state ** 0.5) / (n_action+1)

def rpo(q, p, n_actions, n_state, c=2**0.5):
    '''
    https://arxiv.org/pdf/2007.12509.pdf
    '''
    l = c * n_state**0.5 / (n_actions + n_state)
    a = something_somehow
    pi = l * p / (a-q)
