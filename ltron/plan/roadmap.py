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
        self.paths = {}
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
        for i in range(len(path)):
            sub_path = path[:i+1]
            observation_seq.extend(self.paths[sub_path]['observation_seq'][:-1])
            action_seq.extend(self.paths[sub_path]['action_seq'])
        
        observation_seq.append(self.paths[path]['observation_seq'][-1])
        
        return observation_seq, action_seq

def get_visible_snaps(observation, region):
    v = set()
    for polarity in 'pos', 'neg':
        snaps = observation['%s_%s_snap_render'%(region,polarity)].reshape(-1,2)
        nonzero = numpy.where(snaps[:,0])
        nonzero_snaps = snaps[nonzero]
        v = v | {tuple(snap) for snap in nonzero_snaps}
    
    return v

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
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_assembly(
            self.start_assembly,
            self.roadmap.brick_shape_to_shape_id,
            self.roadmap.color_ids,
        )
        self.start_collision_map = build_collision_map(start_scene)
        
        first_path = (self.start_membership,)
        self.initialize_path(first_path)
        self.roadmap.paths[first_path]['env_state'] = start_env_state
    
    def make_false_positive_labels(self, fp):
        max_fp = max(self.roadmap.false_positive_labels, default=0)
        new_labels = {f + max_fp:f for f in fp}
        self.roadmap.false_positive_labels |= new_labels.keys()
        
        return new_labels
    
    def best_path(self):
        pass
    
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
        print('plan start')
        t_start = time.time()
        found_path = False
        while True:
            print('new run')
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                #raise PathNotFoundError
                break
            
            # plan a path
            candidate_path, goal_found = self.plan_collision_free()
            if goal_found:
                #(candidate_path,
                 #viewpoint_changes,
                # goal_feasible) = self.edge_checker.check_path(
                candidate_path, goal_feasible = self.edge_checker.check_path(
                    candidate_path, max_cost)
                #cost = 0
                #for i in range(1, len(candidate_path)+1):
                #    sub_path = candidate_path[:i]
                #    cost += self.roadmap.paths[sub_path]['viewpoint_changes']
                cost = self.roadmap.paths[candidate_path][
                    'total_viewpoint_changes']
                
                print('cost:', cost, goal_feasible)
                q = -cost
                if goal_feasible:
                    found_path = True
                    if cost <= max_cost:
                        return found_path
                else:
                    q += self.failure_penalty
            else:
                q = self.failure_penalty
            
            #self.update_path_visits(candidate_path, q)
        
        return found_path
    
    def plan_collision_free(self):
        
        current_path = (self.start_membership,)
        goal_found = True
        
        while current_path[-1] != self.roadmap.goal_membership:
            # add the current state to the road map nodes
            if self.roadmap.paths[current_path]['successors'] is None:
                self.expand_successors(current_path)
            
            # sample a high level node based on visit counts
            next_membership = self.sample_collision_free(current_path)
            if next_membership is None:
                goal_found = False
                break
            else:
                current_path = current_path + (next_membership,)
        
        return current_path, goal_found
    
    def expand_successors(self, path):
        # compute false positives and false negatives
        false_positives = path[-1] - self.roadmap.goal_membership
        false_negatives = self.roadmap.goal_membership - path[-1]
        
        # compute possible successors
        successors = {}
        
        # if there are any false positives, remove them first
        if false_positives:
            for false_positive in false_positives:
                false_positive_shape = (
                    self.start_assembly['shape'][false_positive])
                remove_action = self.collision_free_remove_brick_action(
                    false_positive,
                    self.roadmap.shape_id_to_brick_shape[false_positive_shape],
                    path[-1],
                    maintain_connectivity=True,
                )
                if remove_action is not None:
                    successor = path[-1] - frozenset((false_positive,))
                    successors[remove_action] = successor
        
        # if there are no false positives, but false negatives, add them
        elif false_negatives:
            for false_negative in false_negatives:
                # check if false_negative can be added
                false_negative_shape = self.roadmap.goal_assembly['shape'][
                    false_negative]
                add_action = self.collision_free_add_brick_action(
                    false_negative,
                    self.roadmap.shape_id_to_brick_shape[false_negative_shape],
                    path[-1],
                )
                if add_action is not None:
                    successor = path[-1] | frozenset((false_negative,))
                    successors[add_action] = successor
        
        # add the succesors to the roadmap
        self.roadmap.paths[path]['successors'] = successors
        
        # add the successor paths roadmap
        for action, successor in successors.items():
            self.initialize_path(path + (successor,))
    
    def initialize_path(self, path):
        self.roadmap.paths[path] = {
            'viewpoint_changes':0, #None,
            'total_viewpoint_changes':0,
            'env_state':None,
            'action_seq':[],
            'observation_seq':[],
            'connected_to_goal' : False,
            'successors':None,
            'visible_snaps':None,
            #'n':0,
            #'w':0,
            #'q':0,
        }
    
    '''
    def update_path_visits(self, path, q):
        for i in range(len(path)):
            sub_path = path[:i+1]
            print('updating:', sub_path)
            sub_path_data = self.roadmap.paths[sub_path]
            sub_path_data['w'] += q
            sub_path_data['n'] += 1
            sub_path_data['q'] = sub_path_data['w'] / sub_path_data['n']
    '''
    
    def sample_collision_free(self, path):
        successors = list(self.roadmap.paths[path]['successors'].values())
        if len(successors):
            '''
            u = [ucb(
                    q=self.roadmap.paths[path + (successor,)]['q'],
                    n_action=self.roadmap.paths[path + (successor,)]['n'],
                    n_state=self.roadmap.paths[path]['n'],
                )
                for successor in successors]
            best_u, best_successor = max(zip(u, successors))
            '''
            
            c = [self.roadmap.paths[path + (successor,)]['viewpoint_changes']
                for successor in successors]
            
            # VISIBLE THINGS GOES HERE
            
            best_c, best_successor = min(zip(c, successors))
            
            print(c, best_c)
            
            return best_successor
        else:
            return None
    
    def collision_free_add_brick_action(
        self,
        new_brick,
        new_brick_shape_name,
        existing_bricks,
    ):
        
        collision_map = self.roadmap.goal_collision_map
        goal_assembly = self.roadmap.goal_assembly
        
        if not len(existing_bricks):
            brick_shape = BrickShape(new_brick_shape_name)
            brick_transform = goal_assembly['pose'][new_brick]
            brick_instance = BrickInstance(0, brick_shape, 0, brick_transform)
            upright_snaps = brick_instance.get_upright_snaps()
            #return bool(len(upright_snaps))
            if len(upright_snaps):
                return AddFirstBrick(new_brick, upright_snaps)
            else:
                return None
        
        else:
            # there must be a connection to a new brick
            new_edges = numpy.where(goal_assembly['edges'][0] == new_brick)[0]
            #connected_bricks = goal_assembly['edges'][1, new_edges]
            #new_connected_snaps = goal_assembly['edges'][2, new_edges]
            new_edge_data = goal_assembly['edges'][:,new_edges]
            new_connected_existing_snaps = [
                sa for ia, ib, sa, sb in new_edge_data.T
                if ib in existing_bricks
            ]
            #connected_bricks = frozenset(connected_bricks)
            #connected_existing_bricks = connected_bricks & existing_bricks
            #if not len(connected_existing_bricks):
            if not len(new_connected_existing_snaps):
                #return False
                return None
            
            updated_bricks = existing_bricks | frozenset((new_brick,))
            
            # make sure each brick that hasn't been added yet will still have
            # a way to get connected without collision after this brick has been
            # added
            unadded_bricks = self.roadmap.goal_membership - updated_bricks
            for unadded_brick in unadded_bricks:
                for axis, polarity, snap_group in collision_map[unadded_brick]:
                    colliding_bricks = collision_map[unadded_brick][
                        axis, polarity, snap_group]
                    colliding_bricks = frozenset(colliding_bricks)
                    if not len(colliding_bricks & updated_bricks):
                        break
                
                else:
                    return None
            
            return AddNthBrick(new_brick, new_connected_existing_snaps)

    def collision_free_remove_brick_action(
        self,
        remove_brick,
        remove_brick_shape_name,
        existing_bricks,
        maintain_connectivity=False,
    ):
        
        collision_map = self.start_collision_map
        start_assembly = self.start_assembly
        
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
                return None
        
        remove_snaps = []
        for axis, polarity, snap_group in collision_map[remove_brick]:
            colliding_bricks = collision_map[remove_brick][
                axis, polarity, snap_group]
            colliding_bricks = frozenset(colliding_bricks)
            if not len(colliding_bricks & existing_bricks):
                remove_snaps.extend(snap_group)
        
        if len(remove_snaps):
            return RemoveNthBrick(remove_brick, remove_snaps)
        else:
            return None

class EdgeChecker:
    def __init__(self, planner, roadmap):
        self.planner = planner
        self.roadmap = roadmap
    
    def check_path(self, candidate_path, max_cost):
        goal_to_wip = copy.deepcopy(self.planner.goal_to_wip)
        #total_viewpoint_changes = 0
        
        silent = True
        iterate = range(1, len(candidate_path))
        if not silent:
            iterate = tqdm.tqdm(iterate)
        for i in iterate:
            a_path = candidate_path[:i]
            b_path = candidate_path[:i+1]
            a_path_data = self.roadmap.paths[a_path]
            b_path_data = self.roadmap.paths[b_path]
            print(i)
            if b_path_data['env_state'] is None:
                print('  update')
                
                observation_seq, action_seq = self.check_edge(
                    b_path, goal_to_wip)
                b_path_data['observation_seq'] = observation_seq
                b_path_data['action_seq'] = action_seq
                
                # update the env_state
                b_path_data['env_state'] = self.roadmap.env.get_state()
                viewpoint_changes = len([
                    a for a in action_seq if (
                        a['table_viewpoint'] != 0 or
                        a['hand_viewpoint'] != 0
                    )
                ])
                b_path_data['viewpoint_changes'] = viewpoint_changes
                v = a_path_data['total_viewpoint_changes'] + viewpoint_changes
                b_path_data['total_viewpoint_changes'] = v
                print(' ', viewpoint_changes, v)
                
                # if this action is not feasible, remove b_path from a_path's
                # succcessors and delete all subsequent paths
                #if action_seq is None or total_viewpoint_changes > max_cost:
                if action_seq is None or v > max_cost:
                    b = b_path[-1]
                    successors = self.roadmap.paths[a_path]['successors']
                    action = next(a for a, s in successors.items() if s == b)
                    #successors.remove(action)
                    del(successors[action])
                    
                    for j in range(i, len(candidate_path)):
                        post_feasible_path = candidate_path[:j+1]
                        del(self.roadmap.paths[post_feasible_path])
                    
                    return a_path, False #total_viewpoint_changes, False
                #total_viewpoint_changes += viewpoint_changes
            
            a, b = b_path[-2:]
            if len(a) < len(b):
                if len(goal_to_wip):
                    next_instance = max(goal_to_wip.values()) + 1
                else:
                    next_instance = 1
                instance = next(iter(b-a))
                goal_to_wip[instance] = next_instance
        
        import pdb
        pdb.set_trace()
        
        return candidate_path, True #total_viewpoint_changes, True
    
    def check_edge(self, path, goal_to_wip):
        prev_path = path[:-1]
        env_state = self.roadmap.paths[prev_path]['env_state']
        observation = self.roadmap.env.set_state(env_state)
        
        v = get_visible_snaps(observation, 'table')
        self.roadmap.paths[path]['visible_snaps'] = v
        
        #if len(goal_to_wip):
        #    next_instance = max(goal_to_wip.values())+1
        #else:
        #    next_instance = 1
        
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
                    debug=True,
                )
                #goal_to_wip[instance] = next_instance
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
                    debug=True,
                )
                
                #goal_to_wip[instance] = next_instance
                return observation_seq, action_seq
        
        elif len(b) < len(a):
            # remove a brick
            instance = next(iter(a-b))
            return plan_remove_nth_brick(
                self.roadmap.env,
                self.roadmap.goal_assembly,
                instance,
                observation,
                self.planner.false_positive_lookup,
                self.roadmap.shape_id_to_brick_shape,
                debug=True,
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

class HighLevelAction:
    def __init__(self, instance, snaps):
        self.instance = instance
        self.snaps = frozenset(snaps)
    
    def __getitem__(self, i):
        if i == 0:
            return self.instance
        elif i == 1:
            return self.snaps
        else:
            raise IndexError
    
    def __hash__(self):
        return hash(tuple(self))
    
    def __eq__(self, other):
        return tuple(self) == tuple(other)
    
    def __str__(self):
        return str(tuple(self))

class RemoveNthBrick(HighLevelAction):
    pass

class AddFirstBrick(HighLevelAction):
    pass

class AddNthBrick(HighLevelAction):
    pass
