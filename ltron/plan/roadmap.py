import random
import time
import math
import copy
from bisect import insort

import tqdm

import numpy

from ltron.exceptions import LtronException
#from ltron.bricks.brick_scene import BrickScene
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

class PlannerTimeoutError(PlanningException):
    pass

class FrontierError(PlanningException):
    pass

#def get_visible_snaps(observation, region):
#    v = set()
#    for polarity in 'pos', 'neg':
#        snaps = observation['%s_%s_snap_render'%(region,polarity)].reshape(-1,2)
#        nonzero = numpy.where(snaps[:,0])
#        nonzero_snaps = snaps[nonzero]
#        v = v | {tuple(snap) for snap in nonzero_snaps}
#    
#    return v

class Roadmap:
    '''
    High level planning is done over a path of nodes.  Each node represents
    a set of bricks that exist in the scene at a particular time.  The
    membership is stored as a frozenset, with integers representing brick
    instance indices under the labelling provided by the goal assembly.
    Instances from the start assembly that exist in the goal assembly as well
    are transferred to the goal's indexing scheme using a matching.  Instances
    from the start assembly that do not exist in the goal assembly are assigned
    new labels after the largest index in the goal assembly.
    '''
    def __init__(
        self,
        env,
        start_env_state,
        start_collision_map,
        goal_assembly,
        goal_collision_map,
        shape_ids,
        color_ids,
        target_steps_per_view_change=4,
        split_cursor_actions=False,
        allow_snap_flip=False,
    ):
        
        # store arguments
        self.env = env
        self.start_env_state = start_env_state
        self.start_assembly = start_env_state['table_scene']
        self.goal_assembly = goal_assembly
        self.goal_membership = frozenset(numpy.where(goal_assembly['shape'])[0])
        self.brick_shape_to_shape_id = shape_ids
        self.shape_id_to_brick_shape = {
            value:key for key, value in self.brick_shape_to_shape_id.items()}
        self.color_ids = color_ids
        self.target_steps_per_view_change = target_steps_per_view_change
        self.split_cursor_actions = split_cursor_actions
        self.allow_snap_flip = allow_snap_flip
        
        self.start_collision_map = start_collision_map
        self.goal_collision_map = goal_collision_map
        
        # initialize paths
        self.paths = {}
        
        # compute a matching from start to goal
        matching, offset = match_assemblies(
            self.start_assembly,
            self.goal_assembly,
            self.shape_id_to_brick_shape,
        )
        self.wip_to_goal, self.goal_to_wip, fp, fn = match_lookup(
            matching, self.start_assembly, self.goal_assembly)
        
        # make membership ids for the false-positive bricks
        first_fp = max(self.goal_membership, default=0)+1
        self.false_positive_goal_ids = {
            i + first_fp:f for i, f in enumerate(fp)}
        self.false_positive_start_ids = {
            j:i for i,j in self.false_positive_goal_ids.items()}
        
        # compute the start assembly and membership
        observation = self.env.set_state(start_env_state)
        self.start_membership = (
            frozenset(self.goal_to_wip.keys()) |
            frozenset(self.false_positive_goal_ids.keys())
        )
        
        # initialize the first path
        first_path = (self.start_membership,)
        self.initialize_path(first_path)
        self.paths[first_path]['env_state'] = start_env_state
        self.paths[first_path]['evaluated'] = True
        
        '''
        # build start collision map
        temp_scene = BrickScene(renderable=True, track_snaps=True)
        temp_scene.import_assembly(
            self.start_assembly, self.brick_shape_to_shape_id, self.color_ids)
        self.start_collision_map = build_collision_map(temp_scene)
        
        # build goal collision map
        temp_scene.clear_instances()
        temp_scene.import_assembly(
            self.goal_assembly, self.brick_shape_to_shape_id, self.color_ids)
        self.goal_collision_map = build_collision_map(temp_scene)
        '''
    
    #def get_observation_action_seq(self, path):
    #    observation_seq = []
    #    action_seq = []
    #    for i in range(len(path)):
    #        sub_path = path[:i+1]
    #        observation_seq.extend(self.paths[sub_path]['observation_seq'][:-1])
    #        action_seq.extend(self.paths[sub_path]['action_seq'])
    #    
    #    observation_seq.append(self.paths[path]['observation_seq'][-1])
    #    
    #    return observation_seq, action_seq
    
    def cost(self, path):
        steps = len(path) - 1
        view_changes = self.paths[path]['total_view_changes']
        return view_changes - steps / self.target_steps_per_view_change
    
    def plan(self, timeout=float('inf')):
        t_start = time.time()
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PlannerTimeoutError
            
            # plan a collision free path
            path, next_best_cost = self.plan_collision_free()
            
            # use the edge checker to evaluate the edges until the goal is
            # reached or until the cost has become greater than next_best_cost
            good_path = self.check_path(path, next_best_cost)
            if good_path:
                return path
    
    def get_observation_action_reward_seq(
        self,
        path,
        include_last_observation=True
    ):
        observation_seq = []
        action_seq = []
        reward_seq = []
        for i in range(len(path)):
            sub_path = path[:i+1]
            path_data = self.paths[sub_path]
            observation_seq.extend(path_data['observation_seq'][:-1])
            action_seq.extend(path_data['action_seq'])
            reward_seq.extend(path_data['reward_seq'])
        
        if include_last_observation:
            observation_seq.append(path_data['observation_seq'][-1])
        
        return observation_seq, action_seq, reward_seq
    
    def plan_collision_free(self):
        
        while True:
            
            # find the best evaluated starting path so far
            path_costs = sorted([
                (self.cost(path), path)
                for path, path_data in self.paths.items()
                if path_data['evaluated'] and
                not path_data['all_successors_evaluated']
            ])
            if len(path_costs) == 0:
                raise PathNotFoundError
            
            cost, path = path_costs[0]
            if len(path_costs) > 1:
                next_best_cost, _ = path_costs[1]
            else:
                next_best_cost = 0
            
            starting_path = path
            
            # find a path from this starting path to the goal through
            # unevaluated nodes
            #print(path[-1])
            while path[-1] != self.goal_membership:
                
                # add the current state to the road map paths
                if self.paths[path]['successors'] is None:
                    self.expand_successors(path)
                
                # if this path has no successors, prune it and either
                # step backward or break to find a new starting path
                if not len(self.paths[path]['successors']):
                    del(self.paths[path])
                    previous_path = path[:-1]
                    leaf = path[-1]
                    self.paths[previous_path]['successors'].remove(leaf)
                    
                    if path != starting_path:
                        path = previous_path
                    else:
                        break
                
                # find all unevaluated successors
                unevaluated_successors = [
                    s for s in self.paths[path]['successors']
                    if not self.paths[path + (s,)]['evaluated']
                ]
                
                # if there are no unevaluated successors, record this and break
                if not len(unevaluated_successors):
                    self.paths[path]['all_successors_evaluated'] = True
                    break
                
                successor = unevaluated_successors[0]
                path = path + (successor,)
            
            if path[-1] == self.goal_membership:
                return path, next_best_cost
    
    def expand_successors(self, path):
        
        # compute false positives and false negatives
        current_membership = path[-1]
        false_positives = current_membership - self.goal_membership
        false_negatives = self.goal_membership - current_membership
        
        # compute possible successors
        successors = set()
        
        # if there are any false positives, remove them first
        if false_positives:
            for fp in false_positives:
                fp_start_id = self.false_positive_start_ids[fp]
                fp_shape = self.start_assembly['shape'][fp_start_id]
                if self.removable_collision_free(
                    fp_start_id,
                    current_membership,
                    maintain_connectivity=True,
                ):
                    successors.add(path[-1] - frozenset((fp,)))
        
        # if there are no false positives, but false negatives, add them
        elif false_negatives:
            for fn in false_negatives:
                # check if the false negative can be added
                fn_shape = self.goal_assembly['shape'][fn]
                if self.addable_collision_free(
                    fn,
                    self.shape_id_to_brick_shape[fn_shape],
                    path[-1],
                ):
                    successors.add(path[-1] | frozenset((fn,)))
        
        # add the succesors to the roadmap
        self.paths[path]['successors'] = successors
        
        # add the successor paths roadmap
        for successor in successors:
            self.initialize_path(path + (successor,))
    
    def removable_collision_free(
        self,
        remove_brick,
        current_membership,
        maintain_connectivity=False,
    ):
        
        collision_map = self.start_collision_map
        start_assembly = self.start_assembly
        
        # check if removing this brick will affect connectivity
        if maintain_connectivity and len(current_membership) > 2:
            
            # first build up the connectivity structure of this membership
            edges = start_assembly['edges']
            connectivity = {}
            for i in range(edges.shape[1]):
                if (edges[0,i] == 0 or
                    edges[0,i] not in current_membership or
                    edges[0,i] == remove_brick):
                    continue
                
                if edges[1,i] in current_membership:
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
            
            if len((current_membership - {remove_brick}) - connected):
                return False
        
        remove_snaps = []
        for axis, polarity, snap_group in collision_map[remove_brick]:
            colliding_bricks = collision_map[remove_brick][
                axis, polarity, snap_group]
            colliding_bricks = frozenset(colliding_bricks)
            if not len(colliding_bricks & current_membership):
                remove_snaps.extend(snap_group)
        
        if len(remove_snaps):
            return True
        else:
            return False
    
    def addable_collision_free(
        self,
        new_brick,
        new_brick_shape_name,
        existing_bricks,
    ):
        
        collision_map = self.goal_collision_map
        goal_assembly = self.goal_assembly
        
        if not len(existing_bricks):
            brick_shape = BrickShape(new_brick_shape_name)
            brick_transform = goal_assembly['pose'][new_brick]
            brick_instance = BrickInstance(0, brick_shape, 0, brick_transform)
            upright_snaps = brick_instance.get_upright_snaps()
            if len(upright_snaps):
                return True
            else:
                return False
        
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
                return False
            
            updated_bricks = existing_bricks | frozenset((new_brick,))
            
            # make sure each brick that hasn't been added yet will still have
            # a way to get connected without collision after this brick has been
            # added
            unadded_bricks = self.goal_membership - updated_bricks
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
    
    def initialize_path(self, path):
        self.paths[path] = {
            'view_changes':0, #None,
            'total_view_changes':0,
            'env_state':None,
            'action_seq':[],
            'observation_seq':[],
            'reward_seq':[],
            'successors':None,
            'evaluated':False,
            'all_successors_evaluated':False,
            'good_actions':set(),
        }
    
    def check_path(self, candidate_path, next_best_cost):
        goal_to_wip = copy.deepcopy(self.goal_to_wip)
        
        iterate = range(1, len(candidate_path))
        for i in iterate:
            a_path = candidate_path[:i]
            b_path = candidate_path[:i+1]
            a_path_data = self.paths[a_path]
            b_path_data = self.paths[b_path]
            
            # evaluate the edge if it has not been evaluated yet
            if not b_path_data['evaluated']:
                
                # check the edge
                observation_seq, action_seq, reward_seq = (
                    self.check_edge(b_path, goal_to_wip))
                
                #a_path_data['good_actions'] != better_actions
                
                b_path_data['evaluated'] = True
                
                # if this action is not feasible:
                # delete b_path and all successors from the graph
                # and remove b_path from a_path's successors
                # then return False (path to goal not found yet)
                if action_seq is None:
                    b = b_path[-1]
                    successors = self.paths[a_path]['successors']
                    successors.remove(b)
                    
                    for j in range(i, len(candidate_path)):
                        post_feasible_path = candidate_path[:j+1]
                        del(self.paths[post_feasible_path])
                    
                    return False
                
                # if the path is feasible:
                # update the observations, actions, rewards and env_state
                b_path_data['observation_seq'] = observation_seq
                b_path_data['action_seq'] = action_seq
                b_path_data['reward_seq'] = reward_seq
                b_path_data['env_state'] = self.env.get_state()
                
                # find out how many view changes were necessary
                view_changes = len([
                    a for a in action_seq if (
                        a['table_viewpoint'] != 0 or
                        a['hand_viewpoint'] != 0
                    )
                ])
                b_path_data['view_changes'] = view_changes
                b_path_data['total_view_changes'] = (
                    a_path_data['total_view_changes'] + view_changes)
            
                # if the cost so far is greater than the next best path:
                # return False (path to goal not found yet)
                # (only do this when evaluating new nodes to ensure progress)
                b_cost = self.cost(b_path)
                if b_cost > next_best_cost:
                    return False
            
            a, b = b_path[-2:]
            if len(a) < len(b):
                if len(goal_to_wip):
                    next_instance = max(goal_to_wip.values()) + 1
                else:
                    next_instance = 1
                instance = next(iter(b-a))
                goal_to_wip[instance] = next_instance
        
        return True
    
    def check_edge(self, path, goal_to_wip):
        
        # initialize the env state
        prev_path = path[:-1]
        env_state = self.paths[prev_path]['env_state']
        start_observation = self.env.set_state(env_state)
        
        a, b = path[-2:]
        assert abs(len(a) - len(b)) == 1
        
        # add a brick
        if len(a) < len(b):
            
            # add the first brick
            if len(a) == 0:
                instance = next(iter(b))
                return plan_add_first_brick(
                    self.env,
                    self.goal_assembly,
                    instance,
                    start_observation,
                    goal_to_wip,
                    self.shape_id_to_brick_shape,
                    split_cursor_actions=self.split_cursor_actions,
                    debug=False,
                )
            
            # add the nth brick
            else:
                instance = next(iter(b-a))
                return plan_add_nth_brick(
                    self.env,
                    self.goal_assembly,
                    instance,
                    start_observation,
                    goal_to_wip,
                    self.shape_id_to_brick_shape,
                    split_cursor_actions=self.split_cursor_actions,
                    allow_snap_flip=self.allow_snap_flip,
                    debug=False,
                )
        
        # remove a brick
        elif len(b) < len(a):
            instance = next(iter(a-b))
            return plan_remove_nth_brick(
                self.env,
                self.goal_assembly,
                instance,
                start_observation,
                self.false_positive_goal_ids,
                self.shape_id_to_brick_shape,
                split_cursor_actions=self.split_cursor_actions,
                debug=False,
            )
    
    #def make_false_positive_labels(self, fp):
    # this was always wrong I guess?
    #    max_fp = max(self.false_positive_labels, default=0)
    #    new_labels = {f + max_fp:f for f in fp}
    #    self.false_positive_labels |= new_labels.keys()
    #    
    #    return new_labels

class RoadmapPlanner:
    #def __init__(self, roadmap, start_env_state):
    #    
    #    # initialize
    #    self.roadmap = roadmap
    
    
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
            
            c = [self.roadmap.paths[path + (successor,)]['point_changes']
                for successor in successors]
            
            # VISIBLE THINGS GOES HERE
            
            best_c, best_successor = min(zip(c, successors))
            
            return best_successor
        else:
            return None
    

class EdgeChecker:
    def __init__(self, planner, roadmap):
        self.planner = planner
        self.roadmap = roadmap
    

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
