import random
import time
import math
import copy
from bisect import insort

import numpy

from ltron.exceptions import LTronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map
from ltron.geometry.epsilon_array import EpsilonArray
from ltron.matching import match_configurations, match_lookup
from ltron.bricks.brick_instance import BrickInstance
from ltron.bricks.brick_type import BrickType
from ltron.gym.envs.reassembly_env import reassembly_template_action

from ltron.planner.edge_planner import (
    plan_add_first_brick,
    plan_add_nth_brick,
)

class PlanningException(LTronException):
    pass

class PathNotFoundError(PlanningException):
    pass

def node_connected_collision_free(
    new_brick,
    new_brick_type_name,
    existing_bricks,
    goal_state,
    collision_map,
    goal_configuration,
):
    if not len(existing_bricks):
        brick_type = BrickType(new_brick_type_name)
        brick_transform = goal_configuration['pose'][new_brick]
        brick_instance = BrickInstance(0, brick_type, 0, brick_transform)
        upright_snaps = brick_instance.get_upright_snaps()
        return bool(len(upright_snaps))
    
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
        self.brick_type_to_class = class_ids
        self.class_to_brick_type = {
            value:key for key, value in self.brick_type_to_class.items()}
        self.color_ids = color_ids
        
        goal_scene = BrickScene(renderable=True, track_snaps=True)
        goal_scene.import_configuration(
            self.goal_config, self.brick_type_to_class, self.color_ids)
        self.goal_collision_map = build_collision_map(goal_scene)

class RoadmapPlanner:
    def __init__(self, roadmap, start_env_state):
        
        # initialize
        self.roadmap = roadmap
        self.edge_checker = EdgeChecker(self, self.roadmap)
        
        # set the env to the start state
        observation = self.roadmap.env.set_state(start_env_state)
        self.start_config = observation['workspace_scene']['config']
        
        # compute a matching
        matching, offset = match_configurations(
            self.start_config, roadmap.goal_config)
        self.wip_to_goal, self.goal_to_wip, fp, fn = match_lookup(
            matching, self.start_config, roadmap.goal_config)
        
        self.start_state = frozenset(self.goal_to_wip.keys())
        self.roadmap.nodes.add(self.start_state)
        assert self.start_state not in self.roadmap.env_states
        self.roadmap.env_states[self.start_state] = start_env_state
        
        self.visits = {}
        
        # build start collision map
        start_scene = BrickScene(renderable=True, track_snaps=True)
        start_scene.import_configuration(
            self.start_config,
            self.roadmap.brick_type_to_class,
            self.roadmap.color_ids,
        )
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
            if goal_found:
                print('goal found')
                candidate_path, goal_feasible = self.edge_checker.check_path(
                    candidate_path)
                print('feasible?', goal_feasible)
                import pdb
                pdb.set_trace()
                q = something_else
            else:
                q = goal_found
            
            #if goal_feasible:
            #    q = 1
            #else:
            #    q = -1
            #w += q
            #n += 1
            #print(w/n)
            
            self.update_path_visits(candidate_path, goal_found)
    
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
                successor = state - frozenset((false_positive,))
                successors.add(successor)
        
        # if there are no false positives, but false negatives, add them
        elif false_negatives:
            for false_negative in false_negatives:
                # check if false_negative can be added
                false_negative_class = self.roadmap.goal_config['class'][
                    false_negative]
                if node_connected_collision_free(
                    false_negative,
                    self.roadmap.class_to_brick_type[false_negative_class],
                    state,
                    self.roadmap.goal_state,
                    self.roadmap.goal_collision_map,
                    self.roadmap.goal_config,
                ):
                    successor = state | frozenset((false_negative,))
                    successors.add(successor)
        
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
        for a, b in zip(candidate_path[:-1], candidate_path[1:]):
            edge = self.roadmap.edges[a,b]
            if b not in self.roadmap.env_states:
                action_seq = self.check_edge(a, b, goal_to_wip)
                self.roadmap.edges[a,b]['feasible'] = action_seq
                if action_seq is False:
                    self.roadmap.successors[a].remove(b)
                    return successful_path, False
                successful_path.append(b)
                last_state = self.roadmap.env.get_state()
                self.roadmap.env_states[b] = last_state
            else:
                successful_path.append(b)
        else:
            return successful_path, True
    
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
                action_seq = plan_add_first_brick(
                    self.roadmap.env,
                    self.roadmap.goal_config,
                    instance,
                    observation,
                    goal_to_wip,
                    self.roadmap.class_to_brick_type,
                )
                goal_to_wip[instance] = next_instance
                return action_seq
            
            else:
                # add the nth brick
                instance = next(iter(b-a))
                #action_seq = self.plan_add_nth_brick(
                #    instance, observation, goal_to_wip)
                action_seq = plan_add_nth_brick(
                    self.roadmap.env,
                    self.roadmap.goal_config,
                    instance,
                    observation,
                    goal_to_wip,
                )
                goal_to_wip[instance] = next_instance
        
        elif len(b) < len(a):
            # remove a brick
            return self.plan_remove_nth_brick(next(iter(a-b)), observation)
    
    '''
    def plan_add_first_brick(self, instance, observation, goal_to_wip):
        action_seq = []
        brick_class = self.roadmap.goal_config['class'][instance]
        brick_color = self.roadmap.goal_config['color'][instance]
        brick_transform = self.roadmap.goal_config['pose'][instance]
        brick_type = BrickType(self.roadmap.class_to_brick_type[brick_class])
        brick_instance = BrickInstance(
            0, brick_type, brick_color, brick_transform)
        upright_snaps, upright_snap_ids = brick_instance.get_upright_snaps()
        
        if not len(upright_snaps):
            return False
        
        # make the insert action
        insert_action = reassembly_template_action()
        insert_action['insert_brick'] = {
            'class_id' : brick_class,
            'color_id' : brick_color,
        }
        action_seq.append(insert_action)
        observation, reward, terminal, info = self.roadmap.env.step(
            insert_action)
        
        snap_y = []
        snap_x = []
        snap_p = []
        snap_s = []
        for snap, i in zip(upright_snaps, upright_snap_ids):
            if snap.polarity == '+':
                snap_map = observation['handspace_pos_snap_render']
            else:
                snap_map = observation['handspace_neg_snap_render']
            y, x = numpy.where((snap_map[:,:,0] == 1) & (snap_map[:,:,1] == i))
            snap_y.append(y)
            snap_x.append(x)
            snap_p.append(numpy.ones(len(y)) * (snap.polarity == '+'))
            snap_s.append(numpy.ones(len(y), dtype=numpy.long) * i)
        
        snap_y = numpy.concatenate(snap_y)
        snap_x = numpy.concatenate(snap_x)
        snap_p = numpy.concatenate(snap_p)
        snap_s = numpy.concatenate(snap_s)
        
        # this would be strange, but let's do it anyway for safety
        if not len(snap_y):
            # TODO test other camera locations
            return False
        
        i = random.randint(0, len(snap_y)-1)
        y = snap_y[i]
        x = snap_x[i]
        p = snap_p[i]
        s = snap_s[i]
        
        print('picked snap:', s)
        
        # make the pick and place action
        pick_and_place_action = reassembly_template_action()
        pick_and_place_action['handspace_cursor'] = {
            'activate':True,
            'position':[y,x],
            'polarity':p,
        }
        pick_and_place_action['pick_and_place'] = {
            'activate':True,
            'place_at_origin':True,
        }
        action_seq.append(insert_action)
        observation = self.roadmap.env.step(pick_and_place_action)
        
        return action_seq
    '''
    def plan_add_nth_brick(self, instance, observation, goal_to_wip):
        
        dump_obs(observation, 0)
        
        print('adding: %i'%instance)
        action_seq = []
        brick_class = self.roadmap.goal_config['class'][instance]
        brick_color = self.roadmap.goal_config['color'][instance]
        brick_transform = self.roadmap.goal_config['pose'][instance]
        
        connections = self.roadmap.goal_config['edges'][0] == instance
        extant_connections = numpy.array([
            self.roadmap.goal_config['edges'][1,i] in goal_to_wip
            for i in range(self.roadmap.goal_config['edges'].shape[1])])
        connections = connections & extant_connections
        connected_instances = self.roadmap.goal_config['edges'][1][connections]
        # need to remove the ones that don't exist yet
        remapped_instances = [
            goal_to_wip[i] for i in connected_instances]
        instance_snaps = self.roadmap.goal_config['edges'][2][connections]
        connected_snaps = self.roadmap.goal_config['edges'][3][connections]
        h_to_w = {
            (i,s):(1,cs) for i, s, cs in
            zip(remapped_instances, instance_snaps, connected_snaps)
        }
        w_to_h = {v:k for k,v in h_to_w.items()}
        
        # make the insert action
        insert_action = reassembly_template_action()
        insert_action['insert_brick'] = {
            'class_id' : brick_class,
            'color_id' : brick_color,
        }
        action_seq.append(insert_action)
        observation, reward, terminal, info = self.roadmap.env.step(
            insert_action)
        
        dump_obs(observation, 1)
        
        # compute the necessary workspace camera motion
        workspace_condition = self.make_snap_finder(
            remapped_instances, connected_snaps, 'workspace')
        state = self.roadmap.env.get_state()
        start_workspace_camera_position = (
            tuple(state['workspace_viewpoint']['position']) + (0,))
        workspace_camera_position, workspace_snaps = self.search_camera_space(
            'workspace_viewpoint', state, workspace_condition, float('inf'))
        
        # convert the workspace camera motion to camera actions
        workspace_camera_actions = self.compute_camera_actions(
            'workspace',
            start_workspace_camera_position,
            workspace_camera_position,
        )
        action_seq.extend(workspace_camera_actions)
        
        # update the state
        self.replace_camera_in_state(
            state, 'workspace_viewpoint', workspace_camera_position)
        observation = self.roadmap.env.set_state(state)
        dump_obs(observation, 2)
        
        # figure out which snaps to look for in hand space
        wy, wx, wp, wi, ws = workspace_snaps
        #instance_s = [connected_to_instance_snaps[s] for s in ws]
        instance_s = [w_to_h[i,s][1] for i,s in zip(wi, ws)]
        handspace_condition = self.make_snap_finder(
            numpy.ones(ws.shape[0], dtype=numpy.long), instance_s, 'handspace')
        start_handspace_camera_position = (
            tuple(state['handspace_viewpoint']['position']) + (0,))
        handspace_camera_position, handspace_snaps = self.search_camera_space(
            'handspace_viewpoint', state, handspace_condition, float('inf'))
        
        # convert the handspace camera motion to camera actions
        handspace_camera_actions = self.compute_camera_actions(
            'handspace',
            start_handspace_camera_position,
            handspace_camera_position,
        )
        action_seq.extend(handspace_camera_actions)
        
        # update the state
        self.replace_camera_in_state(
            state, 'handspace_viewpoint', handspace_camera_position)
        observation = self.roadmap.env.set_state(state)
        dump_obs(observation, 3)
        
        # pick one of these handspace snaps, then figure out which workspace
        # snap it corresponds to
        hy, hx, hp, hi, hs = handspace_snaps
        r = random.randint(0, hs.shape[0]-1)
        hyy, hxx, hpp, hii, hss = handspace_snaps[:,r]
        connected_i, connected_s = h_to_w[hii, hss]
        workspace_locations = numpy.where(
            (wi == connected_i) & (ws == connected_s))[0]
        r = random.choice(workspace_locations)
        wyy, wxx, wpp, wii, wss = workspace_snaps[:,r]
        
        # make the pick and place action
        pick_and_place_action = reassembly_template_action()
        pick_and_place_action['workspace_cursor'] = {
            'activate':True,
            'position':[wyy, wxx],
            'polarity':wpp,
        }
        pick_and_place_action['handspace_cursor'] = {
            'activate':True,
            'position':[hyy, hxx],
            'polarity':hpp,
        }
        pick_and_place_action['pick_and_place'] = {
            'activate':True,
            'place_at_origin':False,
        }
        action_seq.append(pick_and_place_action)
        observation, reward, terminal, info = self.roadmap.env.step(
            pick_and_place_action)
        
        print('d4?')
        dump_obs(observation, 4)
        
        # TODO: why is the brick not being added???
        
        import pdb
        pdb.set_trace()
        
        # TODO: rotation action goes here
        
        return action_seq
    
    def plan_remove_brick(self, instance):
        pass
    
    def compute_camera_actions(self, view_space, start_position, end_position):
        return []
    
    '''
    def replace_camera_in_state(self, state, component_name, position):
        state[component_name]['position'] = position[:3]
        if position[-1]:
            component = self.roadmap.env.components[component_name]
            center = component.compute_center()
            state[component_name]['center'] = center
    
    def test_camera_position(self, position, component_name, state, condition):
        new_state = copy.deepcopy(state)
        self.replace_camera_in_state(new_state, component_name, position)
        observation = self.roadmap.env.set_state(new_state)
        return condition(observation)
    
    def make_snap_finder(self, instances, snaps, view_space):
        def condition(observation):
            matching_y = []
            matching_x = []
            matching_p = []
            matching_i = []
            matching_s = []
            for i, s in zip(instances, snaps):
                pos_snap_render = observation['%s_pos_snap_render'%view_space]
                neg_snap_render = observation['%s_neg_snap_render'%view_space]
                pos_y, pos_x = numpy.where(
                    (pos_snap_render[:,:,0] == i) &
                    (pos_snap_render[:,:,1] == s)
                )
                pos_p = numpy.ones(pos_y.shape[0], dtype=numpy.long)
                pos_i = numpy.ones(pos_y.shape[0], dtype=numpy.long) * i
                pos_s = numpy.ones(pos_y.shape[0], dtype=numpy.long) * s
                matching_y.append(pos_y)
                matching_x.append(pos_x)
                matching_p.append(pos_p)
                matching_i.append(pos_i)
                matching_s.append(pos_s)
                neg_y, neg_x = numpy.where(
                    (neg_snap_render[:,:,0] == i) &
                    (neg_snap_render[:,:,1] == s)
                )
                neg_p = numpy.zeros(neg_y.shape[0], dtype=numpy.long)
                neg_i = numpy.ones(neg_y.shape[0], dtype=numpy.long) * i
                neg_s = numpy.ones(neg_y.shape[0], dtype=numpy.long) * s
                matching_y.append(neg_y)
                matching_x.append(neg_x)
                matching_p.append(neg_p)
                matching_i.append(neg_i)
                matching_s.append(neg_s)
            
            matching_y = numpy.concatenate(matching_y)
            matching_x = numpy.concatenate(matching_x)
            matching_p = numpy.concatenate(matching_p)
            matching_i = numpy.concatenate(matching_i)
            matching_s = numpy.concatenate(matching_s)
            
            matching_yxpis = numpy.stack(
                (matching_y, matching_x, matching_p, matching_i, matching_s),
                axis=0,
            )
            
            success = bool(matching_yxpis.shape[1])
            print('?', instances, snaps, success)
            if success is False:
                import pdb
                pdb.set_trace()
            return success, matching_yxpis
        return condition
    
    def search_camera_space(self, component_name, state, condition, max_steps):
        
        # BFS
        component = self.roadmap.env.components[component_name]
        current_position = tuple(component.position) + (0,)
        explored = set()
        frontier = [(0,) + current_position]
        min_position = (0,0,0,0)
        max_position = (
            component.azimuth_steps-1,
            component.elevation_steps-1,
            component.distance_steps-1,
            1,
        )
        
        def modular_distance(a,b,m):
            first = abs(a-b)
            second = abs(a+m-b)
            third = abs(b+m-a)
            return min(first, second, third)
        
        explored.add(current_position)
        
        while frontier:
            distance, *position = frontier.pop(0)
            position = tuple(position)
            
            test_result, test_info = self.test_camera_position(
                position, component_name, state, condition
            )
            if test_result:
                return position, test_info
            
            
            for i in range(4):
                for direction in (-1, 1):
                    offset = numpy.zeros(4, dtype=numpy.long)
                    offset[i] += direction
                    new_position = position + offset
                    new_position = numpy.concatenate((
                        new_position[[0]] % component.azimuth_steps,
                        new_position[1:]
                    ))
                    new_distance = numpy.sum(
                        numpy.abs(new_position[1:] - position[1:]))
                    new_distance += modular_distance(
                        new_position[0], position[0], component.azimuth_steps)
                    if (numpy.all(new_position >= min_position) and
                        numpy.all(new_position <= max_position) and
                        new_distance <= max_steps and
                        tuple(new_position) not in explored
                    ):
                        new_position = tuple(new_position)
                        explored.add(new_position)
                        new_distance_position = (new_distance,) + new_position
                        insort(frontier, new_distance_position)
        
        return None, None
    '''

def upper_confidence_bound(q, n_action, n_state, c=2**0.5):
    return q + c * (math.log(n_state+1)/(n_action+1))**0.5
