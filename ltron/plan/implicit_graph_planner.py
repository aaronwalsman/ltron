import random

import numpy

from ltron.geometry.epsilon_array import EpsilonArray
from ltron.plan.planner import Planner

class GridPlanner(Planner):
    def __init__(self, dimension, goal):
        super(GridPlanner, self).__init__()
        self.dimension = dimension
        self.goal = goal
        self.graph = {}
    
    def sample_initial_state(self):
        return tuple([0] * self.dimension)
    
    def action_space(self, state):
        if state not in self.graph:
            actions = []
            for d in range(self.dimension):
                s = list(state)
                s[d] += 1
                actions.append(tuple(s))
                s = list(state)
                s[d] -= 1
                actions.append(tuple(s))
            
            random.shuffle(actions)
            self.graph[state] = actions
        
        n = len(self.graph[state])
        prediction = [1./n] * n
        return self.graph[state], self.graph[state], prediction
    
    def terminal(self, states, actions):
        return states[-1] == self.goal
    
    def score_rollout(self, states, actions):
        if states[-1] != self.goal:
            return 0.
        
        distance = len(actions)
        best_distance = sum(abs(gg) for gg in self.goal)
        return best_distance / distance

if __name__ == '__main__':
    planner = GridPlanner(5, (3,3,3,3,3))
    
    rollouts = planner.rollout_until_duration(30, max_steps=20)
    import pdb
    pdb.set_trace()
