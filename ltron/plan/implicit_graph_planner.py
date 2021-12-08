import random

import numpy

from ltron.plan.planner import Planner

class GridPlanner(Planner):
    def __init__(self, dimension, goal, obstacles):
        super(GridPlanner, self).__init__()
        self.dimension = dimension
        self.goal = goal
        self.graph = {}
        self.obstacles = obstacles
    
    def sample_initial_state(self):
        return tuple([0] * self.dimension)
    
    def action_space(self, state):
        if state not in self.graph:
            actions = []
            for d in range(self.dimension):
                for direction in 1, -1:
                    s = list(state)
                    s[d] += direction
                    s = tuple(s)
                    if s not in self.obstacles:
                        actions.append(s)
            
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
    obstacles = [
        (0,1),
        (1,0),
        (1,1),
        (2,0),
        (2,1),
        (2,2),
        (1,2),
        (0,2),
    ]
    planner = GridPlanner(2, (3,3), obstacles)
    
    rollouts = planner.rollout_until_duration(10, max_steps=10)
    import pdb
    pdb.set_trace()
