import random
import time

class Planner:
    
    def __init__(self, c=1.):
        self.stats = {}
        self.successors = {}
        self.c = c
    
    def rollout_until_n(self, n, max_steps=float('inf'), mode='pucb'):
        rollouts = []
        for i in range(n):
            rollout = self.rollout(max_steps=max_steps, mode=mode)
            self.update_stats(*rollout)
            rollouts.append(rollout)
        
        return rollouts
    
    def rollout_until_duration(self, t, max_steps=float('inf'), mode='pucb'):
        t_start = time.time()
        rollouts = []
        while time.time() - t_start < t:
            rollout = self.rollout(max_steps=max_steps, mode=mode)
            self.update_stats(*rollout)
            rollouts.append(rollout)
        
        return rollouts
    
    def rollout_until_score(self, s, max_steps=float('inf'), mode='pucb'):
        score = float('-inf')
        rollouts = []
        while score < s:
            rollout = self.rollout(max_steps=max_steps, mode=mode)
            score, s, a = rollout
            self.update_stats(score, s, a)
            rollouts.append(rollout)
        
        return rollouts
    
    def rollout(self, max_steps=float('inf'), mode='pucb'):
        state = self.sample_initial_state()
        
        states = [state]
        actions = []
        step = 0
        while not self.terminal(states, actions) and step < max_steps:
            
            if state not in self.stats:
                self.initialize_stats(state)
            
            if mode == 'pucb':
                action, state = self.pucb_successor(state)
            elif mode == 'max':
                action, state = self.max_successor(state)
            
            states.append(state)
            actions.append(action)
            step += 1
        
        score = self.score_rollout(states, actions)
        
        return score, states, actions
    
    def initialize_stats(self, state):
        self.stats[state] = {}
        self.stats[state]['n'] = 0
        self.successors[state] = {}
        actions, successors, predictions = self.action_space(state)
        for action, successor, p in zip(actions, successors, predictions):
            self.stats[state, action] = {}
            self.stats[state, action]['w'] = 0.
            self.stats[state, action]['p'] = p
            self.stats[state, action]['n'] = 0
            self.successors[state][action] = successor
    
    def update_stats(self, score, states, actions):
        for state, action in zip(states[:-1], actions):
            self.stats[state]['n'] += 1
            self.stats[state, action]['w'] += score
            self.stats[state, action]['n'] += 1
    
    def pucb_successor(self, state):
        bounds = {}
        for action in self.successors[state]:
            bounds[action] = self.pucb_action(state, action)
        
        action = max(self.successors[state], key=lambda a : bounds[a])
        return action, self.successors[state][action]
    
    def pucb_action(self, state, action):
        n_action = self.stats[state, action]['n']
        n_state = self.stats[state]['n']
        q = self.stats[state, action]['w'] / (n_action + 1)
        p = self.stats[state, action]['p']
        return pucb(q, p, n_action, n_state, self.c)
    
    def max_successor(self, state):
        qs = []
        for action in self.sucessors[state]:
            n_action = self.stats[state, action]['n']
            q = self.stats[state, action]['w'] / n_action
            qs.append((q, action))
        
        q, action = max(qs)
        return action
    
    def sample_initial_state(self):
        raise NotImplementedError
    
    def action_space(self, state):
        return NotImplementedError
    
    def terminal(self, states, actions):
        raise NotImplementedError
    
    def score_rollout(self, states, actions):
        raise NotImplementedError

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
