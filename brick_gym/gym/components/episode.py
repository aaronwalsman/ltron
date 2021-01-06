from brick_gym.gym.spaces import StepSpace
from brick_gym.envs.components.brick_env_component import BrickEnvComponent

class MaxEpisodeLengthComponent(BrickEnvComponent):
    def __init__(self,
            episode_length,
            episode_step_key = 'episode_step'):
        
        self.episode_length = episode_length
        self.episode_step_key = episode_step_key
    
    def update_observation_space(self, observation_space):
        observation_space[self.episode_step_key] = StepSpace(
                self.episode_length)
    
    def initialize_state(self, state):
        state[self.episode_step_key] = None
    
    def reset_state(self, state):
        state[self.episode_step_key] = 0
    
    def update_state(self, state, action):
        state[self.episode_step_key] += 1
    
    def compute_observation(self, state, observation):
        observation[self.episode_step_key] = state[self.episode_step_key]
    
    def check_terminal(self, state):
        terminal = state[self.episode_step_key] >= self.episode_length
        return terminal
