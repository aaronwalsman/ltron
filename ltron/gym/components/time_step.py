from ltron.gym.spaces import TimeStepSpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class EpisodeLengthComponent(LtronGymComponent):
    def __init__(self, max_episode_length=None, observe_step=True):
        self.max_episode_length = max_episode_length
        self.episode_step = None
        self.observe_step = observe_step
        if self.observe_step:
            self.observation_space = TimeStepSpace(max_episode_length)
    
    def observe(self):
        if self.observe_step:
            self.observation = self.episode_step
        else:
            self.observation = None
    
    def reset(self):
        self.episode_step = 0
        self.observe()
        return self.observation
    
    def step(self, action):
        self.episode_step += 1
        self.observe()
        if self.max_episode_length is not None:
            terminal = self.episode_step >= self.max_episode_length
        else:
            terminal = False
        return self.observation, 0., terminal, None
    
    def set_state(self, state):
        self.episode_step = state['episode_step']
        self.observe()
        return self.observation
    
    def get_state(self):
        return {'episode_step':self.episode_step}
