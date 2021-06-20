from ltron.gym.spaces import StepSpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class MaxEpisodeLengthComponent(LtronGymComponent):
    def __init__(self, max_episode_length, observe_step=True):
        self.max_episode_length = max_episode_length
        self.episode_step = None
        self.observe_step = observe_step
        if self.observe_step:
            self.observation_space = StepSpace(max_episode_length)
    
    def reset(self):
        self.episode_step = 0
        if self.observe_step:
            observation = self.episode_step
        else:
            observation = None
        return observation
    
    def step(self, action):
        self.episode_step += 1
        if self.observe_step:
            observation = self.episode_step
        else:
            observation = None
        terminal = self.episode_step >= self.max_episode_length
        return observation, 0., terminal, None
