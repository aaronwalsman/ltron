from ltron.gym import ltron_env
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym import spaces
import numpy

class TowerEnv(ltron_env.LtronEnv):
    def __init__(self, component):
        super().__init__(component)
        self.components = component
        self.observation_space = spaces.Box(low = 0, high = 256, shape = [256, 256, 4])
        self.action_space = component['pick'].action_space

    def reset(self):
        obs = numpy.zeros([256,256,4])
        for name, component in self.components.items():
            if name == "pos_snap":
                obs[:,:,:2] = component.reset()
            if name == "neg_snap":
                obs[:,:,2:] = component.reset()
            else:
                component.reset()

        return obs

    def step(self, action):
        observation = numpy.zeros([256,256,4])
        reward = 0
        terminal = False
        info = {}
        for component_name, component in self.components.items():
            if component_name == "pick":
                component_action = action
            else:
                component_action = None
            o, r, t, i = component.step(component_action)
            if component_name == "neg_snap":
                observation[:,:,:2] = o
            if component_name == "pos_snap":
                observation[:,:,2:] = o
            reward += r
            terminal |= t
            if i is not None:
                info[component_name] = i

        return observation, reward, terminal, info