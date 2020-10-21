import math

import gym

from brick_gym.viewpoint_control import AzimuthElevationViewpointControl
from brick_gym.mpd_sequence import MPDSequence

class ViewpointEnv(gym.Env):
    def __init__(self,
            directory,
            split,
            reward_function,
            steps_per_episode = 8,
            model_selection_mode = 'random',
            width = 256,
            height = 256,
            distance = 500,
            azimuth_steps = 24,
            elevation_steps = 7,
            elevation_range = (-math.pi/3, math.pi/3),
            subset=None):
        
        self.viewpoint = AzimuthElevationViewpointControl(
                distance = distance,
                azimuth_steps = azimuth_steps,
                elevation_steps = elevation_steps,
                elevation_range = elevation_range)
        
        self.mpd_sequence = MPDSequence(
                directory, split, subset, model_selection_mode, width, height)
        
        self.reward_function = reward_function
        self.steps_per_episode = steps_per_episode
        
        self.action_space = self.viewpoint.action_space
        self.observation_space = self.mpd_sequence.observation_space
    
    def compute_azimuth_elevation(self):
        self.azimuth = (self.azimuth_index / self.azimuth_steps) * math.pi * 2
        self.elevation = (
                (self.elevation_index / self.elevation_steps) *
                (self.elevation_range[1] - self.elevation_range[0]) +
                self.elevation_range[0])
    
    def set_state(self, state):
        self.viewpoint.set_state(state)
    
    def get_state(self):
        return self.viewpoint.get_state()
    
    def step(self, action):
        self.viewpoint.step(action)
        camera_pose = self.viewpoint.get_transform()
        self.mpd_sequence.set_camera_pose(camera_pose)
        observation = self.mpd_sequence.observe()
        reward = self.reward_function(
                observation, self.mpd_sequence.recent_mask)
        done = False
        
        return observation, reward, done, {}
    
    def reset(self, state=None, render=True):
        self.mpd_sequence.increment_scene()
        center = self.mpd_sequence.scene_center
        distance = self.mpd_sequence.scene_distance
        self.viewpoint.reset(state, center=center, distance=distance)
        self.mpd_sequence.set_camera_pose(self.viewpoint.get_transform())
        
        if render:
            return self.mpd_sequence.observe()
        else:
            return None
    
    def close(self):
        pass
    
    def render(self, mode='human', close=False):
        self.mpd_sequence.render(self, mode, close)
