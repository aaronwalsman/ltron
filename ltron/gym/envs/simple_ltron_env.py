import numpy

from gymnasium import Env

from steadfast.config import Config

from ltron.bricks.scene import BrickScene

class LtronEnvConfig(Config):
    observe_color=True

NO THIS ISNT SIMPLER, ITS JUST HARDCODED

class LtronEnv(Env):
    def __init__(self, config):
        self.config = config
        self.scene = BrickScene()
        
        self.build_observation_space()
        self.build_action_space()
        
        if config.observe_color:
            self.scene.make_renderable()
            self.framebuffer = FrameBufferWrapper(
                config.screen_width, config.screen_height, True)
    
    def build_action_space(self):
        action_space = {}
        
        if self.config.controllable_viewpoint:
            viewpoint_action_space = {}
            viewpoint_action_space['azimuth'] = Discrete(3)
            viewpoint_action_space['elevation'] = Discrete(3)
            viewpoint_action_space['distance'] = Discrete(3)
            viewpoint_action_space['translate'] = MultiDiscrete((3,3,3))
            action_space['viewpoint'] = Dict(viewpoint_action_space)
        
        
    
    def build_observation_space(self):
        observation_space = {}
        
        if self.config.observe_viewpoint:
            observation_space['viewpoint'] = SE3Space()
    
    def reset(self, seed, options):
        self.time_step = 0
        
        if self.controllable_camera:
        
        observation = self.compute_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        observation = self.compute_observation()
        reward = self.compute_reward()
        terminal = self.compute_terminal()
        truncated = False
        info = {}
        
        self.time_step += 1
        
        return observation, reward, terminal, truncated, info
    
    def compute_terminal(self):
        if self.time_step >= self.config.max_time_steps:
            return True
        
        return False
    
    def compute_observation(self):
        observation = {}
        if self.config.observe_viewpoint:
            observation['viewpoint'] = numpy.linalg.inv(
                self.scene.get_view_matrix())
    
    def compute_reward(self):
        return 0.
    
    def no_op_action(self):
        action = {}
        
        if self.config.controllable_viewpoint:
            action['viewpoint'] = {
                'azimuth' : 1,
                'elevation' : 1,
                'distance' : 1,
                'translate' : (1,1,1),
            }
        
            }
