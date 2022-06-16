import math

from gym.spaces import Discrete

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class PhaseSwitch(LtronGymComponent):
    def __init__(self,
        scene_components,
        clear_scenes=False,
        num_phases=2,
    ):
        self.scene_components = scene_components
        self.clear_scenes = clear_scenes
        
        self.action_space = Discrete(num_phases+1)
        self.observation_space = Discrete(num_phases)
        self.phase = 0

    def observe(self):
        self.observation = self.phase

    def reset(self):
        self.phase = 0
        self.observe()
        return self.observation
    
    def no_op_action(self):
        return 0

    def step(self, action):
        if action > self.phase-1:
            self.phase = action
            
            if clear_scenes:
                for name, component in self.scene_components.items():
                    component.brick_scene.clear_instances()
        
        self.observe()
        return self.observation, 0., (action == num_actions-1), {}

    def set_state(self, state):
        self.phase = state['phase']
        return self.observation

    def get_state(self):
        return {'phase':self.phase}

