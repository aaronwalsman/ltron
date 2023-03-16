from ltron.gym.components.ltron_gym_component import LtronGymComponent

class SensorComponent(LtronGymComponent):
    def __init__(self, update_frequency='step', observable=True):
        assert update_frequency in (
            'init', 'reset', 'step', 'on_demand', 'always')
        self.update_frequency = update_frequency
        self.observable = observable
        self.stale = True
        
        if update_frequency in ('init',):
            self.observe()
    
    def observe(self):
        if self.stale:
            self.update_observation()
            if self.update_frequency != 'always':
                self.stale = False
        return self.observation
    
    def reset(self):
        if self.update_frequency not in ('init',):
            self.stale = True
        if self.update_frequency in ('step', 'reset', 'always'):
            self.observe()
        if self.observable:
            return self.observation, None
        else:
            return None, None
    
    def step(self, action):
        if self.update_frequency in ('step', 'on_demand', 'always'):
            self.stale = True
        if self.update_frequency in ('step', 'always'):
            self.observe()
        if self.observable:
            return self.observation, 0., False, False, None
        else:
            return None, 0., False, False, None
    
    def get_state(self):
        return self.observation, self.stale
    
    def get_state(self, state):
        self.observation, self.stale = state
        return self.observation
    
    def update_observation(self):
        raise NotImplementedError
