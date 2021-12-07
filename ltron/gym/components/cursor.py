import numpy

from gym.spaces import Dict, Discrete, MultiDiscrete

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class SnapCursor(LtronGymComponent):
    def __init__(self,
        max_instances,
        pos_snap_component,
        neg_snap_component,
        observe_instance_snap=False,
    ):
        assert pos_snap_component.height == neg_snap_component.height
        assert pos_snap_component.width == neg_snap_component.width
        self.height = pos_snap_component.height
        self.width = pos_snap_component.width
        self.max_instances = max_instances
        self.pos_snap_component = pos_snap_component
        self.neg_snap_component = neg_snap_component
        self.observe_instance_snap = observe_instance_snap
        
        self.action_space = Dict({
            'activate':Discrete(2),
            'position':MultiDiscrete((self.height, self.width)),
            'polarity':Discrete(2),
        })
        
        observation_space = {
            'position':MultiDiscrete((self.height, self.width)),
            'polarity':Discrete(2),
        }
        if self.observe_instance_snap:
            observation_space['instance_id'] = Discrete(self.max_instances+1)
            observation_space['snap_id'] = Discrete(4096)
        self.observation_space = Dict(observation_space)
        self.set_cursor(0,0,0)
    
    def set_cursor(self, y, x, polarity):
        self.position = y,x
        self.polarity = polarity
    
    def observe(self):
        if self.polarity:
            snap_map = self.pos_snap_component.observation
        else:
            snap_map = self.neg_snap_component.observation
        self.instance_id, self.snap_id = snap_map[self.position]
        
        if self.observe_instance_snap:
            self.observation = {
                'position' : self.position,
                'polarity' : self.polarity,
                'instance_id' : self.instance_id,
                'snap_id' : self.snap_id,
            }
        else:
            self.observation = {
                'position' : self.position,
                'polarity' : self.polarity,
            }
    
    def reset(self):
        self.set_cursor(0,0,0)
        self.observe()
        return self.observation
    
    def step(self, action):
        if action['activate']:
            y,x = action['position']
            polarity = action['polarity']
            self.set_cursor(y, x, polarity)
        self.observe()
        return self.observation, 0, False, {}
    
    def get_state(self):
        return {'position':self.position, 'polarity':self.polarity}
    
    def set_state(self, state):
        self.position = state['position']
        self.polarity = state['polarity']
        self.observe()
        return self.observation
    
    def no_op_action(self):
        return {'activate' : 0, 'position' : numpy.array([0,0]), 'polarity':0}
