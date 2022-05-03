import random

import numpy

from gym.spaces import Dict, Discrete, MultiDiscrete
from ltron.gym.spaces import MultiScreenPixelSpace

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

class MultiScreenCursor(LtronGymComponent):
    def __init__(self,
        max_instances,
        pos_render_components,
        neg_render_components,
        randomize_starting_position=True,
    ):
        self.max_instances = max_instances
        self.screen_dimensions = {
            n : (c.height, c.width, 2)
            for n, c in pos_render_components.items()}
        assert all(
            (c.height, c.width, 2) == self.screen_dimensions[n]
            for n, c in neg_render_components.items()
        )
        #assert all(
        #    (neg_render_components[c].height, neg_render_components[c].width) == hw
        #    for c, hw in zip(neg_render_components, self.screen_dimensions))
        self.pos_render_components = pos_render_components
        self.neg_render_components = neg_render_components
        self.screen_order = list(self.pos_render_components.keys())
        
        self.randomize_starting_position=randomize_starting_position
        
        #self.action_space = Dict({
        #    'activate':Discrete(2),
        #    'position':MultiScreenPixelSpace(self.screen_dimensions, 2),
        #})
        self.action_space = MultiScreenPixelSpace(self.screen_dimensions)
        
        self.observation_space = MultiScreenPixelSpace(self.screen_dimensions)
        if self.randomize_starting_position:
            screen = random.choice(self.screen_order)
            h,w,_ = self.screen_dimensions[screen]
            y = random.randint(0,h-1)
            x = random.randint(0,w-1)
            p = random.randint(0,1)
            self.set_cursor(screen, y, x, p)
        else:
            self.set_cursor(self.screen_order[0], 0, 0, 0)
    
    def set_cursor(self, n, y, x, p):
        self.screen_name = n
        self.position = y,x
        self.polarity = p
    
    def observe(self):
        #n = self.screen_order.index(self.screen_name)
        self.observation = self.action_space.ravel(
            self.screen_name, *self.position, self.polarity)
        
        return self.observation
    
    def reset(self):
        self.set_cursor(self.screen_order[0], 0, 0, 0)
        return self.observe()
    
    def step(self, action):
        if action:
            n, y, x, p = self.action_space.unravel(action)
            self.set_cursor(n, y, x, p)
        observation = self.observe()
        return observation, 0, False, {}
    
    def get_state(self):
        return {
            'screen_name':self.screen_name,
            'position':self.position,
            'polarity':self.polarity,
        }
    
    def set_state(self, state):
        self.screen_name = state['screen_name']
        self.position = state['position']
        self.polarity = state['polarity']
    
    def get_selected_snap(self):
        if self.polarity:
            render_component = self.pos_render_components[self.screen_name]
        else:
            render_component = self.neg_render_components[self.screen_name]
        
        snap_map = render_component.observe()
        instance_id, snap_id = snap_map[self.position[0], self.position[1]]
        return self.screen_name, instance_id, snap_id
    
    def where(self, screen_name, instance, snap):
        w = []
        pos_component = self.pos_render_components[screen_name]
        neg_component = self.neg_render_components[screen_name]
        for p, component in (1, pos_component), (0, neg_component):
            snap_map = component.observe()
            ys, xs = numpy.where(
                (snap_map[:,:,0] == instance) &
                (snap_map[:,:,1] == snap)
            )
            for y, x in zip(ys, xs):
                w.append(self.action_space.ravel(screen_name, y, x, p))
        
        return w
    
    def no_op_action(self):
        #return {'activate' : 0, 'position' : numpy.array([0])}
        return 0
