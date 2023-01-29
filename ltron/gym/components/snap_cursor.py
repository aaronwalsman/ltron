import numpy

from gymnasium.spaces import Dict, MultiDiscrete

from supermecha import CursorComponent
from supermecha.gym.spaces import IntegerMaskSpace

from ltron.constants import MAX_SNAPS_PER_BRICK, MAX_INSTANCES_PER_SCENE
from ltron.gym.components import SnapMaskRenderComponent

class SnapCursorComponent(CursorComponent):
    def __init__(self,
        scene_component,
        pos_render_component,
        neg_render_component,
        height,
        width,
        train
    ):
        super().__init__(height, width)
        self.pos_render_component = pos_render_component
        self.neg_render_component = neg_render_component
        self.train = train
        self.click_snap = numpy.array([0,0])
        self.release_snap = numpy.array([0,0])
        
        self.observation_space = Dict({
            'click_snap' : MultiDiscrete(
                (MAX_INSTANCES_PER_SCENE, MAX_SNAPS_PER_BRICK)),
            'release_snap' : MultiDiscrete(
                (MAX_INSTANCES_PER_SCENE, MAX_SNAPS_PER_BRICK)),
        })
    
    def compute_observation(self):
        return {
            'click_snap' : numpy.array(self.click_snap),
            'release_snap' : numpy.array(self.release_snap),
        }
    
    def reset(self, seed=None, rng=None, options=None):
        o,i = super().reset(seed=seed, rng=rng, options=options)
        o = self.compute_observation()
        
        self.click_snap = numpy.array([0,0])
        self.release_snap = numpy.array([0,0])
        
        return o,i
    
    def step(self, action):
        o,r,t,u,i = super().step(action)
        o = self.compute_observation()
        
        p = self.pos_render_component.observation
        n = self.neg_render_component.observation
        
        if self.button:
            self.click_snap = p[tuple(self.click)]
            self.release_snap = n[tuple(self.release)]
        else:
            self.click_snap = n[tuple(self.click)]
            self.release_snap = p[tuple(self.release)]
        
        return o,r,t,u,i
    
    def get_state(self):
        return (self.click_snap, self.release_snap), super().get_state()
    
    def set_state(self, state):
        (self.click_snap, self.release_snap), cursor_state = state
        super().set_state(cursor_state)
