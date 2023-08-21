from gymnasium.spaces import Discrete

from supermecha import SuperMechaComponent

class BreakAndMakePhaseSwitchComponent(SuperMechaComponent):
    def __init__(self, scene_component):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)
        self.scene_component = scene_component
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.phase = 0
        return self.phase, {}
    
    def step(self, action):
        if action:
            if self.phase == 0:
                self.scene_component.brick_scene.clear_instances()
                self.phase = 1
                terminal = False
            elif self.phase == 1:
                terminal = True
            else:
                raise ValueError('bad phase: %s'%self.phase)
        else:
            terminal = False
        
        return self.phase, 0., terminal, False, {}
    
    def no_op_action(self):
        return 0

class PhaseScoreComponent(SuperMechaComponent):
    def __init__(self, phase_component, score_component):
        self.phase_component = phase_component
        self.score_component = score_component
        if hasattr(self.score_component, 'action_space'):
            self.action_space = self.score_component.action_space
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.started_make = False
        return None, {}
    
    def step(self, action):
        if self.phase_component.phase == 1:
            if self.started_make == False:
                _ = self.score_component.reset()
            
            _, r, _, _, _ = self.score_component.step(action)
            self.started_make = True
            
            return None, r, False, False, {}
        else:
            return None, 0., False, False, {}

'''
class BreakAndMakePhaseComponent(SuperMechaComponent):
    def __init__(self, scene_component, build_score_component):
        self.action_space = Discrete(2)
        self.scene_component = scene_component
        self.build_score_component = build_score_component
    
    def reset(self, options=None):
        self.phase = 0
        return None, {}
    
    def step(self, action):
        if action:
            if self.phase == 0:
                self.phase = 1
                #self.scene_component.brick_scene.clear_instances()
                self.build_score_component.reset()
                terminal = False
            elif self.phase == 1:
                terminal = True
            else:
                raise ValueError('bad phase: %s'%self.phase)
        else:
            terminal = False
        
        if self.phase == 0:
            reward = 0
        elif self.phase == 1:
            _, reward, _, _, _ = self.build_score_component.step(None)
        else:
            raise ValueError('bad phase: %s'%self.phase)
        
        return None, reward, terminal, False, {}
    
    def no_op_action(self):
        return 0
'''
