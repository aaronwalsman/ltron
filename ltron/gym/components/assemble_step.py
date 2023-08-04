import numpy

from gymnasium.spaces import Discrete

from steadfast.hierarchy import map_hierarchies

from supermecha import SuperMechaComponent

class AssembleStepComponent(SuperMechaComponent):
    def __init__(self, phase_component):
        self.phase_component = phase_component
        self.action_space = Discrete(2)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        return None, {}
    
    def step(self, action):
        if not self.phase_component.phase:
            self.current_step += action
        else:
            self.current_step -= action
        return None, 0., False, False, {}
    
    def no_op_action(self):
        return 0

class AssembleStepTargetRecorder(SuperMechaComponent):
    def __init__(self,
        observation_component,
        assemble_step_component,
        phase_component,
    ):
        self.observation_component = observation_component
        self.assemble_step_component = assemble_step_component
        self.phase_component = phase_component
        self.observation_space = self.observation_component.observation_space
    
    def observe(self):
        if self.assemble_step_component.current_step == len(self.observations):
            self.observations.append(self.observation_component.observation)
        current_step = self.assemble_step_component.current_step
        current_step = max(0, current_step)
        current_step = min(len(self.observations)-1, current_step)
        o = self.observations[current_step]
        #if self.phase_component.phase == 0:
        #    o = map_hierarchies(numpy.zeros_like, o)
        return o
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.observations = []
        o = self.observe()
        return o, {}
    
    def step(self, action):
        o = self.observe()
        return o, 0., False, False, {}
