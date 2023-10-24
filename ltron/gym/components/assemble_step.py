import numpy

from gymnasium.spaces import Discrete

from steadfast.hierarchy import map_hierarchies

from supermecha import SuperMechaComponent

class AssembleStepComponent(SuperMechaComponent):
    #def __init__(self, phase_component, max_steps_per_assemble_step=None):
    #    self.phase_component = phase_component
    def __init__(self,
        max_steps_per_assemble_step=None,
        truncate=False
    ):
        self.action_space = Discrete(3)
        self.max_steps_per_assemble_step = max_steps_per_assemble_step
        self.truncate = truncate
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.steps_since_assemble_step = 0
        return None, {}
    
    def step(self, action):
        #if not self.phase_component.phase:
        #    self.current_step += action
        #else:
        #    self.current_step -= action
        if action == 1:
            self.current_step += 1
        elif action == 2:
            self.current_step -= 1
        
        self.steps_since_assemble_step += 1
        if action:
            self.steps_since_assemble_step = 0
        
        u = False
        if self.max_steps_per_assemble_step is not None and (
            self.steps_since_assemble_step > self.max_steps_per_assemble_step
        ):
            u = True
        
        if self.truncate and action != 0:
            u = True
        
        return None, 0., False, u, {}
    
    def no_op_action(self):
        return 0

class AssembleStepTargetRecorder(SuperMechaComponent):
    def __init__(self,
        observation_component,
        assemble_step_component,
        phase_component,
        zero_phase_zero=False,
        zero_out_of_bounds=False,
    ):
        self.observation_component = observation_component
        self.assemble_step_component = assemble_step_component
        self.phase_component = phase_component
        self.observation_space = self.observation_component.observation_space
        self.zero_phase_zero = zero_phase_zero
        self.zero_out_of_bounds = zero_out_of_bounds
    
    def observe(self):
        if self.assemble_step_component.current_step == len(self.observations):
            self.observations.append(self.observation_component.observation)
        current_step = self.assemble_step_component.current_step
        if not self.zero_out_of_bounds:
            current_step = max(0, current_step)
            current_step = min(len(self.observations)-1, current_step)
        if current_step < 0 or current_step >= len(self.observations):
            o = map_hierarchies(numpy.zeros_like, self.observations[0])
        else:
            o = self.observations[current_step]
        #if (self.phase_component is not None and
        #    self.phase_component.phase == 0 and
        #    self.zero_phase_zero
        #):
        if self.zero_phase_zero:
            if self.phase_component is None:
                o = map_hierarchies(numpy.zeros_like, o)
            elif self.phase_component.phase == 0:
                o = map_hierarchies(numpy.zeros_like, o)
        return o
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.observations = []
        o = self.observe()
        return o, {}
    
    def step(self, action):
        o = self.observe()
        return o, 0., False, False, {}
