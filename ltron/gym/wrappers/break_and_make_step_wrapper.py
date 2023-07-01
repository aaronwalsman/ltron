from gymnasium import Wrapper

class BreakAndMakeStepWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        observation_space = deepcopy(self.env.observation_space)
        action_space = deepcopy(self.env.action_space)
        action_space['step_done'] = Discrete(2)
        del(action_space['phase'])
