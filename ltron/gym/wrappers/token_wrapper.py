from gymnasium import Wrapper

class TokenWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = Discrete(num_tokens)
        self.action_space = Discrete(num_tokens)
    
    def reset(self):
        self.action_sequence = [0]
        o, i = self.env.reset()
        info = {
            'wrapped_step' : True,
            'wrapped_observation' : o,
            'wrapped_info' : i,
        }
        return 0, info
    
    def step(self, action):
        if action == 0:
            # compute action dictionary from self.action_sequence
            action_dict = SOMETHING(self.action_sequence)
            
            o, r, t, u, i = self.env.step(action_dict)
            info = {
                'wrapped_step' : True,
                'wrapped_observation' : o,
                'wrapped_info' : i,
                'equivalent_actions' : []
            }
            self.action_sequence.clear()
        
        else:
            r = 0.
            t = False
            u = False
            info = {'wrapped_step' : False}
        
        self.action_sequence.append(action)
        
        return action, r, t, u, info
