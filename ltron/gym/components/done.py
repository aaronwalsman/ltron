from gymnasium.spaces import Discrete

from supermecha import SuperMechaComponent

class DoneComponent(SuperMechaComponent):
    def __init__(self):
        self.action_space = Discrete(2)
    
    def step(self, action):
        return None, 0., bool(action), False, {}
    
    def no_op_action(self):
        return 0
