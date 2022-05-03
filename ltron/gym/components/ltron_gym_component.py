class LtronGymComponent:
    def reset(self):
        return None
    
    def step(self, action):
        return None, 0, False, {}
    
    def render(self):
        pass
    
    def get_state(self):
        return None
    
    def set_state(self, state):
        pass
    
    def close(self):
        pass
