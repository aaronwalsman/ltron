class LtronGymComponent:
    '''
    def update_observation_space(self, observation_space):
        pass
    
    def update_action_space(self, action_space):
        pass
    
    def initialize_state(self, state):
        pass
    '''
    
    def reset(self):
        return None
    
    def step(self, action):
        return None, 0, False, {}
    
    '''
    def update_state(self, action):
        pass
    
    def compute_observation(self):
        pass
    
    def compute_reward(self, action):
        return 0.
    
    def check_terminal(self):
        return False
    
    def update_info(self, info):
        pass
    '''
    def render(self):
        pass
    
    def get_state(self):
        return None
    
    def set_state(self, state):
        pass
    
    def close(self):
        pass
