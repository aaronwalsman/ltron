class BrickEnvComponent:
    def update_observation_space(self, observation_space):
        pass
    
    def update_action_space(self, action_space):
        pass
    
    def initialize_state(self, state):
        pass
    
    def reset_state(self, state):
        pass
    
    def update_state(self, state, action):
        pass
    
    def compute_observation(self, state, observation):
        pass
    
    def compute_reward(self, state, action):
        return 0.
    
    def check_terminal(self, state):
        return False
    
    def update_info(self, state, info):
        pass
    
    def render(self, state):
        pass
    
    def cleanup(self, state):
        pass
