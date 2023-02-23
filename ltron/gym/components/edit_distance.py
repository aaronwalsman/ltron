from ltron.score import edit_distance

from supermecha import SuperMechaComponent

class EditDistance(SuperMechaComponent):
    def __init__(self,
        target_assembly_component,
        current_assembly_component,
        false_positive_penalty=1,
        false_negative_penalty=2,
        pose_penalty=1,
    ):
        self.target_assembly_component = target_assembly_component
        self.current_assembly_component = current_assembly_component
        self.miss_a_penalty = miss_a_penalty
        self.miss_b_penalty = miss_b_penalty
        self.pose_penalty = pose_penalty
    
    def compute_edit_distance(self):
        target_assembly = self.target_assembly_component.observe()
        current_assembly = self.current_assembly_component.observe()
        
        edit_distance, _ = edit_distance(
            current_assembly,
            target_assembly,
            self.a_penalty=self.miss_a_penalty,
            miss_b_penalty=self.miss_b_penalty,
            pose_penalty=self.pose_penalty
        )
        
        return edit_distance
    
    def reset(self):
        self.previous_edit_distance = self.compute_edit_distance()
        return None, {}
    
    def step(self, action):
        edit_distance = self.compute_edit_distance()
        improvement = self.previous_edit_distance - edit_distance
        self.previous_edit_distance = edit_distance
        return None, improvement, False, False, {}
    
    def get_state(self, state):
        return self.previous_edit_distance
    
    def set_state(self, state):
        self.previous_edit_distance = state
        return None, {}
