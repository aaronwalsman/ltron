from ltron.matching import match_assemblies, compute_misaligned

from supermecha import SuperMechaComponent

class BuildScore(SuperMechaComponent):
    def __init__(self,
        target_assembly_component,
        current_assembly_component,
    ):
        self.target_assembly_component = target_assembly_component
        self.current_assembly_component = current_assembly_component
    
    def compute_error(self):
        target_assembly = self.target_assembly_component.observation
        current_assembly = self.current_assembly_component.observation
        
        matches, offset = match_assemblies(current_assembly, target_assembly)
        (connected_misaligned_current,
         connected_misaligned_target,
         disconnected_misaligned_current,
         disconnected_misaligned_target,
         false_positives,
         false_negatives) = compute_misaligned(
            current_assembly, target_assembly, matches)
        
        error = (
            len(connected_misaligned_current) + 
            len(disconnected_misaligned_current) * 2 +
            len(false_positives) +
            len(false_negatives) * 3
        )
        
        return error
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_error = self.compute_error()
        return None, {}
    
    def step(self, action):
        error = self.compute_error()
        improvement = self.previous_error - error
        self.previous_error = error
        return None, improvement, False, False, {}
    
    def get_state(self, state):
        return self.previous_error
    
    def set_state(self, state):
        self.previous_error = state
        return None, {}
