from ltron.matching import match_assemblies, compute_misaligned
from ltron.evaluation import precision_recall, f1

from supermecha import SuperMechaComponent

class DetectObjective(SuperMechaComponent):
    def __init__(self, current_assembly_component):
        self.current_assembly_component = current_assembly_component
        self.action_space = self.current_assembly_component.observation_space
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return None, {}
    
    def step(self, action):
        current_assembly = self.current_assembly_component.observation
        matches, offset = match_assemblies(action, current_assembly)
        (connected_misaligned_action,
         connected_misaligned_current,
         disconnected_misaligned_action,
         disconnected_misaligned_current,
         false_positives,
         false_negatives) = compute_misaligned(
            action, current_assembly, matches)
        fp = len(false_positives)
        fn = len(false_negatives)
        
        p, r = precision_recall(len(matches), fp, fn)
        r = f1(p,r)
        
        return None, r, False, False, {}
    
    def no_op_action(self):
        return self.action_space.empty()
