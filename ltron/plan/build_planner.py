import numpy

from ltron.matching import match_assemblies, match_lookup
from ltron.plan.planner import Planner

class BuildPlanner(Planner):
    def __init__(
        self,
        env,
        start_assembly,
        start_camera,
        goal_assembly,
        camera_motion_penalty=-0.1
    ):
        super(BuildPlanner, self).__init__()
        self.env = env
        self.start_state = self.make_state(start_assembly, start_camera)
        self.goal_assembly = goal_assembly
        self.camera_move_penalty=camera_motion_penalty
        
        matching, offset = match_assemblies(start_assembly, goal_assembly)
        
        # work out a labelling for false negatives here
        (self.start_to_state,
         self.state_to_start,
         self.fp,
         self.fn) = match_lookup(
            matching, start_assembly, goal_assembly)
        m = numpy.max(numpy.where(self.goal_assembly['class']))
        for i, f in enumerate(fp):
            self.start_to_state[f] = m+i
            self.state_to_start[m+i] = f
    
    def make_state(self, assembly, camera):
        return (self.make_frozen_assembly(assembly), camera)
    
    def make_frozen_assembly(self, assembly):
        ip_labels = numpy.where(assembly['class'])[0]
        goal_labels = 
    
    def sample_initial_state(self):
        return self.start_state()
    
    def action_space(self, state):
        assembly, camera_pose = state
        
        if false_positives:
            pass
            # remove something visible
        
        elif false_negatives:
            pass
            # add something to a visible brick
            # need collision map here
        
        '''
        if not reassembling:
            if len(assembly):
                observation = self.set_state(state)
                
                # find bricks to remove
            
            else:
                # switch to reassembly
                action = self.env.no_op_action()
                action['reassembly'] = 1
                actions = [action]
                successors = [(assembly, camera_pose, True)]
        
        else:
            if not len(assembly):
                # find first brick to place
            
            else:
                # find nth brick to add
        '''
        
        predictions = [1./len(actions) for _ in actions]
        return actions, successors, predictions
    
    def terminal(self, path_states, path_actions):
        assembly, camera_pose, reassembling = path_states[-1]
        return assembly == self.goal_assembly and reassembling
