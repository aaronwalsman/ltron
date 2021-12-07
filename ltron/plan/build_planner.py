from ltron.plan.planner import Planner

class BuildPlanner(Planner):
    def __init__(
        self,
        env,
        start_assembly,
        start_camera_pose,
        goal_assembly,
        camera_motion_penalty=-0.1
    ):
        super(BuildPlanner, self).__init__()
        self.env = env
        self.start_state = self.make_state(start_assembly, start_camera_pose)
        self.goal_assembly = goal_assembly
        self.camera_move_penalty=camera_motion_penalty
        
        matches, offset = match_assemblies(start_assembly, goal_assembly)
    
    def make_state(self, assembly, camera_pose):
        return (
            self.make_frozen_assembly(assembly),
            EpsilonArray(camera_pose),
        )
    
    def make_frozen_assembly(self, assembly):
        return frozenset(numpy.where(assembly['class_id'])[0])
    
    def sample_initial_state(self):
        return self.start_state()
    
    def action_space(self, state):
        assembly, camera_pose = state
        
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
        
        predictions = [1./len(actions) for _ in actions]
        return actions, successors, predictions
    
    def terminal(self, path_states, path_actions):
        assembly, camera_pose, reassembling = path_states[-1]
        return assembly == self.goal_assembly and reassembling
