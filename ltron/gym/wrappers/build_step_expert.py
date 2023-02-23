from copy import deepcopy

from gymnasium import make, ObservationWrapper

def wrapped_build_step_expert(env_name, **kwargs):
    return BuildStepExpert(make(env_name, **kwargs))

class BuildStepExpert(ObservationWrapper):
    def __init__(self, env, max_instructions=16):
        super().__init__(env)
        
        '''
        options
        1. do the thing where we make the observation space a tuple version
            of the full action space
        2. shrink it down to just what we need to convey what we want
            2.1. would need mode, click, release, rotate angle
        '''
        observation_space = deep_copy(self.env.observation_space)
        observation_space['expert'] = batch_space(
            self.env.action_space, max_instructions)
        observation_space['num_expert_actions'] = Discrete(max_instructions)
    
    def observation(self, observation):
        # get assemblies
        current_assembly = observation['assembly']
        target_assembly = observation['target_assembly']
        
        # get the current matches (assumes current and target are aligned)
        matches, offset = find_matches_under_transform(
            current_assembly, target_assembly, numpy.eye(4))
        (ct_connected,
         tc_connected,
         ct_disconnected,
         tc_disconnected,
         fp,
         fn) = compute_misaligned(
            current_assembly, target_assembly, matches)
        
        # this only connects bricks, and will not remove or insert them
        assert len(fp) == 0
        assert len(fn) == 0
        
        num_connected = len(ct_connected)
        num_disconnected = len(ct_disconnected)
        num_misplaced = num_connected + num_disconnected
        
        assert num_misplaced <= 1
   
        # three cases
        # 1. no misaligned -> DONE
        # 2. disconnected -> pick_and_place
        # 3. connected -> rotate
        
        if num_misplaced == 0:
            actions = self.done_actions()
        
        elif num_connected == 1:
            actions = self.rotate_actions(ct_connected, tc_connected)
        
        elif num_disconnected == 2:
            actions = self.pick_and_place_actions(
                ct_disconnected, tc_disconnected)
    
    def done_actions(self):
        action = self.env.no_op_action()
        mode_space = self.env.action_space['action_primitives']['mode']
        done_index = mode_space.names.index('done')
        action['action_primitives']['mode'] = done_index
        action['action_primitives']['done'] = 1
        
        return [action]
    
    def rotate_actions(self, ct_connected, tc_connected):
        pass
    
    def pick_and_place_actions(self, ct_disconnected, tc_disconnected):
        pass
