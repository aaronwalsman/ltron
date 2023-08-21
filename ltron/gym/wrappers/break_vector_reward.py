import copy

import numpy

class BreakVectorEnvAssemblyRewardWrapper:
    def __init__(self,
        vector_env,
        current_assembly_component='assembly',
    ):
        self.vector_env = vector_env
        self.current_assembly_component = current_assembly_component
        self.reward_range = (-2., 2.)
    
    def __getattr__(self, attr):
        return getattr(self.vector_env, attr)
    
    '''
    def update_observed_instances(self, observation):
        current_assembly = observation[self.current_assembly_component]
        for i in range(self.vector_env.num_envs):
            visible_instances = set(
                numpy.where(current_assembly['shape'][i])[0])
            self.observed_instances[i] = visible_instances
    '''
    '''
    def reset(self, *args, **kwargs):
        observation, info = self.vector_env.reset(*args, **kwargs)
        self.observed_instances = [
            set() for _ in range(self.vector_env.num_envs)]
        self.update_observed_instances(observation)
        self.target_instances = copy.deepcopy(self.observed_instances)
        
        return observation, info
    '''
    
    def reset(self, *args, **kwargs):
        obs, info = self.vector_env.reset(*args, **kwargs)
        self.previous_instance_count = numpy.sum(
            obs[self.current_assembly_component]['shape'] != 0,
            axis=1,
        )
        #self.mask = numpy.ones(self.vector_env.num_envs, dtype=bool)
        self.initial_instance_count = self.previous_instance_count
        return obs, info
    
    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = self.vector_env.step(*args, **kwargs)
        instance_count = numpy.sum(
            obs[self.current_assembly_component]['shape'] != 0,
            axis=1,
        )
        
        done = term | trunc
        
        diff = self.previous_instance_count - instance_count
        rew += diff * 1./self.initial_instance_count * ~done
        '''
        bonus = diff * 1./self.initial_instance_count #self.mask
        print(bonus)
        #bonus = bonus * 2 - 1./self.initial_instance_count # TEMP
        rew = rew + bonus * ~done
        '''
        self.previous_instance_count = instance_count
        self.initial_instance_count = (
            self.initial_instance_count * ~done +
            instance_count * done
        )
        #self.mask = ~done
        
        rew += term * (self.previous_instance_count == 0)
        rew -= term * (self.previous_instance_count != 0)
        
        #term = term | (instance_count == 0)
        
        return obs, rew, term, trunc, info
        
    '''
    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = self.vector_env.step(*args, **kwargs)
        initial_instances = []
        for i in range(self.vector_env.num_envs):
            initial_instances.append(len(self.observed_instances[i]))
        self.update_observed_instances(obs)
        for i in range(self.vector_env.num_envs):
            final_instances = len(self.observed_instances[i])
            reward_scale = 1. / len(self.target_instances[i])
            rew[i] += (initial_instances[i]-final_instances) * reward_scale
            
            # extra penalty/bonus when the episode terminates non/empty
            if term[i]:
                if initial_instances[i] == 0:
                    rew[i] += 1
                else:
                    rew[i] -= 1
        
        return obs, rew, term, trunc, info
    '''

class BreakVectorEnvRenderRewardWrapper:
    def __init__(self,
        vector_env,
        instance_render_component,
        target_assembly_component,
    ):
        self.vector_env = vector_env
        self.instance_render_component = instance_render_component
        self.target_assembly_component = target_assembly_component
        self.reward_range = (0., 1.)
    
    def __getattr__(self, attr):
        return getattr(self.vector_env, attr)
    
    def update_observed_instances(self, observation):
        instance_map = observation[self.instance_render_component]
        for i in range(self.vector_env.num_envs):
            visible_instances = set(numpy.unique(instance_map[i]))
            self.observed_instances[i] |= visible_instances
    
    def reset(self, *args, **kwargs):
        observation, info = self.vector_env.reset(self, *args, **kwargs)
        target_assembly = observation[self.target_assembly_component]
        self.target_instances = []
        self.observed_instances = []
        for i in range(self.vector_env.num_envs):
            target_instances = set(numpy.where(target_assembly['shape'][i]))
            self.target_instances.append(target_instances)
            self.observed_instances.append(set())
        self.update_observed_instances()

        return observation, info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = self.vector_env.step(*args, **kwargs)
        initial_instances = []
        for i in range(self.vector_env.num_envs):
            initial_instances.append(len(self.observed_instances[i]))
        self.update_observed_instances()
        for i in range(self.vector_env.num_envs):
            final_instances = len(self.observed_instances[i])
            reward_scale = 1. / len(self.target_instances[i])
            rew[i] += (final_instances - initial_instances[i]) * reward_scale
        
        return obs, rew, term, trunc, info
