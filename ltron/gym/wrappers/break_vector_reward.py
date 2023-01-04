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
