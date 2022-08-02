import numpy

# make this more general, maybe using the gym registration
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig,
    BreakAndMakeEnv,
)

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron

from ltron.dataset.tar_dataset import generate_tar_dataset

class GenerateEpisodeCollectionConfig(BreakAndMakeEnvConfig):
    total_episodes = 50000
    shards = 1
    save_episode_frequency = 256

    parallel_envs = 4
    async_ltron = True

def generate_episode_collection(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = GenerateEpisodeCollectionConfig.from_commandline()
    
    if config.async_ltron:
        vector_env = async_ltron
    else:
        vector_env = sync_ltron
    env = vector_env(
        config.parallel_envs,
        BreakAndMakeEnv,
        config,
        include_expert=True,
        print_traceback=True,
    )
    
    def actor_fn(observation, terminal, memory):
        b = terminal.shape[0]
        distribution = numpy.zeros((b, env.metadata['action_space'].n))
        for i in range(b):
            expert_actions = list(set(observation['expert'][0,i]))
            expert_actions = [
                a for a in expert_actions
                if a != env.metadata['no_op_action']
            ]
            d = numpy.zeros(env.metadata['action_space'].n)
            d[expert_actions] = 1. / len(expert_actions)
            if numpy.sum(d) < 0.9999:
                print('uh?')
                import pdb
                pdb.set_trace()
            distribution[i] = d
        
        return distribution, None
    
    name = '%s_%s'%(config.dataset, config.split)
    generate_tar_dataset(
        name,
        config.total_episodes,
        shards=config.shards,
        save_episode_frequency=config.save_episode_frequency,
        path='.',
        env=env,
        actor_fn=actor_fn,
    )
