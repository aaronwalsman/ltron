from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig,
    BreakAndMakeEnv,
)

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron

from ltron.dataset.tar_dataset import generate_tar_dataset

class BreakAndMakeDatasetConfig(BreakAndMakeEnvConfig):
    total_episodes = 50000
    shards = 1
    save_episode_frequency = 256
    
    parallel_envs = 4
    async_ltron = True

def ltron_break_and_make_dataset(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeDatasetConfig.from_commandline()
    
    if config.async_ltron:
        vector_env = async_ltron
    else:
        vector_env = sync_ltron
    env = vector_env(
        config.parallel_envs,
        BreakAndMakeEnv,
        config,
        print_traceback=True,
    )
    
    name = '%s_%s'%(config.dataset, config.split)
    generate_tar_dataset(
        name,
        config.total_episodes,
        shards=config.shards,
        save_episode_frequency=config.save_episode_frequency,
        path='.',
        env=env,
        model=None,
        expert_probability=1.,
        store_distributions=True,
    )
