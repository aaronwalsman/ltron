import numpy

import tqdm

from ltron.hierarchy import stack_numpy_hierarchies
from ltron.gym.rollout_storage import RolloutStorage

def rollout(
    episodes,
    env,
    actor_fn,
    initial_memory=None,
    store_observations=True,
    store_actions=True,
    store_distributions=True,
    store_rewards=True,
    rollout_mode='sample',
):
    
    # initialize storage for observations, actions, rewards and distributions
    b = env.num_envs
    storage = {}
    if store_observations:
        storage['observation'] = RolloutStorage(b)
    if store_actions:
        storage['action'] = RolloutStorage(b)
    if store_distributions:
        storage['distribution'] = RolloutStorage(b)
    if store_rewards:
        storage['reward'] = RolloutStorage(b)
    
    assert len(storage)
    first_key, first_storage = next(iter(storage.items()))
    
    # reset
    observation = env.reset()
    
    terminal = numpy.ones(b, dtype=numpy.bool)
    reward = numpy.zeros(env.num_envs)
    
    memory = initial_memory
    
    progress = tqdm.tqdm(total=episodes)
    with progress:
        while first_storage.num_finished_seqs() < episodes:
            # start new sequences if necessary
            if store_observations:
                storage['observation'].start_new_seqs(terminal)
            if store_actions:
                storage['action'].start_new_seqs(terminal)
            if store_distributions:
                storage['distribution'].start_new_seqs(terminal)
            if store_rewards:
                storage['reward'].start_new_seqs(terminal)
            
            # add latest observation to storage
            if store_observations:
                storage['observation'].append_batch(observation=observation)
            
            # compute actions
            observation = stack_numpy_hierarchies(observation)
            
            distribution, memory = actor_fn(
                observation, terminal, memory)
            #s, b, n = distribution.shape
            #distribution = distribution.reshape(b, n)
            b, n = distribution.shape
            if rollout_mode == 'sample':
                actions = [
                    numpy.random.choice(range(n), p=d) for d in distribution
                ]
            elif rollout_mode == 'max':
                actions = numpy.argmax(distribution, axis=-1).tolist()
            
            # step
            observation, reward, terminal, info = env.step(actions)
            
            # storage
            if store_actions:
                a = stack_numpy_hierarchies(*actions)
                storage['action'].append_batch(action=a)
            
            if store_distributions:
                storage['distribution'].append_batch(distribution=distribution)
            
            if store_rewards:
                storage['reward'].append_batch(reward=reward)
            
            # progress
            update = first_storage.num_finished_seqs() - progress.n
            progress.update(update)
        
        progress.n = episodes
        progress.refresh()
    
    combined_storage = first_storage
    for key, s in storage.items():
        if key == first_key:
            continue
        
        combined_storage = combined_storage | s
    
    return combined_storage
