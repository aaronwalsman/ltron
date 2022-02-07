import random
import os

import numpy

import tqdm

import ltron.settings as settings
from ltron.hierarchy import stack_numpy_hierarchies
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnv, BreakAndMakeEnvConfig)
from ltron.plan.roadmap import Roadmap
from ltron.dataset.paths import get_dataset_info, get_dataset_paths

class BreakAndMakeEpisodeConfig(BreakAndMakeEnvConfig):
    episodes_per_model = 1
    dataset = 'random_construction_6_6'
    collection = 'random_construction_6_6'
    split = 'train'
    subset = None
    start = 0
    end = None
    
    seed = 1234567890

def generate_episodes_for_dataset(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeEpisodeConfig.from_commandline()
    
    config.dataset_reset_mode = 'single_pass'
    
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    
    dataset_info = get_dataset_info(config.dataset)
    shape_ids = dataset_info['shape_ids']
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(
        config.dataset, config.split, config.subset)
    
    episode_path = os.path.join(
        settings.collections[config.collection], 'episodes')
    if not os.path.exists(episode_path):
        os.makedirs(episode_path)
    
    num_models = len(dataset_paths['mpd'])
    if config.end is None:
        end = num_models
    else:
        end = config.end
    
    print('='*80)
    print('Planning Plans %i-%i'%(config.start, end))
    iterate = tqdm.tqdm(range(config.start, end))
    errors = []
    for i in iterate:
        for j in range(config.episodes_per_model):
            env = BreakAndMakeEnv(
                config, rank=i, size=num_models, print_traceback=True)
            first_observation = env.reset()
            
            try:
                o, a, r = plan_break_and_make(
                    env, first_observation, shape_ids, color_ids)
            except:
                errors.append(i)
            
            o = stack_numpy_hierarchies(*o)
            a = stack_numpy_hierarchies(*a)
            r = numpy.array(r)
            
            file_name = os.path.basename(dataset_paths['mpd'][i])
            file_name = file_name.replace('.mpd', '_%i.npz'%j)
            file_name = file_name.replace('.ldr', '_%i.npz'%j)
            path = os.path.join(episode_path, file_name)
            episode = {'observations':o, 'actions':a, 'reward':r}
            numpy.savez_compressed(path, episode=episode)
        
        iterate.set_description('Errors: %i'%len(errors))
    
    if len(errors:
        print('Errors for items:')
        print(errors)
    else:
        print('No errors')

def plan_break_and_make(
    env,
    observation,
    shape_ids,
    color_ids,
    target_steps_per_view_change=2,
    timeout=float('inf'),
):
    # get the full and empty assemblies
    full_assembly = observation['table_assembly']
    full_state = env.get_state()
    
    empty_assembly = {
        'shape' : numpy.zeros_like(full_assembly['shape']),
        'color' : numpy.zeros_like(full_assembly['color']),
        'pose' : numpy.zeros_like(full_assembly['pose']),
        'edges' : numpy.zeros_like(full_assembly['edges']),
    }
    
    # initialize the result lists
    observations = []
    actions = []
    rewards = []
    
    # compute the break path
    break_roadmap = Roadmap(
        env,
        full_state,
        empty_assembly,
        shape_ids,
        color_ids,
        target_steps_per_view_change=target_steps_per_view_change,
    )
    break_path = break_roadmap.plan(timeout=timeout)
    o, a, r = break_roadmap.get_observation_action_reward_seq(
        break_path,
        include_last_observation=True,
    )
    
    # update the result lists
    observations.extend(o)
    actions.extend(a)
    rewards.extend(r)
    
    # switch phase
    action = env.no_op_action()
    action['phase'] = 1
    observation, reward, terminal, info = env.step(action)
    
    # get the empty (post phase switch) state
    empty_state = env.get_state()
    
    # update the result lists
    # do not add the observation because it will come from the make sequence
    actions.append(action)
    rewards.append(reward)
    
    # compute the make path
    make_roadmap = Roadmap(
        env,
        empty_state,
        full_assembly,
        shape_ids,
        color_ids,
        target_steps_per_view_change=target_steps_per_view_change,
    )
    make_path = make_roadmap.plan(timeout=timeout)
    o, a, r = make_roadmap.get_observation_action_reward_seq(
        make_path,
        include_last_observation=True,
    )
    
    # update the result lists
    observations.extend(o)
    actions.extend(a)
    rewards.extend(r)
    
    # finish
    action = env.no_op_action()
    action['phase'] = 2
    observation, reward, terminal, info = env.step(action)
    
    # update the result lists
    actions.append(action)
    rewards.append(reward)
    
    return observations, actions, rewards
