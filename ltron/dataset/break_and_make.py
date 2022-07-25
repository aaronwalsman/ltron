import random
import os

import numpy

import tqdm

from splendor.contexts import egl

import ltron.settings as settings
from ltron.hierarchy import stack_numpy_hierarchies
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnv, BreakAndMakeEnvConfig)
from ltron.plan.roadmap import Roadmap, PlannerTimeoutError
from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.geometry.collision import build_collision_map

class BreakAndMakeEpisodeConfig(BreakAndMakeEnvConfig):
    episodes_per_model = 1
    dataset = 'random_construction_6_6'
    collection = 'random_construction_6_6'
    split = 'train_2'
    
    target_steps_per_view_change = 2.
    
    error_handling = 'count'
    
    episode_directory = 'episodes_2'
    
    split_cursor_actions = False
    
    seed = 1234567890
    
    timeout = None
    
    allow_snap_flip = False

def generate_episodes_for_dataset(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeEpisodeConfig.from_commandline()
    
    config.dataset_reset_mode = 'single_pass'
    timeout = config.timeout
    if timeout is None:
        timeout = float('inf')
    
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    
    dataset_info = get_dataset_info(config.dataset)
    shape_ids = dataset_info['shape_ids']
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(
        config.dataset, config.split, config.subset)
    
    episode_path = os.path.join(
        settings.collections[config.collection], config.episode_directory)
    if not os.path.exists(episode_path):
        os.makedirs(episode_path)
    
    #num_models = len(dataset_paths['mpd'])
    #if config.end is None:
    #    end = num_models
    #else:
    #    end = config.end
    
    env = BreakAndMakeEnv(
        config, rank=0, size=1, print_traceback=True)
    
    print('='*80)
    print('Planning Plans')
    iterate = tqdm.tqdm(range(env.components['dataset'].length))
    errors = []
    timeout_i = []
    final_rs = []
    for i in iterate:
        for j in range(config.episodes_per_model):
            #env = BreakAndMakeEnv(
            #    config, rank=i, size=num_models, print_traceback=True)
            first_observation = env.reset()
            
            try:
                t = config.target_steps_per_view_change
                try:
                    o, a, r = plan_break_and_make(
                        env, first_observation, shape_ids, color_ids,
                        target_steps_per_view_change=t,
                        split_cursor_actions=config.split_cursor_actions,
                        allow_snap_flip=config.allow_snap_flip,
                        timeout = timeout,
                    )
                    final_r = r[-1]
                    final_rs.append(final_r)
                except PlannerTimeoutError:
                    timeout_i.append(i)
                    continue
                
                o = stack_numpy_hierarchies(*o)
                a = stack_numpy_hierarchies(*a)
                r = numpy.array(r)
                
                file_name = os.path.basename(dataset_paths['mpd'][i])
                file_name = file_name.replace('.mpd', '_%i.npz'%j)
                file_name = file_name.replace('.ldr', '_%i.npz'%j)
                path = os.path.join(episode_path, file_name)
                episode = {'observations':o, 'actions':a, 'reward':r}
                numpy.savez_compressed(path, episode=episode)
            except KeyboardInterrupt:
                raise
            except:
                if config.error_handling == 'count':
                    errors.append(i)
                else:
                    raise
        
        if len(final_rs):
            avg_final_r = sum(final_rs)/len(final_rs)
        else:
            avg_final_r = 0.
        iterate.set_description('Errors: %i, Timeout: %i, R: %f'%(
            len(errors), len(timeout_i), avg_final_r))
    
    if len(errors):
        print('Errors for items:')
        print(errors)
    else:
        print('No errors')
    
    if len(timeout_i):
        print('Timeout for items:')
        print(timeout_i)

def plan_break_and_make(
    env,
    observation,
    shape_ids,
    color_ids,
    target_steps_per_view_change=2,
    split_cursor_actions=False,
    allow_snap_flip=False,
    timeout=float('inf'),
):
    # get the full and empty assemblies
    full_assembly = observation['table_assembly']
    full_state = env.get_state()
    full_collision_map = build_collision_map(
        env.components['table_scene'].brick_scene)
    
    empty_assembly = {
        'shape' : numpy.zeros_like(full_assembly['shape']),
        'color' : numpy.zeros_like(full_assembly['color']),
        'pose' : numpy.zeros_like(full_assembly['pose']),
        'edges' : numpy.zeros_like(full_assembly['edges']),
    }
    empty_collision_map = {}
    
    # initialize the result lists
    observations = []
    actions = []
    rewards = []
    
    #print('='*80)
    #print('Break')
    
    # compute the break path
    break_roadmap = Roadmap(
        env,
        full_state,
        full_collision_map,
        empty_assembly,
        empty_collision_map,
        shape_ids,
        color_ids,
        target_steps_per_view_change=target_steps_per_view_change,
        split_cursor_actions=split_cursor_actions,
        allow_snap_flip=allow_snap_flip,
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
    
    #print('='*80)
    #print('Make')
    
    # compute the make path
    make_roadmap = Roadmap(
        env,
        empty_state,
        empty_collision_map,
        full_assembly,
        full_collision_map,
        shape_ids,
        color_ids,
        target_steps_per_view_change=target_steps_per_view_change,
        split_cursor_actions=split_cursor_actions,
        allow_snap_flip=allow_snap_flip,
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
