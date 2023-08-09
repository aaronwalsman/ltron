import random
import os

import numpy

import tqdm

import ltron.settings as settings
from ltron.hierarchy import stack_numpy_hierarchies
from ltron.gym.envs.edit_env import EditEnv, EditEnvConfig
from ltron.plan.edge_planner import plan_add_nth_brick
from ltron.dataset.paths import get_dataset_info, get_dataset_paths

class EditEpisodeConfig(EditEnvConfig):
    episodes_per_model = 1
    dataset = 'random_construction_6_6'
    collection = 'random_construction_6_6'
    split = 'train'
    
    episode_directory = 'edit_2'
    
    split_cursor_actions = True
    
    seed = 1234567890

def generate_episodes_for_dataset(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = EditEpisodeConfig.from_commandline()
    
    config.dataset_reset_mode = 'single_pass'
    
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    
    dataset_info = get_dataset_info(config.dataset)
    shape_ids = {v:k for k,v in dataset_info['shape_ids'].items()}
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(
        config.dataset, config.split, config.subset)
    
    episode_path = os.path.join(
        settings.collections[config.collection], config.episode_directory)
    if not os.path.exists(episode_path):
        os.makedirs(episode_path)
    
    env = EditEnv(config, rank=0, size=1, print_traceback=True)
    
    print('='*80)
    print('Planning Plans')
    iterate = tqdm.tqdm(range(env.components['dataset'].length))
    for i in iterate:
        for j in range(config.episodes_per_model):
            first_observation = env.reset()
            initial_assembly = first_observation['initial_table_assembly']
            current_assembly = first_observation['table_assembly']
            instance = numpy.where(
                initial_assembly['shape'] != current_assembly['shape'])[0][0]
            goal_to_wip = {
                i:i for i in numpy.nonzero(initial_assembly['shape'])[0]
                if i != instance
            }
            o, a, c, r = plan_add_nth_brick(
                env,
                initial_assembly,
                instance,
                first_observation,
                goal_to_wip,
                shape_ids,
                split_cursor_actions=config.split_cursor_actions,
                allow_snap_flip=True,
            )
            action = env.no_op_action()
            action['phase'] = 1
            observation, reward, terminal, info = env.step(action)
            #o.append(observation) # don't need the last one
            a.append(action)
            th, tw = observation['table_pos_snap_render'].shape[:2]
            hh, hw = observation['hand_pos_snap_render'].shape[:2]
            table_zero = numpy.zeros((th, tw, 2), dtype=bool)
            hand_zero = numpy.zeros((hh, hw, 2), dtype=bool)
            c.append((table_zero, hand_zero))
            r.append(reward)
            
            o = stack_numpy_hierarchies(*o)
            a = stack_numpy_hierarchies(*a)
            c = stack_numpy_hierarchies(*c)
            r = numpy.array(r)
            
            file_name = os.path.basename(dataset_paths['mpd'][i])
            file_name = file_name.replace('.mpd', '_%i.npz'%j)
            file_name = file_name.replace('.ldr', '_%i.npz'%j)
            path = os.path.join(episode_path, file_name)
            episode = {
                'observations':o, 'actions':a, 'click_maps':c, 'reward':r}
            numpy.savez_compressed(path, episode=episode)


if __name__ == '__main__':
    generate_episodes_for_dataset()
