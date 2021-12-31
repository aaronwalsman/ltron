import random
import os

import numpy

import tqdm

from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete, Dict

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import stack_numpy_hierarchies
from ltron.score import f1
from ltron.gym.envs.ltron_env import traceback_decorator

class BlocksEnvConfig(Config):
    table_height = 64
    table_width = 64
    hand_height = 24
    hand_width = 24
    
    render_upsample = 4
    render_gap = 1
    
    num_blocks = 8
    
    block_shapes = ((2,2), (4,2), (2,4), (4,4), (8,2), (2,8))
    block_colors = (
        ( 170,   0,   0),
        (   0, 170,   0),
        (   0,   0, 170),
        ( 170,   0, 170),
        ( 255, 140,   0),
        ( 255, 255,   0),
    )
    
    def set_dependents(self):
        # table image height, width, shape
        self.table_image_height = self.table_height * self.render_upsample
        self.table_image_width = self.table_width * self.render_upsample
        self.table_image_shape = (
            self.table_image_height, self.table_image_width, 3)
        self.table_tile_shape = (
            self.table_image_height // 16, self.table_image_width // 16)
        
        # hand image height, width, shape
        self.hand_image_height = self.hand_height * self.render_upsample
        self.hand_image_width = self.hand_width * self.render_upsample
        self.hand_image_shape = (
            self.hand_image_height, self.hand_image_width, 3)
        self.hand_tile_shape = (
            self.hand_image_height // 16, self.hand_image_width // 16)
        
        # num shapes/colors
        self.num_shapes = len(self.block_shapes)
        self.num_colors = len(self.block_colors)
        
        # max steps
        self.max_steps_per_episode = self.num_blocks * 4

def make_random_assembly(config):
    assembly = []
    for b in range(config.num_blocks):
        bh, bw = random.choice(config.block_shapes)
        bc = random.choice(config.block_colors)
        y = random.randint(0, config.table_height-bh)
        x = random.randint(0, config.table_width-bw)
        assembly.append(((bh, bw), bc, (y,x)))
    
    return assembly

def block_region(bh, bw, y, x, h, w):
    low = int(min(max(y, 0), h))
    high = int(min(max(y+bh, 0), h))
    left = int(min(max(x, 0), w))
    right = int(min(max(x+bw, 0), w))
    return low, high, left, right

def block_image_region(bh, bw, y, x, h, w, upsample, gap):
    low = int(min(max(y * upsample + gap, 0), h))
    high = int(min(max((y+bh) * upsample - gap, 0), h))
    left = int(min(max(x * upsample + gap, 0), w))
    right = int(min(max((x+bw) * upsample - gap, 0), w))
    return low, high, left, right

def render_state(state, config):
    assembly, hand_block = state
    
    # render table map and image
    table_map = numpy.zeros(
        (config.table_height, config.table_width), dtype=numpy.long)
    table_image = numpy.zeros(config.table_image_shape, dtype=numpy.uint8)
    table_image[:,:] = (102, 102, 102)
    for i, block in enumerate(assembly):
        if block is not None:
            (bh, bw), bc, (y, x) = block
            
            # render block onto table map
            h, w = config.table_height, config.table_width
            low, high, left, right = block_region(bh, bw, y, x, h, w)
            table_map[low:high, left:right] = i + 1
            
            # render block onto table image
            h, w, _ = config.table_image_shape
            low, high, left, right = block_image_region(
                bh, bw, y, x, h, w, config.render_upsample, config.render_gap)
            table_image[low:high, left:right] = bc
    
    # render hand map and image
    hand_map = numpy.zeros(
        (config.hand_height, config.hand_width), dtype=numpy.long)
    hand_image = numpy.zeros(config.hand_image_shape, dtype=numpy.uint8)
    hand_image[:,:] = (102, 102, 102)
    if hand_block is not None:
        (bh, bw), bc = hand_block
        y = config.hand_height // 2 - bh // 2
        x = config.hand_width // 2 - bw // 2
        
        # render block onto hand map
        h, w = config.hand_height, config.hand_width
        low, high, left, right = block_region(bh, bw, y, x, h, w)
        hand_map[low:high, left:right] = 1
        
        # render block onto hand image
        h, w, _ = config.hand_image_shape
        low, high, left, right = block_image_region(
            bh, bw, y, x, h, w, config.render_upsample, config.render_gap)
        hand_image[low:high, left:right] = bc
    
    return table_map, table_image, hand_map, hand_image

class BlocksEnv(Env):
    def __init__(self, config, rank=0, size=1):
        self.config = config
        
        self.observation_space = Dict({
            'table_render':Box(
                low=0, high=255,
                shape=config.table_image_shape, dtype=numpy.uint8),
            'table_tile_mask':Box(
                low=0, high=1,
                shape=config.table_tile_shape, dtype=numpy.bool),
            'hand_render':Box(
                low=0, high=255,
                shape=config.hand_image_shape, dtype=numpy.uint8),
            'hand_tile_mask':Box(
                low=0, high=1,
                shape=config.hand_tile_shape, dtype=numpy.bool),
            'phase':Discrete(2),
            'step':Discrete(config.num_blocks*3+2),
        })
        
        self.action_space = Dict({
            'table_cursor':MultiDiscrete(
                (config.table_height, config.table_width)),
            'hand_cursor':MultiDiscrete(
                (config.hand_height, config.hand_width)),
            # disassembly, pick and place, pick-up, end phase, end, no-op
            'mode':Discrete(6),
            'color':Discrete(len(config.block_colors)),
            'shape':Discrete(len(config.block_shapes)),
        })
        
        self.episode_step = 0
        self.print_traceback = True
    
    def observe(self):
        self.table_map, table_render, self.hand_map, hand_render = render_state(
            (self.assembly, self.hand_block), self.config)
        
        table_match = table_render != self.previous_table_render
        table_match = table_match.reshape(16, 16, 16, 16, 3)
        table_match = numpy.moveaxis(table_match, 2, 1)
        table_match = table_match.reshape(16, 16, -1)
        table_tile_mask = numpy.any(table_match, axis=-1).reshape(16, 16)
        
        hand_match = hand_render != self.previous_hand_render
        hand_match = hand_match.reshape(6, 16, 6, 16, 3)
        hand_match = numpy.moveaxis(hand_match, 2, 1)
        hand_match = hand_match.reshape(6, 6, -1)
        hand_tile_mask = numpy.any(hand_match, axis=-1).reshape(6, 6)
        
        self.observation = {
            'table_render':table_render,
            'table_tile_mask':table_tile_mask,
            'hand_render':hand_render,
            'hand_tile_mask':hand_tile_mask,
            'phase':self.phase,
            'step':self.episode_step,
        }
        
        self.previous_table_render = table_render
        self.previous_hand_render = hand_render
    
    @traceback_decorator
    def reset(self):
        self.episode_step = 0
        self.phase = 0
        self.target_assembly = make_random_assembly(self.config)
        self.assembly = self.target_assembly[:]
        self.hand_block = None
        self.previous_table_render = 102
        self.previous_hand_render = 102
        self.observe()
        return self.observation
    
    def compute_reward(self):
        if self.phase == 0:
            return 0.
        
        else:
            target = set(self.target_assembly)
            estimate = set(self.assembly) - set([None])
            tp = len(target & estimate)
            fp = len(estimate - target)
            fn = len(target - estimate)
            return f1(tp, fp, fn)
    
    @traceback_decorator
    def step(self, action):
        self.episode_step += 1
        terminal = False
        if action['mode'] == 0:
            y, x = action['table_cursor']
            instance_id = int(self.table_map[y,x])
            if instance_id:
                shape, color, position = self.assembly[instance_id-1]
                self.assembly[instance_id-1] = None
                self.hand_block = shape, color
        
        elif action['mode'] == 1:
            hy, hx = action['hand_cursor']
            ty, tx = action['table_cursor']
            if self.hand_block is not None:
                (bh, bw), color = self.hand_block
                
                if self.hand_map[hy, hx]:
                    th = self.config.table_height
                    tw = self.config.table_width
                    hh = self.config.hand_height
                    hw = self.config.hand_width
                    
                    y = int(ty - (hy + bh//2 - hh//2))
                    x = int(tx - (hx + bw//2 - hw//2))
                    color = tuple(int(c) for c in color)
                    new_block = ((int(bh), int(bw)), color, (y, x))
                    if new_block not in self.assembly:
                        self.assembly.append(new_block)
                        self.hand_block = None
        
        elif action['mode'] == 2:
            shape = self.config.block_shapes[action['shape']]
            color = self.config.block_colors[action['color']]
            self.hand_block = shape, color
        
        elif action['mode'] == 3:
            if self.phase == 0:
                self.phase = 1
                self.assembly = []
                self.hand_block = None
        
        elif action['mode'] == 4:
            self.assembly = []
            self.hand_block = None
            terminal=True
        
        if self.episode_step >= self.config.max_steps_per_episode:
            terminal = True
        
        self.observe()
        reward = self.compute_reward()
        return self.observation, reward, terminal, {}
    
    @staticmethod
    def no_op_action():
        return {
            'table_cursor':numpy.array((0,0)),
            'hand_cursor':numpy.array((0,0)),
            'mode':5,
            'color':0,
            'shape':0,
        }

'''
class BlocksVectorEnvConfig(BlocksEnvConfig):
    envs = 4

def blocks_vector_env(config):
    def constructor():
        return BlocksEnv(config)
    constructors = [constructor for i in range(config.envs)]
    vector_env = AsyncVectorEnv(constructors, context='spawn')
    
    return vector_env
'''

class BlockPlannerConfig(BlocksEnvConfig):
    output_directory = '.'
    num_episodes = 50000

def dump_obs(obs, name):
    save_image(obs['table_render'], '%s_table.png'%name)
    save_image(obs['hand_render'], '%s_hand.png'%name)

def block_planner(config):
    env = BlocksEnv(config)
    for i in tqdm.tqdm(range(config.num_episodes)):
        observations = []
        actions = []
        obs = env.reset()
        observations.append(obs)
        #dump_obs(obs, 0)
        for j, block in enumerate(reversed(env.assembly)):
            # disassemble
            act = env.no_op_action()
            (h,w), c, (y,x) = block
            pick_y = random.randint(y, y+h-1)
            pick_x = random.randint(x, x+w-1)
            act['table_cursor'] = numpy.array((pick_y, pick_x))
            act['mode'] = 0
            actions.append(act)
            obs, reward, terminal, info = env.step(act)
            observations.append(obs)
            #dump_obs(obs, j+1)
        
        act = env.no_op_action()
        act['mode'] = 3
        actions.append(act)
        obs, reward, terminal, info = env.step(act)
        observations.append(obs)
        #dump_obs(obs, j+2)
        
        for k, block in enumerate(env.target_assembly):
            # pick up brick
            act = env.no_op_action()
            (h,w), c, (y,x) = block
            act['mode'] = 2
            act['shape'] = config.block_shapes.index((h,w))
            act['color'] = config.block_colors.index(c)
            actions.append(act)
            obs, reward, terminal, info = env.step(act)
            observations.append(obs)
            #dump_obs(obs, j+2+2*k+1)
            
            # place brick
            act = env.no_op_action()
            act['mode'] = 1
            py = random.randint(0, h-1)
            pick_y = py + config.hand_height//2 - h//2
            px = random.randint(0, w-1)
            pick_x = px + config.hand_width//2 - w//2
            act['hand_cursor'] = numpy.array((pick_y, pick_x))
            place_y = y + py
            place_x = x + px
            act['table_cursor'] = numpy.array((place_y, place_x))
            actions.append(act)
            obs, reward, terminal, info = env.step(act)
            observations.append(obs)
        
        act = env.no_op_action()
        act['mode'] = 4
        actions.append(act)
        obs, reward, terminal, info = env.step(act)
        
        observations = stack_numpy_hierarchies(*observations)
        actions = stack_numpy_hierarchies(*actions)
        
        episode = {'observations':observations, 'actions':actions}
        episode_path = os.path.join(
            config.output_directory, 'episode_%06i.npz'%i)
        numpy.savez_compressed(episode_path, episode=episode)

if __name__ == '__main__':
    config = BlockPlannerConfig()
    config.num_episodes = 50000
    block_planner(config)

