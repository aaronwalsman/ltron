import numpy

from splendor.image import save_image

from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnv, BreakAndMakeEnvConfig
)

class LiveConfig(BreakAndMakeEnvConfig):
    episodes = 10

def visualize_obs(obs):
    table_color_render = obs['table_color_render']
    hand_color_render = obs['hand_color_render']
    
    initial_table_render = obs['initial_table_color_render']
    
    table_small_render = table_color_render.reshape((64,4,64,4,3))
    table_small_render = numpy.mean(table_small_render, axis=1)
    table_small_render = numpy.mean(table_small_render, axis=2)
    table_small_render = table_small_render.astype(numpy.uint8)
    
    hand_small_render = hand_color_render.reshape((24,4,24,4,3))
    hand_small_render = numpy.mean(hand_small_render, axis=1)
    hand_small_render = numpy.mean(hand_small_render, axis=2)
    hand_small_render = hand_small_render.astype(numpy.uint8)
    
    save_image(initial_table_render, './initial_table.png')
    save_image(table_color_render, './table.png')
    save_image(table_small_render, './table_small.png')
    save_image(hand_color_render, './hand.png')
    save_image(hand_small_render, './hand_small.png')

def get_action(env):
    print('-'*80)
    valid_action = False
    while not valid_action:
        tokens = input('Action : ').split()
        clean_tokens = []
        for t in tokens:
            try:
                t = int(t)
            except ValueError:
                pass
            clean_tokens.append(t)
           
        try:
            action = env.action_space.ravel(*clean_tokens)
            valid_action = True
        except:
            print('Invalid Action')
    
    return clean_tokens, action

def main():
    config = LiveConfig.from_commandline()

    env = BreakAndMakeEnv(config)

    obs = env.reset()
    terminal = False
    
    for e in range(1, config.episodes+1):
        print('='*80)
        print('Episode: %i'%e)
        while not terminal:
            visualize_obs(obs)
            clean_tokens, action = get_action(env)
            print('Taking Action', action, tuple(clean_tokens))
            obs, reward, terminal, _ = env.step(action)
        
        print('-'*80)
        print('Reward for Episode %i: %f'%(e, reward))
        obs = env.reset()
        terminal = False
