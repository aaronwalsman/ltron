import math
from collections import OrderedDict

import gym
import gym.spaces as spaces

import splendor.contexts.glut as glut

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent, SnapRenderComponent)
from ltron.gym.components.disassembly import PixelDisassemblyComponent
from ltron.gym.components.rotation import RotationAroundSnap
from ltron.gym.components.pick_and_place import PickAndPlace
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.reassembly import Reassembly


def reassembly_template_action():
    return {
        'viewpoint' : 0,
        
        'disassembly' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
        },
        
        'rotate' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
        },
        
        'pick_and_place' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
            'place':(0,0),
        },
        
        'reassembly' : {
            'start':False,
        },
    }

class SimplifiedReassemblyWrapper(gym.Env):
    def __init__(*args, **kwargs):
        self.env = reassembly_env(*args, **kwargs)
        
        # setup action space
        render_component = self.env.components['color_render']
        height = render_component.height
        width = render_component.width
        num_modes = (
            6 + # camera motion
            1 + # disassembly
            1 + # rotate
            1 + # pick and place
            1   # start disassembly
        )
        self.action_space = spaces.MultiDiscrete(
            num_modes, 2, 2, height, width, height, width)
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)
    
    def reset(self):
        observation = self.env.reset()
        observation = self.convert_observation(observation)
        return observation
    
    def step(self, action):
        action = self.convert_action(action)
        observation, reward, terminal, info = self.env.step(action)
        observation = self.convert_observation(observation)
        return observation, reward, terminal, info
    
    def convert_observation(self, observation):
        return observation['color_render']
    
    def convert_action(self, action):
        mode, polarity, direction, pick_y, pick_x, place_y, place_x = action
        dict_action = {}
        
        # viewpoint
        if mode < 6:
            viewpoint_action = mode + 1
        else:
            viewpoint_action = 0
        dict_action['viewpoint'] = viewpoint_action
        
        # disassembly
        dict_action['disassembly'] = {
            'activate':(mode == 6),
            'polarity':polarity,
            'direction':direction,
            'pick':numpy.array(pick_y, pick_x),
        }
        
        # rotate
        dict_action['rotate'] = {
            'activate':(mode == 7),
            'polarity':polarity,
            'direction':direction,
            'pick':(pick_y, pick_x),
        }
        
        # pick and place
        dict_action['pick_and_place'] = {
            'activate':(mode == 8),
            'polarity':polarity,
            'direction':direction,
            'pick':(pick_y, pick_x),
            'place':(place_y, place_x),
        }
        
        # reassembly
        dict_action['reassembly'] = {
            'start':(mode == 9),
        }
        
        return dict_action

class InteractiveReassemblyEnv:
    def __init__(self, **kwargs):
        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)
        render_args = {
            'opengl_mode':'glut',
            'window_width':width,
            'window_height':height,
            'load_scene':'grey_cube',
        }
        self.env = reassembly_env(render_args=render_args, **kwargs)
        
        self.scene = self.env.components['scene'].brick_scene
        self.window = self.scene.render_environment.window
        self.height = self.env.components['color_render'].height
        self.width = self.env.components['color_render'].width
        self.map_height = self.env.components['pos_snap_render'].height
        self.map_width = self.env.components['pos_snap_render'].width
        assert self.height % self.map_height == 0
        assert self.width % self.map_width == 0
        self.height_scale = self.height // self.map_height
        self.width_scale = self.width // self.map_width
        
        self.window.register_callbacks(
            glutDisplayFunc = self.render,
            glutIdleFunc = self.render,
            glutKeyboardFunc = self.key_press,
            glutKeyboardUpFunc = self.key_release,
            glutSpecialFunc = self.special_key,
        )
        
        self.polarity = '+'
        self.direction = 'pull'
        self.render_mode = 'color'
        self.pick = (0,0)
    
    def render(self):
        self.window.enable_window()
        if self.render_mode == 'color':
            self.scene.color_render(flip_y=False)
        elif self.render_mode == 'mask':
            self.scene.mask_render(flip_y=False)
        elif self.render_mode == 'snap':
            snap_instances = self.scene.get_snaps(polarity=self.polarity)
            snap_names = self.scene.get_snap_names(snap_instances)
            self.scene.snap_render_instance_id(snap_names, flip_y=False)
    
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        print('Reward: %f'%reward)
    
    def key_press(self, key, x, y):
        if key == b'r':
            observation = self.env.reset()
        
        elif key == b'd':
            print('Disassemble: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['disassembly']['activate'] = True
            action['disassembly']['polarity'] = '-+'.index(self.polarity)
            action['disassembly']['direction'] = (
                ('pull', 'push').index(self.direction))
            action['disassembly']['pick'] = (yy,xx)
            self.step(action)
        
        elif key == b'p':
            print('Pick: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            self.pick = (yy, xx)
        
        elif key == b'[':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':0,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b']':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':1,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b'm':
            if self.render_mode == 'mask':
                self.render_mode = 'color'
            else:
                self.render_mode = 'mask'
        
        elif key == b's':
            if self.render_mode == 'snap':
                self.render_mode = 'color'
            else:
                self.render_mode = 'snap'
        
        elif key == b'-':
            print('Polarity: "-"')
            self.polarity = '-'
        
        elif key == b'+':
            print('Polarity: "+"')
            self.polarity = '+'
        
        elif key == b'<':
            print('Direction: "pull"')
            self.direction = 'pull'
        
        elif key == b'>':
            print('Direction: "push"')
            self.direction = 'push'
        
        elif key == b'|':
            if not self.env.components['reassembly'].reassembling:
                print('Switching to Reassembly')
                action = reassembly_template_action()
                action['reassembly']['start'] = 1
                self.step(action)
            else:
                print('Already Reassembling')
    
    def key_release(self, key, x, y):
        if key == b'p':
            print('Place: %i, %i'%(x,y))
            pick_y, pick_x = self.pick
            place_x = x // self.width_scale
            place_y = y // self.height_scale
            action = reassembly_template_action()
            action['pick_and_place'] = {
                'activate':True,
                'polarity':'-+'.index(self.polarity),
                'direction':('pull', 'push').index(self.direction),
                'pick':(pick_y, pick_x),
                'place':(place_y, place_x),
            }
            self.step(action)
    
    def special_key(self, key, x, y):
        if key == glut.GLUT.GLUT_KEY_LEFT:
            print('Camera Left')
            action = reassembly_template_action()
            action['viewpoint'] = 1
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_RIGHT:
            print('Camera Right')
            action = reassembly_template_action()
            action['viewpoint'] = 2
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_UP:
            print('Camera Up')
            action = reassembly_template_action()
            action['viewpoint'] = 3
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_DOWN:
            print('Camera Down')
            action = reassembly_template_action()
            action['viewpoint'] = 4
            self.step(action)
    
    def get_snap_under_mouse(x, y, polarity):
        if polarity == '-':
            render_component = self.env.components['neg_snap_render']
        elif polarity == '+':
            render_component = self.env.components['pos_snap_render']
        instance_id, snap_id = render_component.observation[y, x]
        return instance_id, snap_id
    
    def start(self):
        glut.start_main_loop()
    

def reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    image_width=256,
    image_height=256,
    map_width=64,
    map_height=64,
    dataset_reset_mode='uniform',
    render_args=None,
    randomize_colors=True,
    check_collisions=True,
    print_traceback=True,
):
    components = OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    
    # scene
    components['scene'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=[0],
        render_args=render_args,
        track_snaps=True,
        collision_checker=check_collisions,
    )
    
    # viewpoint
    components['viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['scene'],
        azimuth_steps=8,
        elevation_range=[math.radians(-30), math.radians(30)],
        elevation_steps=2,
        distance_range=[250, 250],
        distance_steps=1,
        azimuth_offset=math.radians(45.),
        aspect_ratio=image_width/image_height,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['all_colors'],
            components['scene'],
            randomize_frequency='scene',
        )
    
    # utility rendering components
    pos_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='+')
    neg_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='-')
    
    # action spaces
    components['disassembly'] = PixelDisassemblyComponent(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    components['rotate'] = RotationAroundSnap(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    components['pick_and_place'] = PickAndPlace(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    
    # reassembly
    components['reassembly'] = Reassembly(
        components['scene'])
    
    # color render
    components['color_render'] = ColorRenderComponent(
        image_width, image_height, components['scene'], anti_alias=True)
    
    # snap render
    components['pos_snap_render'] = pos_snap_render
    components['neg_snap_render'] = neg_snap_render
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env

if __name__ == '__main__':
    interactive_env = InteractiveReassemblyEnv(
        dataset='random_six',
        split='simple_single',
        subset=1)
    interactive_env.start()
