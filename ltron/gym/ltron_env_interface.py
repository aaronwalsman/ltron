import numpy

import gymnasium as gym

import splendor.contexts.glut as glut
from splendor.image import save_image

from steadfast.hierarchy import hierarchy_getitem

from ltron.constants import SHAPE_CLASS_LABELS, COLOR_CLASS_LABELS
from ltron.gym.envs import (
    FreebuildEnvConfig,
    BreakEnvConfig,
    MakeEnvConfig,
    BreakAndMakeEnvConfig,
)
from ltron.gym.components import ViewpointActions

class LtronInterfaceConfig(
    FreebuildEnvConfig, BreakEnvConfig, MakeEnvConfig, BreakAndMakeEnvConfig,
):
    seed = 1234567890
    env_name = 'LTRON/Freebuild-v0'
    train = True
    auto_reset = True

class LtronInterface:
    def __init__(self, config):
        self.env = gym.make(
            config.env_name,
            config=config,
            train=config.train,
        )
        if config.auto_reset:
            self.env = gym.wrappers.AutoResetWrapper(self.env)
        o,i = self.env.reset(seed=config.seed)
        self.recent_observation = o
        
        self.scene = self.env.components['scene'].brick_scene
        self.window = self.scene.render_environment.window
        self.renderer = self.scene.render_environment.renderer
        
        self.button = 0
        self.click = (0,0)
        self.release = (0,0)
        self.left_shift_down = False
        self.right_shift_down = False
        
        self.window.register_callbacks(
            glutDisplayFunc = self.render,
            glutIdleFunc = self.render,
            glutKeyboardFunc = self.key_press,
            glutSpecialFunc = self.special_key_press,
            glutSpecialUpFunc = self.special_key_release,
            glutMouseFunc = self.mouse_button,
            glutMotionFunc = self.mouse_move,
        )
        print('Left-click: hole')
        print('Right-click: stud')
    
    def render(self):
        self.window.set_active()
        self.window.enable_window()
        self.scene.color_render(flip_y=False)
    
    def dump_image(self, image_path):
        image = self.recent_observation['image']
        target_image = self.recent_observation['target_image']
        save_image(image, image_path)
        save_image(target_image, image_path.replace('.', '_target.'))
    
    def dump_scene(self, scene_path):
        self.scene.export_ldraw(scene_path)
    
    def key_press(self, key, x, y):
        if key == b'r':
            self.env.reset()
            return
        
        if key == b'\x1b':
            breakpoint()
            return
        
        elif key == b'\r':
            print('Please specify an output image file path:')
            file_path = input()
            self.dump_image(file_path)
            return
        
        elif key == b'\\':
            print('Please specify an output scene file path:')
            file_path = input()
            self.dump_scene(file_path)
            return
        
        action = None
        
        mode_space = self.env.action_space['action_primitives']['mode']
        pick_and_place_mode = mode_space.names.index('pick_and_place')
        rotate_mode = mode_space.names.index('rotate')
        if 'viewpoint' in mode_space.names:
            viewpoint_mode = mode_space.names.index('viewpoint')
        remove_mode = mode_space.names.index('remove')
        
        # pick and place
        if key == b'p':
            # pick and place
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = pick_and_place_mode
            action['action_primitives']['pick_and_place'] = 1
        
        # rotate
        elif key == b'[':
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = rotate_mode
            action['action_primitives']['rotate'] = 18 #1
        
        elif key == b']':
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = rotate_mode
            action['action_primitives']['rotate'] = 22 #3
        
        # table viewpoint
        elif key == b'w':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.ELEVATION_NEG.value)
        
        elif key == b's':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.ELEVATION_POS.value)
        
        elif key == b'a':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.AZIMUTH_NEG.value)
        
        elif key == b'd':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.AZIMUTH_POS.value)
        
        
        elif key == b'A':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.X_NEG.value)
        
        elif key == b'W':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.Y_POS.value)
        
        elif key == b'D':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.X_POS.value)
        
        elif key == b'S':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.Y_NEG.value)
        
        elif key == b'q':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.DISTANCE_NEG.value)
        
        elif key == b'e':
            if 'viewpoint' in mode_space.names:
                action = self.env.no_op_action()
                action['action_primitives']['mode'] = viewpoint_mode
                action['action_primitives']['viewpoint'] = (
                    ViewpointActions.DISTANCE_POS.value)
        
        elif key == b'\x08':
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = remove_mode
            action['action_primitives']['remove'] = 1
        
        if action is not None:
            action['cursor']['button'] = self.button
            action['cursor']['click'] = self.click
            action['cursor']['release'] = self.release
        
        if key == b' ':
            #num_expert = self.recent_observation['num_expert_actions']
            expert_data = self.recent_observation['expert']
            if expert_data is not None:
                expert_valid, action, *_ = expert_data
                if expert_valid:
                    #expert_i = numpy.random.randint(num_expert)
                    #action = hierarchy_getitem(
                    #    self.recent_observation['expert'], expert_i)
                    mode_index = action['action_primitives']['mode']
                    mode_name = mode_space.names[mode_index]
                    print('Taking Expert Action: %s'%mode_name)
                    print(action)
        
        if action is not None:
            o,r,t,u,i = self.env.step(action)
            self.recent_observation = o
            print('Reward:%.02f Terminal:%s Truncated:%s'%(r, t, u))
    
    def special_key_press(self, key, x, y):
        
        print(key)
        
        mode_space = self.env.action_space['action_primitives']['mode']
        translate_mode = mode_space.names.index('translate')
        assemble_step_mode = mode_space.names.index('assemble_step')
        if 'phase' in mode_space.names:
            done_mode = mode_space.names.index('phase')
        elif 'done' in mode_space.names:
            done_mode = mode_space.names.index('done')
        insert_mode = mode_space.names.index('insert')
        
        action = None
        
        # left arrow
        if key == 100:
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = translate_mode
            action['action_primitives']['translate'] = 5
        
        # up arrow
        if key == 101:
            action = self.env.no_op_action()
            if self.shift_down:
                action['action_primitives']['mode'] = translate_mode
                action['action_primitives']['translate'] = 21
            else:
                action['action_primitives']['mode'] = translate_mode
                action['action_primitives']['translate'] = 12
        
        # right arrow
        if key == 102:
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = translate_mode
            action['action_primitives']['translate'] = 6
        
        # down arrow
        if key == 103:
            action = self.env.no_op_action()
            if self.shift_down:
                action['action_primitives']['mode'] = translate_mode
                action['action_primitives']['translate'] = 22
            else:
                action['action_primitives']['mode'] = translate_mode
                action['action_primitives']['translate'] = 11
        
        if key == 105: # pgdn
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = assemble_step_mode
            action['action_primitives']['assemble_step'] = 1
        
        if key == 107: # end
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = done_mode
            action['action_primitives']['phase'] = 1
        
        if key == 108: # insert
            if self.shift_down:
                print('Please enter the shape name (XXXX.dat) '
                    'and LDRAW color index separated by whitespace:')
                text = input()
                try:
                    s,c = text.split()
                    s = SHAPE_CLASS_LABELS[s]
                    c = COLOR_CLASS_LABELS[c]
                except:
                    print('Misformatted input, expected two whitespace '
                        'separated names')
                    s = 0
                    c = 0
            else:
                print('Please enter the shape and color class labels '
                    'separated by whitespace:')
                text = input()
                try:
                    s,c = text.split()
                    s = int(s)
                    c = int(c)
                except:
                    print('Misformed input, expected two whitespace '
                        'separated ints')
                    s = 0
                    c = 0
            
            action = self.env.no_op_action()
            action['action_primitives']['mode'] = insert_mode
            action['action_primitives']['insert'] = numpy.array([s,c])
        
        if key == 112: # shift
            self.left_shift_down = True
        if key == 113:
            self.right_shift_down = True
        
        if action is not None:
            action['cursor']['button'] = self.button
            action['cursor']['click'] = self.click
            action['cursor']['release'] = self.release
            
            o,r,t,u,i = self.env.step(action)
            self.recent_observation = o
            print('Reward:%.02f Terminal:%s Truncated:%s'%(r, t, u))
    
    def special_key_release(self, key, x, y):
        if key == 112:
            self.left_shift_down = False
        if key == 113:
            self.right_shift_down = False
    
    @property
    def shift_down(self):
        return self.left_shift_down or self.right_shift_down
    
    def mouse_button(self, button, button_state, x, y):
        if button == 0 or button == 2:
            if button_state == 0:
                self.click = (y,x)
                if button == 0:
                    self.button = 0
                elif button == 2:
                    self.button = 1
            else:
                self.release = (y,x)
    
    def mouse_move(self, x, y):
        pass

def ltron_env_interface():
    config = LtronInterfaceConfig.from_commandline()
    config.render_mode = 'glut'
    config.max_time_steps = 1000000
    interface = LtronInterface(config)
    
    glut.start_main_loop()

if __name__ == '__main__':
    ltron_env_interface()
