import numpy

import gymnasium as gym

import splendor.contexts.glut as glut
from splendor.image import save_image

from ltron.gym.envs import (
    InterfaceEnvConfig,
    InterfaceEnv,
)
from ltron.gym.components import ViewpointActions

class LtronInterfaceConfig(InterfaceEnvConfig):
    render_mode = 'glut'
    
    dataset_name = 'rca'
    dataset_split = '3_4_train'
    
    seed = 1234567890

class LtronInterface:
    def __init__(self, config):
        self.env = gym.make(
            'LTRON/Interface-v0',
            config=config,
            dataset_name=config.dataset_name,
        )
        self.env.reset(seed=config.seed)
        
        self.scene = self.env.components['scene'].brick_scene
        self.window = self.scene.render_environment.window
        self.renderer = self.scene.render_environment.renderer
        
        self.button = 0
        self.click = (0,0)
        self.release = (0,0)
        
        self.window.register_callbacks(
            glutDisplayFunc = self.render,
            glutIdleFunc = self.render,
            glutKeyboardFunc = self.key_press,
            glutSpecialFunc = self.special_key_press,
            glutMouseFunc = self.mouse_button,
            glutMotionFunc = self.mouse_move,
        )
        
        self.dumped_images = 0
    
    def render(self):
        self.window.set_active()
        self.window.enable_window()
        self.scene.color_render(flip_y=False)
    
    def dump_image(self):
        self.window.set_active()
        self.window.enable_window()
        self.scene.color_render(flip_y=False)
        image = self.window.read_pixels()[::-1]
        image_path = './ltron_interface_%06i.png'%self.dumped_images
        print('Saving image to: %s'%image_path)
        save_image(image, image_path)
        self.dumped_images += 1
    
    def key_press(self, key, x, y):
        if key == b'r':
            self.env.reset()
            return
        
        elif key == b'\r':
            self.dump_image()
            return
        
        action = self.env.no_op_action()
        
        # pick and place
        if key == b'p':
            # pick and place
            action['interface']['primitives']['mode'] = 1
            action['interface']['primitives']['pick_and_place'] = 1
        
        # rotate
        elif key == b'[':
            action['interface']['primitives']['mode'] = 2
            action['interface']['primitives']['rotate'] = 1
        
        elif key == b']':
            action['interface']['primitives']['mode'] = 2
            action['interface']['primitives']['rotate'] = 2
        
        # table viewpoint
        elif key == b'w':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.ELEVATION_NEG.value)
        
        elif key == b's':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.ELEVATION_POS.value)
        
        elif key == b'a':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.AZIMUTH_NEG.value)
        
        elif key == b'd':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.AZIMUTH_POS.value)
        
        elif key == b'q':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.DISTANCE_NEG.value)
        
        elif key == b'e':
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.DISTANCE_POS.value)
        
        elif key == b'\x08':
            action['interface']['primitives']['mode'] = 3
            action['interface']['primitives']['remove'] = 1
        
        elif key == b'\t':
            print('Please enter the shape id and color id separated by commas:')
            text = input()
            try:
                s,c = text.split(',')
                s = int(s)
                c = int(c)
            except:
                print('Misformatted input, expected two comma-separated ints')
                s = 0
                c = 0
            action['interface']['primitives']['mode'] = 4
            action['interface']['primitives']['insert'] = numpy.array([s,c])
        
        action['interface']['cursor']['button'] = self.button
        action['interface']['cursor']['click'] = self.click
        action['interface']['cursor']['release'] = self.release
        
        
        o,r,t,u,i = self.env.step(action)
    
    def special_key_press(self, key, x, y):
        
        action = self.env.no_op_action()
        if key == 100:
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.X_NEG.value)
        
        if key == 101:
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.Y_POS.value)
        
        if key == 102:
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.X_POS.value)
        
        if key == 103:
            action['interface']['primitives']['mode'] = 0
            action['interface']['primitives']['viewpoint'] = (
                ViewpointActions.Y_NEG.value)
        
        o,r,t,u,i = self.env.step(action)
    
    def mouse_button(self, button, button_state, x, y):
        if button == 0 or button == 2:
            if button_state == 0:
                self.click = (y,x)
                #print(self.click)
                if button == 0:
                    self.button = 0
                elif button == 2:
                    self.button = 1
            else:
                self.release = (y,x)
                #print(self.release)
    
    def mouse_move(self, x, y):
        #print(x, y)
        pass

if __name__ == '__main__':
    config = LtronInterfaceConfig.from_commandline()
    interface = LtronInterface(config)
    
    glut.start_main_loop()
