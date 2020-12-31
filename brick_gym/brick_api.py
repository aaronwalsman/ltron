import renderpy.buffer_manager_egl as buffer_manager_egl
import renderpy.buffer_manager_glut as buffer_manager_glut
from renderpy.core import Renderpy

import brick_gym.config as config
from brick_gym.snap_manager import SnapManager
from brick_gym.ldraw.documents import LDrawDocument

class BrickAPI:
    def __init__(self, render_mode='none', snap_mode='none'):
        self.render_mode = render_mode
        self.snap_mode = snap_mode
        
        if self.render_mode == 'none':
            self.buffer_manager = None
            self.render = None
        else:
            if self.render_mode == 'egl':
                self.buffer_manager = buffer_manager_glut.initialize_shared_
            elif self.render_mode == 'glut':
                self.buffer_manager = buffer_manager_egl.
            else:
                raise ValueError('Unknown render mode: %s'%self.render_mode)
            config_paths = '%s:%s'%(
                    config.paths['renderpy_assets_cfg'],
                    drpy_assets.default_assets_path)
            self.renderer = Renderpy(config_paths)
        
        if self.snap_mode == 'none':
            self.snap_manager = None
        elif self.snap_mode == 'snap':
            self.snap_manager = SnapManager()
    
    def load_ldraw(self, path):
        document = LDrawDocument.parse_document(path)
        parts = document.get_all_parts()
    
    def add_brick(self, brick_name, color, transform):
        pass
        return brick_instance
    
    def remove_brick(self, brick_instance):
        pass
    
    def hide_brick(self, brick_instance):
        pass
    
    def 
