from steadfast.config import Config
from steadfast.gym.component_env import ComponentEnv

class TowerEnvConfig(Config):
    max_episode_length = 32
    
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    check_collision = True
    
    brick_set = 'easy_6_6'
    
class TowerEnv(ComponentEnv):
    def __init__(self, config):
        
        components = OrderedDict()
