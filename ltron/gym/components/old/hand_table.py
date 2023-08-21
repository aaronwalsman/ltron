from steadfast.config import Config
from steadfast.gym.component_env import ComponentEnv

class HandTableConfig(Config):
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    tile_color_render = False
    tile_width = 16
    tile_height = 16
    
    check_collision = True

class HandTable(ComponentEnv):
    def __init__(self,
        config,
        brick_set,
        max_instances_per_scene,
        max_edges_per_scene,
    ):
        components = OrderedDict()
        
        component['table_scene'] = EmptySceneComponent(
            brick_set['shape_ids'],
            brick_set['color_ids'],
            max_instances_per_scene,
            max_edges_per_scene,
            track_snaps=True,
            collision_checker = config.check_collision,
        )
        component['hand_scene'] = EmptySceneComponent(
            brick_set['shape_ids'],
            brick_set['color_ids'],
            max_instances_per_scene,
            max_edges_per_scene,
            track_snaps=True,
            collision_checker = config.check_collision,
        )
