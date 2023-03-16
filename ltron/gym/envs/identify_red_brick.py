from ltron.gym.components import (
    VisualInterfaceConfig,
    VisualInterface,
    DatasetLoader,
)

class IdentifyRedBrickConfig(VisualInterfaceConfig):
    dataset_name = 'red_brick'
    dataset_split = 'train'
    dataset_subset = None
    dataset_repeat = 1

class ColoredBrickPrediction(SuperMechaComponent):
    def __init__(self, scene_component, shape_ids, target_color=4):
        self.scene_component = scene_component
        self.shape_ids = shape_ids
        self.target_color = target_color
        self.action_space = Discrete(max(shape_ids.values)+1)
    
    def reset(self):
        # find the red brick in the scene
        for instance in self.scene_component.brick_scene.instances.values():
            colored_bricks = []
            if instance.color == self.target_color:
                colored_bricks.append(instance)
            assert len(colored_bricks) == 1
            colored_brick = colored_bricks[0]
            self.target_shape = self.shape_ids[str(colored_bricks.shape)]
        
        return None, None
    
    def step(self, action):
        if action:
            if action == self.target_shape:
                reward = 1.
            else:
                reward = 0.
            return None, reward, True, False, None
        else:
            return None, 0., False, False, None

class IdentifyRedBrickEnv(SuperMechaContainer):
    def __init__(self, config):
        components = OrderedDict()
        
        # scene
        components['scene'] = EmptySceneComponent(
            shape_ids,
            color_ids,
            max_instances_per_scene,
            max_edges_per_scene,
            track_snaps=True,
            collision_checker=True,
        )
        
        # dataset loader
        component['loader'] = DatasetLoader(
            components['scene'],
            config.dataset_name,
            config.dataset_split,
            subset=config.dataset_subset,
            shuffle=True,
            shuffle_buffer=1000,
            repeat=config.dataset_repeat,
        )
        
        # visual interface
        component['interface'] = VisualInterface(
            components['scene'],
            config,
            include_manipulation=False,
        )
        
        # shape_prediction
        components['shape_prediction'] = ColoredBrickPrediction(
            components['scene'], shape_ids)
        
        super().__init__(components)
