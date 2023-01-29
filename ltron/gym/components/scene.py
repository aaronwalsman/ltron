from supermecha import SuperMechaComponent

from ltron.gym.spaces import AssemblySpace
from ltron.hierarchy import hierarchy_branch
from ltron.bricks.brick_scene import BrickScene

class EmptySceneComponent(SuperMechaComponent):
    def __init__(self,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        renderable=True,
        render_args=None,
        track_snaps=False,
        collision_checker=False,
        clear_frequency='reset',
    ):
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.clear_frequency = clear_frequency
        
        if render_args is None:
            render_args = {'opengl_mode':'egl', 'load_scene':'front_light'}
        
        print(render_args)
        
        self.brick_scene = BrickScene(
            renderable=renderable,
            render_args=render_args,
            track_snaps=track_snaps,
            collision_checker=collision_checker,
        )
    
    def clear_scene(self):
        self.brick_scene.clear_instances()
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=options)
        if self.clear_frequency in ('step', 'reset'):
            self.clear_scene()
        return None, {}
    
    def step(self, action):
        if self.clear_frequency in ('step',):
            self.clear_scene()
        return None, 0., False, False, {}
    
    def set_state(self, state):
        self.brick_scene.clear_instances()
        self.brick_scene.set_assembly(
            state, self.shape_ids, self.color_ids)
        
        return None, {}
    
    def get_state(self):
        state = self.brick_scene.get_assembly(
            self.shape_ids, self.color_ids, self.max_instances, self.max_edges)
        
        return state


class SingleSceneComponent(EmptySceneComponent):
    def __init__(self,
        initial_scene_path,
        *args,
        **kwargs
    ):
        super(SingleSceneComponent, self).__init__(*args, **kwargs)
        self.current_scene_path = initial_scene_path


class DatasetSceneComponent(EmptySceneComponent):
    def __init__(self,
        dataset_component=None,
        path_location=None,
        *args,
        **kwargs
    ):
        self.dataset_component = dataset_component
        self.path_location = path_location
        
        dataset_info = self.dataset_component.dataset_info
        super(DatasetSceneComponent, self).__init__(
            shape_ids=dataset_info['shape_ids'],
            color_ids=dataset_info['color_ids'],
            max_instances=dataset_info['max_instances_per_scene'],
            max_edges=dataset_info['max_edges_per_scene'],
            *args,
            **kwargs,
        )
    
    def reset(self):
        self.current_scene_path = hierarchy_branch(
            self.dataset_component.dataset_item, self.path_location)
        observation = super(DatasetSceneComponent, self).reset()
        return observation

