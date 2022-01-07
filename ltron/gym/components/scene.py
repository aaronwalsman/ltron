from gym.spaces import Dict, Discrete
from ltron.gym.spaces import AssemblySpace

from ltron.hierarchy import hierarchy_branch
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.bricks.brick_scene import BrickScene

class EmptySceneComponent(LtronGymComponent):
    def __init__(self,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        renderable=True,
        render_args=None,
        track_snaps=False,
        collision_checker=False,
    ):
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.current_scene_path = None
        
        if render_args is None:
            render_args = {'opengl_mode':'egl', 'load_scene':'grey_cube'}
        
        self.brick_scene = BrickScene(
            renderable=renderable,
            render_args=render_args,
            track_snaps=track_snaps,
            collision_checker=collision_checker,
        )
        
        observation_space = {'scene_loaded':Discrete(2)}
        self.observation_space = Dict(observation_space)
    
    def observe(self, initial=False):
        self.observation = {
            'scene_loaded' : int(self.current_scene_path is not None)
        }
    
    def reset(self):
        self.brick_scene.clear_instances()
        if self.current_scene_path is not None:
            self.brick_scene.import_ldraw(self.current_scene_path)
        
        self.observe(initial=True)
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.brick_scene.clear_instances()
        self.brick_scene.set_assembly(
            state, self.shape_ids, self.color_ids)
        
        self.observe()
        return self.observation
    
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

