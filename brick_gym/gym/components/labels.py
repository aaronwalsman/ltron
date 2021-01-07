import brick_gym.utils as utils
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class GraphLabelComponent(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_nodes,
            scene_component):
        self.num_classes=num_classes
        self.max_nodes=max_nodes
        self.scene_component = scene_component
        self.scene_component.brick_gym.make_track_snaps()
        
        self.observation_space = bg_spaces.GraphScoreSpace(
                self.num_classes, self.max_nodes)
    
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        SOME_NEW_THING
        #node_labels, edge_labels = utils.metadata_to_graph(
        #        metadata, max_nodes=self.max_nodes)
        observation = {
                'nodes' : node_labels,
                'edges' : edge_labels}
        return observation
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
