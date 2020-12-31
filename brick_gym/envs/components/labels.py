import brick_gym.utils as utils
import brick_gym.spaces as bg_spaces
from brick_gym.envs.components.brick_env_component import BrickEnvComponent

class GraphLabelComponent(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_nodes,
            graph_label_key='graph_label',
            scene_metadata_key='scene_metadata'):
        self.num_classes=num_classes
        self.max_nodes=max_nodes
        self.graph_label_key = graph_label_key
        self.scene_metadata_key = scene_metadata_key
    
    def update_observation_space(self, observation_space):
        observation_space[self.graph_label_key] = bg_spaces.GraphScoreSpace(
                self.num_classes, self.max_nodes)
    
    def compute_observation(self, state, observation):
        metadata = state[self.scene_metadata_key]
        node_labels, edge_labels = utils.metadata_to_graph(
                metadata, max_nodes=self.max_nodes)
        observation[self.graph_label_key] = {
                'nodes' : node_labels,
                'edges' : edge_labels}
