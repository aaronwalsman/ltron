import numpy

import brick_gym.utils as utils
import brick_gym.evaluation as evaluation
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class InstanceGraphConstructionTask(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_instances,
            max_edges,
            scene_component,
            dataset_component):
        
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.scene_component = scene_component
        self.dataset_component = dataset_component
        
        self.action_space = bg_spaces.InstanceGraphSpace(
                self.num_classes, self.max_instances, self.max_edges,
                include_edge_score=True)
        
        self.true_edges = None
    
    def reset(self):
        self.true_edges = {}
        
        brick_scene = self.scene_component.brick_scene
        scene_connections = brick_scene.get_all_snap_connections()
        class_lookup = self.dataset_component.dataset_info['class_ids']
        for instance_a in scene_connections:
            brick_instance_a = brick_scene.instances[instance_a]
            class_a = class_lookup[str(brick_instance_a.brick_type)]
            for instance_b, snap_id in scene_connections[instance_a]:
                brick_instance_b = brick_scene.instances[instance_b]
                id_a = int(instance_a)
                id_b = int(instance_b)
                class_b = class_lookup[str(brick_instance_b.brick_type)]
                if id_a < id_b:
                    self.true_edges[(id_a, id_b, class_a, class_b)] = 1.0
        return None
    
    def step(self, action):
        edges = action['edges']
        unidirectional_edges = edges[0] < edges[1]
        edges = edges[:,unidirectional_edges]
        
        predicted_edges = utils.sparse_graph_to_edge_scores(
                image_index = None,
                node_label = action['instances'],
                edges = edges.T,
                scores = action['edge_scores'])
        
        _, _, edge_ap = evaluation.edge_ap(predicted_edges, self.true_edges)
        
        terminal = False
        num_instances = action['num_instances']
        if num_instances == self.max_instances:
            terminal = True
        
        return None, edge_ap, terminal, None

'''
class GraphConstructionTask(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_nodes,
            scene_component):
        
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_track_snaps()
        
        self.action_space = bg_spaces.GraphScoreSpace(
                self.num_classes, self.max_nodes)
    
    def step(self, action):
        predicted_edge_scores = utils.matrix_to_edge_scores(
                None, action['nodes'], action['edges'])
        target_edge_scores = GET_SCENE_GRAPH
        _, _, ap = evaluation.edge_ap(
                predicted_edge_scores, target_edge_scores)
        
        return None, ap, False, None
    
    #def get_predicted_edge_scores(self, action):
    #    predicted_edge_scores = utils.matrix_to_edge_scores(
    #            None, predicted_graph['nodes'], predicted_graph['edges'])
    #    return predicted_edge_scores
    #
    #def compute_reward(self, state, action):
    #    scene_metadata = state[self.scene_metadata_key]
    #    target_edge_scores = utils.metadata_to_edge_scores(
    #            None, scene_metadata)
    #    predicted_edge_scores = self.get_predicted_edge_scores(action)
    #    _, _, ap = evaluation.edge_ap(
    #            predicted_edge_scores, target_edge_scores)
    #    return ap

class SparseGraphConstructionTask(GraphConstructionTask):
    def __init__(self,
            num_classes,
            max_nodes,
            max_edges,
            graph_key='graph',
            scene_metadata_key='scene_metadata'):
        
        self.max_edges = max_edges
        super(SparseGraphReconstructionTask, self).__init__(
                num_classes=num_classes,
                max_nodes=max_nodes,
                graph_key=graph_key,
                scene_metadata_key=scene_metadata_key)
    
    def update_action_space(self, action_space):
        action_space[self.graph_key] = bg_spaces.SparseGraphScoreSpace(
                self.num_classes, self.max_nodes, self.max_edges)
    
    def get_predicted_edge_scores(self, action):
        predicted_sparse_graph = action[self.graph_key]
        predicted_edge_scores = utils.sparse_graph_to_edge_scores(
                None,
                predicted_sparse_graph['nodes'],
                predicted_sparse_graph['edges'],
                predicted_sparse_graph['scores'])
        return predicted_edge_scores
'''
