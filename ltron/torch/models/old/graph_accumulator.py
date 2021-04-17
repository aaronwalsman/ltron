import torch

class GraphAccumulator(torch.nn.Module):
    '''
    This module accumulates step-by-step modifications to a graph into a
    single coherent structure.
    '''
    def __init__(self, edge_model):
        super(GraphAccumulator, self).__init__()
        self.edge_model = edge_model
    
    def forward(self,
            step_brick_features,
            accumulated_brick_features,
            accumulated_edges):
        
        step_internal_edges = self.edge_model(
                step_brick_features, step_brick_features)
        step_external_edges = self.edge_model(
                step_brick_features, accumulated_brick_features)
        
        
