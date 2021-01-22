import torch

from brick_gym.torch.brick_geometric import BrickList, BrickGraph

class GraphStepModel(torch.nn.Module):
    '''
    This module computes brick features from an image and passes them to
    multiple heads that are specified at initialization.
    
    __init__:
        brick_feature_model:
            A model that takes raw input (an image) and converts it to a set of
            brick vectors.
        
        brick_heads:
            A dictionary of heads that will each receive the list of
            brick vectors and will return an output for each one.
    
    forward:
        *args, **kwargs:
            Arbitrary input args passed directly to the brick_feature_model
    
        return:
            A dictionary containing the result of each head. If you want   
            access to the raw brick vectors, add an entry to the brick_heads
            dictionary that contains a torch.nn.Identity layer.
    
    '''
    def __init__(self, backbone, score_model, segmentation_model, heads):
        super(GraphStepModel, self).__init__()
        self.backbone = backbone
        self.score_model = score_model
        self.segmentation_model = segmentation_model
        self.heads = torch.nn.ModuleDict(heads)
    
    def forward(self, x, segmentation=None):
        x = self.backbone(x)
        dense_scores = torch.sigmoid(self.score_model(x))
        head_features = {head_name : head_model(x)
                for head_name, head_model in self.heads.items()}
        
        # compute the segmentation if it wasn't supplied externally
        if segmentation is None:
            assert segmentation_model is not None
            segmentation = self.segmentation_model(x)
        
        dense_graph_scores = dense_scores * (segmentation != 0).unsqueeze(1)
        batch_graphs = BrickList.segmentations_to_brick_lists(
                dense_graph_scores, segmentation, head_features)
        
        return batch_graphs, segmentation, dense_scores, head_features
