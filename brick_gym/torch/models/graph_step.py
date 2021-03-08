import time

import torch

from brick_gym.torch.brick_geometric import BrickList, BrickGraph
from brick_gym.torch.models.spatial import AddSpatialEmbedding

class SlimGraphStepModel(torch.nn.Module):
    def __init__(self,
            backbone,
            dense_heads,
            single_heads,
            add_spatial_embedding=False,
            decoder_channels=256,
            output_resolution=(256,256)):
        super(SlimGraphStepModel, self).__init__()
        self.backbone = backbone
        self.add_spatial_embedding = add_spatial_embedding
        if self.add_spatial_embedding:
            self.spatial_embedding_layer = AddSpatialEmbedding(
                    output_resolution, decoder_channels)
        self.dense_heads = torch.nn.ModuleDict(dense_heads)
        self.single_heads = torch.nn.ModuleDict(single_heads)

    def forward(self, x):
        x, xn = self.backbone(x)
        if len(self.single_heads):
            xs = torch.nn.functional.adaptive_avg_pool2d(xn[0], (1,1))
            xs = torch.flatten(xs, 1)
        if self.add_spatial_embedding:
            x = self.spatial_embedding_layer(x)
        
        head_features = {head_name : head_model(x)
                for head_name, head_model in self.dense_heads.items()}
        head_features.update({head_name : head_model(xs)
                for head_name, head_model in self.single_heads.items()})
        
        return head_features

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
    def __init__(self,
            backbone,
            score_model,
            segmentation_model,
            dense_heads,
            single_heads,
            add_spatial_embedding=False,
            decoder_channels=256,
            output_resolution=(256,256)):
        super(GraphStepModel, self).__init__()
        self.backbone = backbone
        self.add_spatial_embedding = add_spatial_embedding
        if self.add_spatial_embedding:
            self.spatial_embedding_layer = AddSpatialEmbedding(
                    output_resolution, decoder_channels)
        self.score_model = score_model
        self.segmentation_model = segmentation_model
        self.heads = torch.nn.ModuleDict(dense_heads)
        self.single_heads = torch.nn.ModuleDict(single_heads)
    
    def forward(self,
            x,
            segmentation=None,
            max_instances=None,
            brick_vector_mode='average'):
        x, xn = self.backbone(x)
        if len(self.single_heads):
            xs = torch.nn.functional.adaptive_avg_pool2d(xn[0], (1,1))
            xs = torch.flatten(xs, 1)
        if self.add_spatial_embedding:
            x = self.spatial_embedding_layer(x)
        #dense_scores = torch.sigmoid(self.score_model(x))
        #dense_score_logits = self.score_model(x.detach())
        dense_score_logits = self.score_model(x)
        head_features = {head_name : head_model(x)
                for head_name, head_model in self.heads.items()}
        head_features.update({head_name : head_model(xs)
                for head_name, head_model in self.single_heads.items()})
        
        # compute the segmentation if it wasn't supplied externally
        #if segmentation is None:
        #    assert self.segmentation_model is not None
        #    segmentation = self.segmentation_model(head_features)
        
        if segmentation is not None:
            dense_graph_score_logits = (
                    dense_score_logits * (segmentation != 0).unsqueeze(1) + 
                    -50. * (segmentation == 0).unsqueeze(1))
        
        if segmentation is not None:
            if brick_vector_mode == 'single':
                batch_lists = BrickList.segmentations_to_brick_lists(
                        dense_graph_score_logits,
                        segmentation,
                        head_features,
                        max_instances)
            elif brick_vector_mode == 'average':
                batch_lists = BrickList.segmentations_to_brick_lists_average(
                        dense_graph_score_logits,
                        segmentation,
                        head_features,
                        max_instances)
        else:
            batch_lists = BrickList.single_best_brick_lists(
                    dense_score_logits,
                    head_features)
        
        return batch_lists, segmentation, dense_score_logits, head_features
