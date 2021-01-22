import torch.nn

from torch_scatter import scatter_max

raise Exception('Deprecated, reference only!')

class BrickFeatureBackbone(torch.nn.Module):
    '''
    This module computes brick features from an image using a backbone and
    brick_feature_layer.
    
    __init__:
        backbone:
            A model that takes raw input (an image) and outputs either a
            single BATCH_SIZE x CHANNELS x H x W spatial feature tensor
            (i.e. resnet backbone) or multiple spatial feature tensors
            (i.e. fpn backbone).
        
        brick_feature_model:
            A model that takes either a single spatial feature tensor or
            multiple spatial feature tensors and returns a
            BATCH_SIZE x NUM_BRICKS x CHANNELS tensor of features for each
            detected brick in the image.
    
    forward:
        x:
            The image that will be supplied to the backbone.
        *args, **kwargs:
            Arbitrary additional input will be passed along to the
            brick_feature_model.
        
        return:
            The brick features returned by the brick_feature_model
    '''
    def __init__(self, backbone, brick_feature_model):
        super(BrickFeatureModel, self).__init__()
        self.backbone = backbone
        self.brick_feature_model = brick_feature_model
    
    def forward(self, x, *args, **kwargs):
        backbone_features = self.backbone(x)
        brick_graph = self.brick_feature_model(
                backbone_features, *args, **kwargs)
        
        return brick_graph

# THIS IS PROBABLY ALL WRONG... use segments_to_graph.py instead.
class BrickFeatureModel(torch.nn.Module):
    def __init__(self, score_model, head_models):
        self.score_model = score_model
        self.head_models = torch.nn.ModuleDict(head_models)
    
    def forward(self, x, segmentation):
        b, c, h, w = x.shape
        score = self.score_model(x).view(b, -1)
        max_cluster_score, cluster_argmax = scatter_max(
                score, segmentation, fill_value=0.0)
        good_clusters = torch.where(max_cluster_score > 0.0)
        source_index = cluster_argmax[good_clusters]
        brick_scores = max_cluster_score[good_clusters]
        brick_vectors = x.view(b, c, -1)[good_clusters[0], :, source_index]
        
        head_results = {}
        for head_name, head_model in self.head_models.items():
            head_value = head_model(x)
            
        
        return GraphData(x=brick_vectors, scores=brick_scores)
