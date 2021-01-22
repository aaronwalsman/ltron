import torch

from brick_gym.torch.models.utils import AddSpatialEmbedding

class GraphModel(torch.nn.Module):
    def __init__(self,
            backbone,
            node_model,
            edge_model,
            action_model):
        super(GraphModel, self).__init__()
        self.backbone = backbone
        self.node_model = node_model
        self.edge_model = edge_model
        self.action_model = action_model
        
        self.node_features = None
    
    def forward(self, x, terminal):
        b, i, c, h, w = x.shape
        step_features = self.backbone(x.view(b*i,c,h,w))
        step_node_logits = self.node_model(step_features).view(b,i,-1)
        step_action_logits = self.action_model(step_features).view(b,i)
        step_nodes = torch.argmax(step_node_logits, dim=-1)
        
        feature_dim = step_features.shape[-1]
        step_features = step_features.view(b,i,feature_dim)
        if self.node_features is None:
            self.node_features = step_features
        else:
            '''
            zero_nodes = step_nodes == 0
            self.node_features = (
                    self.node_features * zero_nodes.unsqueeze(-1) +
                    step_features * ~zero_nodes.unsqueeze(-1))
            '''
            overwrite = (step_nodes != 0) | terminal.unsqueeze(-1)
            self.node_features = (
                    self.node_features * ~overwrite.unsqueeze(-1) +
                    step_features * overwrite.unsqueeze(-1))
        
        episode_node_logits = self.node_model(
                self.node_features.view(b*i, feature_dim)).view(b,i,-1)
        episode_edge_logits = self.edge_model(self.node_features)
        
        return (step_features,
                step_node_logits,
                step_action_logits,
                episode_node_logits,
                episode_edge_logits)

class GraphPoolModel(torch.nn.Module):
    def __init__(self,
            backbone,
            segment_model,
            node_model,
            edge_model,
            action_model,
            shape,
            backbone_channels,
            pool_mode='swap'):
        super(GraphPoolModel, self).__init__()
        self.backbone = backbone
        self.spatial_embedding = AddSpatialEmbedding(shape, backbone_channels)
        self.segment_model = segment_model
        self.node_model = node_model
        self.edge_model = edge_model
        self.action_model = action_model
        self.pool_mode = pool_mode
        
    def forward(self, x, segments=None):
        features = self.backbone(x)
        features = self.spatial_embedding(features)
        if segments is None:
            segments = self.segment_model(features)
        batch_size = segments.shape[0]
        segment_values = []
        for i in range(batch_size):
            unique_segments = torch.unique(segments[i])
            if unique_segments[0] == 0:
                unique_segments = unique_segments[1:]
            segment_values.append(unique_segments)
        segment_values = torch.nn.utils.rnn.pad_sequence(
                segment_values, batch_first=True)
        segment_features = []
        num_segments = segment_values.shape[1]
        
        for i in range(num_segments):
            segment_value = segment_values[:,i].unsqueeze(1).unsqueeze(2)
            segment_pixels = (segments == segment_value)
            if self.pool_mode == 'swap':
                clamped_features = torch.clamp(features, -10, 10)
                weights = (torch.exp(clamped_features) *
                        segment_pixels.unsqueeze(1))
                #weights = torch.clamp(weights, 0, 1e5)
                normalizer = torch.sum(weights, dim=(2,3))
                normalizer = normalizer.unsqueeze(2).unsqueeze(3) + 1e-5
                weights = weights / normalizer
                segment_feature = torch.sum(features * weights, dim=(2,3))
                segment_features.append(segment_feature)
            elif pool_mode == 'avg':
                normalizer = torch.sum(segment_pixels, dim=(1,2))
                normalizer = normalizer.unsqueeze(1) + 1e-5
                segment_x = x * segment_pixels.unsqueeze(1)
                segment_x = torch.sum(segment_x, dim=(2,3))
                segment_x = segment_x / normalizer
                segment_features.append(segment_x)
        
        segment_features = torch.stack(segment_features, dim=1)
        b, s, c = segment_features.shape
        segment_features = segment_features.view(b*s, c)
        
        step_node_logits = self.node_model(segment_features).view(b, s, -1)
        step_action_logits = self.action_model(segment_features).view(b, s)
        
        segment_weights = segment_values != 0
        batch_weights = torch.sum(segment_weights, dim=1)
        return (step_node_logits,
                step_action_logits,
                segment_values,
                segment_weights)
