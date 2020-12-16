import torch

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
    
    def reset(self, node_features=None):
        self.node_features = node_features
    
    def forward(self, x):
        b, i, c, h, w = x.shape
        step_features = self.backbone(x.view(b*i,c,h,w))
        step_node_logits = self.node_model(step_features).view(b,i,-1)
        step_action_logits = self.action_model(step_features).view(b,i)
        step_nodes = torch.argmax(step_node_logits, dim=-1)
        
        feature_dim = step_features.shape[-1]
        step_features = step_features.view(b,i,feature_dim)
        if self.ndoe_features is None:
            self.node_features = step_features
        else:
            zero_nodes = step_nodes == 0
            self.node_features = (
                    self.node_features * zero_nodes.unsqueeze(-1) +
                    step_features * ~zero_nodes.unsqueeze(-1))
        
        episode_node_logits = self.node_model(
                self.node_features.view(b*i, feature_dim)).view(b,i,num_classes)
        episode_edge_logits = self.edge_model(self.node_features)
        
        return (step_features,
                step_node_logits,
                step_action_logits,
                episode_node_logits,
                episode_edge_logits)
