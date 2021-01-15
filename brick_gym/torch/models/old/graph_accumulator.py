import torch

class GraphAccumulator(torch.nn.Module):
    def __init__(self,
            backbone,
            node_model,
            confidence_model,
            edge_model):
        super(GraphAccumulator, self).__init__()
        self.backbone = backbone
        self.node_model = node_model
        self.confidence_model = confidence_model
        self.edge_model = edge_model
    
    def reset(self, node_features=None, confidence=None):
        assert node_features is None == confidence is None
        self.node_features = node_features
        self.confidence = confidence
    
    def forward(self, x):
        b, i, c, h, w = x.shape
        step_features = self.backbone(x.view(b*i,c,h,w))
        step_confidence_logits = self.confidence_model(step_features)
        if step_confidence_logits.shape[-1] == 2:
            # TODO: deprecate this
            step_confidence_logits = step_confidence_logits.view(b,i,2)
            step_confidence = torch.softmax(
                    step_confidence_logits, dim=-1)[:,:,1]
        else:
            step_confidence_logits = step_confidence_logits.view(bs,i)
            step_confidence = torch.sigmoid(step_confidence_logits)
        
        step_node_logits = self.node_model(step_features)
        num_classes = step_node_logits.shape[-1]
        step_node_logits = step_node_logits.view(b,i,num_classes)
        step_nodes = torch.argmax(step_node_logits, dim=-1)
        
        feature_dim = step_features.shape[-1]
        step_features = step_features.view(b,i,feature_dim)
        if self.confidence is None:
            self.node_features = step_features
            self.confidence = step_confidence
        else:
            nonzero_nodes = step_nodes != 0
            # turns out this is not a great signal at the moment
            #high_confidence = step_confidence >= self.confidence
            #replace_nodes = high_confidence & nonzero_nodes
            replace_nodes = nonzero_nodes
            self.node_features = (
                    self.node_features * ~replace_nodes.unsqueeze(-1) +
                    step_features * replace_nodes.unsqueeze(-1))
            self.confidence = (
                    self.confidence * ~replace_nodes +
                    step_confidence * replace_nodes)
        
        episode_node_logits = self.node_model(
                self.node_features.view(b*i,feature_dim)).view(b,i,num_classes)
        episode_edge_logits = self.edge_model(self.node_features)
        
        return (step_features,
                step_confidence_logits,
                step_node_logits,
                episode_node_logits,
                episode_edge_logits)
