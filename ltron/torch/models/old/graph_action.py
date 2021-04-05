import torch

class GraphActionModel(torch.nn.Module):
    def __init__(self, backbone, num_classes, backbone_channels):
        super(GraphActionModel, self).__init__()
        self.backbone = backbone
        self.output_layer = torch.nn.Linear(backbone_channels, num_classes+2)
    
    def forward(self, x):
        bs, i, c, h, w = x.shape
        x = x.view(bs*i, c, h, w)
        feature_vector = self.backbone(x)
        x = torch.nn.functional.relu(feature_vector)
        out = self.output_layer(x)
        class_logits = out[:,:-2].reshape(bs, i, -1)
        confidence_logits = out[:,-2].reshape(bs, i)
        action_logits = out[:,-1].reshape(bs, i)
        feature_vector = feature_vector.reshape(bs, i, -1)
        return feature_vector, class_logits, confidence_logits, action_logits
