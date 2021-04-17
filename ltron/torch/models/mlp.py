import torch

class LinearStack(torch.nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features):
        super(LinearStack, self).__init__()
        layer_features = (
                [in_features] +
                [hidden_features] * (num_layers-1) +
                [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Linear(in_f, out_f))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.layers(x)

class Conv2dStack(torch.nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features):
        super(Conv2dStack, self).__init__()
        layer_features = (
                [in_features] +
                [hidden_features] * (num_layers-1) +
                [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Conv2d(in_f, out_f, 1))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
    
    def forward(self, x):
        return self.layers(x)
