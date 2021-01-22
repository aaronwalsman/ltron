import torch

class PairwiseFeatureDifferenceLayer(torch.nn.Module):
    '''
    This module takes two lists (x_a and x_b) of features and computes
    the difference between every pairwise combination of features in
    x_a and x_b.
    
    __init__:
        Uses default module init, this layer contains no layers or parameters.
    
    forward:
        x_a:
            A tensor with shape BATCH_SIZE x NUM_A x CHANNELS
        
        x_b:
            A tensor with shape BATCH_SIZE x NUM_B x CHANNELS
        
        return:
            A tensor with shape BATCH_SIZE x NUM_A x NUM_B x CHANNELS
    '''
    def forward(self, x_a, x_b):
        bs_a, n_a, c_a = x_a.shape
        bs_b, n_b, c_b = x_b.shape
        assert bs_a == bs_b
        assert c_a == c_b
        
        x_a = x_a.view(bs_a, n_a, 1, c_a)
        x_b = x_b.view(bs_b, 1, n_b, c_b)
        x = (x_a - x_b)
        
        return x

class FeatureDifferenceEdgeModel(torch.nn.Module):
    def __init__(self,
            input_channels,
            hidden_channels):
        super(FeatureDifferenceEdgeModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.linear_a = torch.nn.Linear(input_channels, hidden_channels)
        self.linear_b = torch.nn.Linear(hidden_channels, hidden_channels)

        self.combination_a = torch.nn.Linear(hidden_channels, hidden_channels)
        self.combination_b = torch.nn.Linear(hidden_channels, hidden_channels)
        self.combination_c = torch.nn.Linear(hidden_channels, hidden_channels)
        self.edge_out = torch.nn.Linear(hidden_channels,1)
    
    def forward(self, x_a, x_b=None):
        batch_size, num_nodes, feature_dim = x.shape
        x_a = x.view(-1, self.input_channels)
        x_a = self.linear_a(x_a)
        x_a = torch.nn.functional.relu(x_a)
        x_a = self.linear_b(x_a)
        x_a = torch.nn.functional.relu(x_a)
        x_a = x_a.reshape(
                batch_size, 1, num_nodes, self.hidden_channels).expand(
                batch_size, num_nodes, num_nodes, self.hidden_channels)
        if x_b is None:
            x_b = x_a.reshape(
                    batch_size, num_nodes, 1, self.hidden_channels).expand(
                    batch_size, num_nodes, num_nodes, self.hidden_channels)
        else:
            x_b = x.view(-1, self.input_channels)
            x_b = self.linear_a(x_b)
            x_b = torch.nn.functional.relu(x_b)
            x_b = self.linear_b(x_b)
            x_b = torch.nn.functional.relu(x_b)
            x_b = x_b.reshape(
                    batch_size, num_nodes, 1, self.hidden_channels).expand(
                    batch_size, num_nodes, num_nodes, self.hidden_channels)
            
        
        x = (x_a - x_b).view(-1, self.hidden_channels)
        x = self.combination_a(x)
        x = torch.nn.functional.relu(x)
        x = self.combination_b(x)
        x = torch.nn.functional.relu(x)
        x = self.combination_c(x)
        x = torch.nn.functional.relu(x)
        return self.edge_out(x).view(
                batch_size, num_nodes, num_nodes)
    
    '''
    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        x = x.view(-1, self.input_channels)
        x = self.linear_a(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_b(x)
        x = torch.nn.functional.relu(x)

        x_a = x.reshape(
                batch_size, 1, num_nodes, self.hidden_channels).expand(
                batch_size, num_nodes, num_nodes, self.hidden_channels)
        x_b = x.reshape(
                batch_size, num_nodes, 1, self.hidden_channels).expand(
                batch_size, num_nodes, num_nodes, self.hidden_channels)
        
        x = (x_a - x_b).view(-1, self.hidden_channels)
        x = self.combination_a(x)
        x = torch.nn.functional.relu(x)
        x = self.combination_b(x)
        x = torch.nn.functional.relu(x)
        x = self.combination_c(x)
        x = torch.nn.functional.relu(x)
        return self.edge_out(x).view(
                batch_size, num_nodes, num_nodes)
    '''
