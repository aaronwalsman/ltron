import torch

class SimpleEdgeModel(torch.nn.Module):
    def __init__(self,
            input_channels,
            hidden_channels):
        super(SimpleEdgeModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.linear_a = torch.nn.Linear(input_channels, hidden_channels)
        self.linear_b = torch.nn.Linear(hidden_channels, hidden_channels)

        #self.combination_a = torch.nn.Linear(hidden_channels*2,hidden_channels)
        self.combination_a = torch.nn.Linear(hidden_channels, hidden_channels)
        self.combination_b = torch.nn.Linear(hidden_channels, hidden_channels)
        self.combination_c = torch.nn.Linear(hidden_channels, hidden_channels)
        self.edge_out = torch.nn.Linear(hidden_channels,1)

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
        
        #x = torch.cat((x_a, x_b), dim=-1).view(-1, self.hidden_channels*2)
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
class DotProductEdgeModel(torch.nn.Module):
    def __init__(self,
            input_channels,
            hidden_channels):
        super(DotProductEdgeModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.linear_a = torch.nn.Linear(input_channels, hidden_channels)
        self.linear_b = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear_c = torch.nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        x = x.view(-1, self.input_channels)
        x = self.linear_a(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_b(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_c(x)
        
        x_a = x.reshape(
                batch_size, 1, num_nodes, self.hidden_channels)#.expand(
                #batch_size, num_nodes, num_nodes, self.hidden_channels)
        x_b = x.reshape(
                batch_size, num_nodes, 1, self.hidden_channels)#.expand(
                #batch_size, num_nodes, num_nodes, self.hidden_channels)
        x = x_a * x_b
        x = torch.sum(x, dim=-1)
        #x = x_a - x_b
        #x = x * x
        #x = -torch.sum(x, dim=-1)
        return x
'''
