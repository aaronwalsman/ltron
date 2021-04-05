import torch

class AddSpatialEmbedding(torch.nn.Module):
    def __init__(self, shape, channels):
        super(AddSpatialEmbedding, self).__init__()
        self.embeddings = torch.nn.ParameterList()
        self.shape = shape
        self.channels = channels
        stdv = 1. / (self.channels**0.5)
        for i, dimension_size in enumerate(self.shape):
            embedding_shape = [1 for _ in range(2+len(self.shape))]
            embedding_shape[1] = self.channels
            embedding_shape[2+i] = dimension_size
            embedding = torch.zeros(embedding_shape).uniform_(-stdv, stdv)
            self.embeddings.append(torch.nn.Parameter(embedding))
    
    def forward(self, x):
        assert x.shape[2:] == self.shape
        for i in range(len(self.shape)):
            x = x + self.embeddings[i]
        
        return x

class SpatialAttention2D(torch.nn.Module):
    def __init__(self, channels):
        super(SpatialAttention2D, self).__init__()
        self.attention_layer = torch.nn.Conv2d(channels, 1, kernel_size=1)
    
    def forward(self, x):
        attention = self.attention_layer(x)
        bs, _, h, w = attention.shape
        attention = torch.softmax(attention.view(bs, 1, -1), dim=-1)
        attention = attention.view(bs, 1, h, w)
        
        x = torch.sum(x * attention, dim=(2,3))
        
        return x
