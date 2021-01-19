import torch

class AddSpatialEmbedding(torch.nn.Module):
    def __init__(self, shape, channels):
        super(AddSpatialEmbedding, self).__init__()
        self.embeddings = torch.nn.ModuleList()
        self.shape = shape
        self.channels = channels
        for dimension_size in self.shape:
            self.embeddings.append(
                    torch.nn.Embedding(dimension_size, channels))
    
    def forward(self, x):
        assert x.shape[2:] == self.shape
        for i, dimension_size in enumerate(self.shape):
            r = torch.arange(dimension_size, dtype=torch.long, device=x.device)
            embedding = self.embeddings[i](r)
            embedding = embedding.permute(1,0)
            embedding_shape = [1 for _ in range(2+len(self.shape))]
            embedding_shape[1] = self.channels
            embedding_shape[2+i] = dimension_size
            embedding = embedding.view(embedding_shape)
            x = x + embedding
        
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
