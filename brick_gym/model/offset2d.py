import torch

EPS = 1e-5

class Offset2D(torch.nn.Module):
    def __init__(self, feature_dim):
        self.conv = torch.nn.Conv2d(feature_dim, 3, kernel_size=1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        offset_attention = self.conv(x)
        
        offset = offset_attention[:,:2]
        offset[:,0] = offset[:,0] * height
        offset[:,1] = offset[:,1] * width
        
        attention = offset_attention[:,[2]]
        attention = torch.exp(attention)
        attention = attention.reshape(batch_size, 1, height*width)
        
        destination_y = torch.arange(height, device=x.device)
        destination_y = destination.view(1,1,height,1)
        destination_y = destination_y.expand(batch_size,1,height,width)
        destination_x = torch.arange(width, device=x.device)
        destination_x = destination_x.view(1,1,1,width)
        destination_x = destination_x.expand(batch_size,1,height,width)
        destination = torch.cat((destination_y, destination_x), dim=1)
        destination = torch.round(destination + offset).long()
        destination_unroll = destination.reshape(batch_size, 2, height*width)
        
        x = x.reshape(batch_size, channels, height*width) * attention
        feature_accumulator = torch.zeros(x.shape, device=x.device)
        feature_accumulator.index_add_(-1, destination_unroll, x)
        
        attention_accumulator = torch.zeros(
                (batch_size, 1, height*width), device=x.device).fill_(EPS)
        attention_accumulator.index_add_(-1, destination_unroll, attention)
        
        x = feature_accumulator / attention_accumulator
        
        return x, offset
