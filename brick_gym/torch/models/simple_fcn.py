import torch
import torchvision.models as tv_models

import brick_gym.torch.models.resnet as resnet

class SimpleBlock(torch.nn.Module):
    def __init__(self, decoder_channels, skip_channels):
        super(SimpleBlock, self).__init__()
        self.skip_conv = torch.nn.Conv2d(
                skip_channels, decoder_channels, kernel_size=1)
    
    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class SimpleDecoder(torch.nn.Module):
    def __init__(self,
            encoder_channels,
            decoder_channels=256,
            decoder_depth=4):
        super(SimpleDecoder, self).__init__()
        
        layer = torch.nn.Conv2d(
                encoder_channels[0], decoder_channels, kernel_size=1)
        layers = [layer]
        for i in range(1, decoder_depth):
            layer = SimpleBlock(decoder_channels, encoder_channels[i])
            layers.append(layer)
        
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, *features):
        x = self.layers[0](features[0])
        for layer, feature in zip(self.layers[1:], features[1:]):
            x = layer(x, feature)
        
        return x
        

class SimpleFCN(torch.nn.Module):
    def __init__(self, pretrained=True, decoder_channels=256):
        super(SimpleFCN, self).__init__()
        backbone = tv_models.resnet50(pretrained=pretrained)
        backbone = resnet.ResnetBackbone(backbone, fcn=True)
        self.encoder = backbone
        self.decoder = SimpleDecoder(
                encoder_channels=(2048, 1024, 512, 256),
                decoder_channels=decoder_channels)
    
    def forward(self, x):
        xn = self.encoder(x)
        x = self.decoder(*xn)
        
        return x
