import torch
from torchvision.models import resnet18

import ltron.torch.models.resnet as bg_resnet

class ActionModel(torch.nn.Module):
    def __init__(self):
        super(ActionModel, self).__init__()
        self.backbone = resnet18(pretrained=True).cuda()
        bg_resnet.replace_conv1(self.backbone, 6)
        bg_resnet.make_spatial_attention_resnet(
                self.backbone, shape=(256,256), do_spatial_embedding=True)
        bg_resnet.replace_fc(self.backbone, 1)
    
    def forward(self, x):
        bs, im, c, h, w = x.shape
        x = x.view(bs*im, c, h, w)
        x = self.backbone(x)
        x = x.view(bs, im)
        
        return x

class FeatureActionModel(torch.nn.Module):
    def __init__(self, input_features=512, bias=True):
        super(FeatureActionModel, self).__init__()
        self.model = torch.nn.Linear(input_features,1, bias=bias)
    
    def forward(self, x):
        b, i, c = x.shape
        x = x.view(b*i, c)
        x = self.model(x)
        x = x.view(b, i)
        return x
