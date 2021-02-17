import torch
from torch.nn.modules.batchnorm import _BatchNorm

class FrozenBatchNormWrapper:
    def __init__(self, model):
        self.model = model
        for m in self.model.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    
    def train(self, mode=True):
        self.model.train(mode)
        for m in self.model.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
    
    def __getattr__(self, attr):
        print(attr)
        return getattr(self.model, attr)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
