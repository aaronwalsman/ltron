from collections import OrderedDict

from ltron.dataset.info import get_dataset_info
from ltron.gym.envs.freebuild_env import FreebuildEnvConfig, FreebuildEnv
from ltron.gym.components import DetectObjective

class DetectEnvConfig(FreebuildEnvConfig):
    pass

class DetectEnv(FreebuildEnv):
    def __init__(self,
        config,
        train=False,
    ):
        super().__init__(config, train=train)
        
        self.components['objective'] = DetectObjective(
            self.components['assembly'],
        )
    
    #def step(self, action):
    #    o,r,t,u,i = super().step(action)
    #    return o, r, True, u, i
