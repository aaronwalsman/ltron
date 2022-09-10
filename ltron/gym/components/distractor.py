import random

from gym.spaces import MultiDiscrete

from ltron.gym.components.sensor_component import SensorComponent

class DistractorComponent(SensorComponent):
    def __init__(self,
        num_tokens,
        num_classes,
        update_frequency,
        observable=True,
    ):
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        if self.observable:
            #self.observation_space = Box(
            #    low=0,
            #    high=num_classes-1,
            #    shape=(num_tokens,),
            #    dtype=numpy.int64,
            #)
            self.observation_space = MultiDiscrete((num_classes,)*num_tokens)
    
    def update_observation(self):
        observation = [
            random.randint(0, self.num_classes-1)
            for _ in range(self.num_tokens)
        ]
        self.observation = numpy.array(observation, dtype=numpy.int64)
