from supermecha import SuperMechaComponent

class MaxInstances(SuperMechaComponent):
    def __init__(self, scene_component, max_instances, mode='truncate'):
        self.scene_component = scene_component
        self.max_instances = max_instances
        self.mode = mode
    
    def step(self, action):
        u = False
        t = False
        if len(self.scene_component.brick_scene.instances) > self.max_instances:
            if self.mode == 'truncate':
                u = True
            elif self.mode == 'terminate':
                t = True
            else:
                raise ValueError('Unknown max-instances mode: %s'%self.mode)

        return None, 0., t, u, {}
