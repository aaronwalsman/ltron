import numpy
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class TallestTower(LtronGymComponent):
    def __init__(self, scenecomponent):
        self.scenecomponent = scenecomponent

    def reset(self):
        return None

    def compute_reward(self):

        instance_tran = {}
        for k, v in self.scenecomponent.brick_scene.instances.instances.items():
            instance_tran[k] = v.transform

        instance_pos = {}
        for k, v in self.scenecomponent.brick_scene.instances.instances.items():
            instance_pos[k] = v.brick_shape.bbox

        point = []
        for ins, bbox in instance_pos.items():
            minb = bbox[0]
            maxb = bbox[1]
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], maxb[2], 1])))[:3])

        min_y = 100000
        max_y = -1000000
        for p in point:
            if p[1] > max_y:
                max_y = p[1]
            if p[1] < min_y:
                min_y = p[1]

        # if abs(max_y - min_y) - 35 > 0: return 10000
        # else: return -1000
        return abs(max_y - min_y)

    def step(self, action):
        return None, self.compute_reward(), False, None

