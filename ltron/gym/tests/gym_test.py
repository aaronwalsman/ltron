from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.render import ColorRenderComponent
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent,
        RandomizedAzimuthalViewpointComponent,
        FixedAzimuthalViewpointComponent)
from ltron.gym.components.render import SnapRenderComponent
from ltron.gym.components.build_tower import PickandPlace
import math
import collections
import numpy

components = collections.OrderedDict()

components['dataset'] = DatasetPathComponent('random_six', 'all')
dataset_info = components['dataset'].dataset_info
print(dataset_info)
max_instances = dataset_info['max_instances_per_scene']
num_classes = max(dataset_info['class_ids'].values()) + 1
# obser = dataset.observe()
# print(obser)
# print(dataset.dataset_info)
# print(dataset.length)
# print(dataset.dataset_ids)

components['scene'] = SceneComponent(dataset_component = None,
                                     initial_scene_path= '/home/nanami/.cache/ltron/collections/omr/ldraw/bricklist.mpd',
                                     path_location = components['dataset'].dataset_paths,
                                     track_snaps = True)
components['episode'] = MaxEpisodeLengthComponent(1000)
components['camera'] = FixedAzimuthalViewpointComponent(
                components['scene'],
                azimuth = math.radians(-135.),
                elevation = math.radians(-30.),
                aspect_ratio = 1,
                distance = (2, 2))
components['neg_snap'] = SnapRenderComponent(256, 256, components['scene'], polarity = "-")
components['pos_snap'] = SnapRenderComponent(256, 256, components['scene'], polarity = '+')
components['pick'] = PickandPlace(components['dataset'], components['scene'], components['pos_snap'], components['neg_snap'])
components['render'] = ColorRenderComponent(256, 256, components['scene'])
# print(render.reset())
# imshow(render.reset())
# plt.show()
env = LtronEnv(components)
#print(list(components['scene'].brick_scene.instances))
# components['scene'].brick_scene.import_ldraw('/home/nanami/.cache/ltron/collections/omr/ldraw/bricklist.mpd')
obs = env.reset()
imshow(obs['render'])
plt.show()
print(numpy.sum(obs['pos_snap']))
# print(numpy.max(obs['snap']))
# print(numpy.sum(obs['snap']))
# print(obs['snap'].shape)
# imshow(obs['render'])
# plt.show()
brickscene = components['scene'].brick_scene
print(brickscene.instances.instances)
instance_pos = {}
for k, v in brickscene.instances.instances.items():
    instance_pos[k] = v.brick_type.bbox
print(instance_pos)
instance_tran = {}
for k, v in brickscene.instances.instances.items():
    instance_tran[k] = v.transform
print(list(components['scene'].brick_scene.instances))
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

print(point)
min_y = 100000
max_y = -1000000
for p in point:
    if p[1] > max_y:
        max_y = p[1]
    if p[1] < min_y:
        min_y = p[1]
print(abs(max_y - min_y))
# imshow(obs['render'])
# components['scene'].brick_scene.import_ldraw("~/.cache/ltron/collections/random_six/ldraw/random_six_000000.mpd")
# print(list(scene.brick_scene.instances))
loc = numpy.array([[1, 0, 0, 10],
                   [0, 1, 0, 10],
                   [0, 0, 1, 10],
                   [0, 0, 0, 1]])
components['scene'].brick_scene.add_instance('3003.dat', 128, numpy.eye(4))
# print(list(brickscene.instances))
print(brickscene.instances.instances)
instance_pos = {}
for k, v in brickscene.instances.instances.items():
    instance_pos[k] = v.brick_type.bbox
print(instance_pos)
instance_tran = {}
for k, v in brickscene.instances.instances.items():
    instance_tran[k] = v.transform
print(instance_tran)
# print(brickscene.get_all_snaps())
# print(brickscene.get_all_snap_connections())
obs, reward, term, info = env.step({'pick' : None})
imshow(obs['render'])
plt.show()
snap = components['scene'].brick_scene.get_all_snaps()
obs, reward, term, info = env.step({'pick' : [1, [128, 128], [142, 142]]})
instance_pos = {}
for k, v in components['scene'].brick_scene.instances.instances.items():
    instance_pos[k] = v.brick_type.bbox
print(instance_pos)
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

print(point)
min_y = 100000
max_y = -1000000
for p in point:
    if p[1] > max_y:
        max_y = p[1]
    if p[1] < min_y:
        min_y = p[1]
print(abs(max_y - min_y))
point = []
for k, v in brickscene.instances.instances.items():
    instance_tran[k] = v.transform
print(instance_tran)

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
print(point)
min_y = 100000
max_y = -1000000
for p in point:
    if p[1] > max_y:
        max_y = p[1]
    if p[1] < min_y:
        min_y = p[1]
print(abs(max_y - min_y))
# print(reward)
print(obs['pick'])
pick_map = components['pos_snap'].observation
#print(numpy.where(pick_map > 1))
imshow(obs['render'])
plt.show()


brickscene.add_instance("3003.dat", 6, loc)
obs, reward, term, info = env.step({'pick' : None})
imshow(obs['render'])
plt.show()

point = []
for k, v in brickscene.instances.instances.items():
    instance_tran[k] = v.transform
print(instance_tran)

for k, v in components['scene'].brick_scene.instances.instances.items():
    instance_pos[k] = v.brick_type.bbox

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
print(point)
min_y = 100000
max_y = -1000000
for p in point:
    if p[1] > max_y:
        max_y = p[1]
    if p[1] < min_y:
        min_y = p[1]
print(abs(max_y - min_y))
# print(brickscene.get_instance_snap_connections(1))
# brickscene.pick_and_place_snap((3,2), (4, 1))
# obs, reward, term, info = env.step(None)
# imshow(obs['render'])
# plt.show()