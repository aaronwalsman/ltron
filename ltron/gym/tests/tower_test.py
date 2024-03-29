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
from ltron.gym.components.build_tower import TallestTower
from ltron.gym.components.pick_and_place import PickandPlace
from ltron.gym.tower_env import TowerEnv
import math
import collections
import numpy
from ltron.gym.components.rotation import RotationAroundSnap

import PIL.Image as Image

#bricklist = '/home/awalsman/.cache/ltron/collections/omr/bricklist.mpd'
bricklist = '/home/nanami/.cache/ltron/collections/omr/ldraw/bricklist.mpd'
components = collections.OrderedDict()
components['scene'] = SceneComponent(dataset_component = None,
                                     initial_scene_path = bricklist,
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
components['pick'] = PickandPlace(components['scene'], components['pos_snap'], components['neg_snap'])
components['rotation'] = RotationAroundSnap(components['scene'], components['pos_snap'], components['neg_snap'])
components['tower'] = TallestTower(components['scene'])
components['render'] = ColorRenderComponent(512, 512, components['scene'])

env = LtronEnv(components)
# env = TowerEnv(components)
obs = env.reset()
# imshow(obs['render'])
plt.show()
#Image.fromarray(obs['render']).save('image_0001.png')

obs, reward, term, info = env.step({'pick' : None, 'rotation' : None})
# obs, reward, term, info = env.step(None)
print(reward)

neg_map = components['neg_snap'].observation
pos_map = components['pos_snap'].observation

# for i in range(2):
#     action = env.action_space.sample()
#     print(action)
#     obs, reward, term, info = env.step(action)
#     if reward > 34:
#         print(reward)
#         imshow(obs)

# print(numpy.where(numpy.any(pos_map, axis = 2)))
# print(numpy.where(numpy.any(neg_map, axis = 2)))
# print(numpy.where(pos_map == [3,1]))
# print(numpy.where(neg_map == [2,11]))
# print(neg_map[113,125])

obs, reward, term, info = env.step({'pick' : [0, 118, 132, 123, 129], 'rotation' : None})
# obs, reward, term, info = env.step([1, 109, 112, 153, 151])
# imshow(obs['render'])
# plt.show()
print(reward)
#Image.fromarray(obs['render']).save('image_0002.png')

obs, reward, term, info = env.step({'pick' : None, 'rotation' : [0, 123, 129, 80]})
imshow(obs['render'])
plt.show()
print(reward)
brick_scene.export_ldraw("random.mpd")
#Image.fromarray(obs['render']).save('image_0003.png')
exit(0)

obs, reward, term, info = env.step({'pick' : None, 'rotation' : [0, 153, 151, -80]})
imshow(obs['render'])
plt.show()
print(reward)
#Image.fromarray(obs['render']).save('image_0004.png')

obs, reward, term, info = env.step({'pick' : None, 'rotation' : [0, 125, 125, -80]})
imshow(obs['render'])
plt.show()
print(reward)
#Image.fromarray(obs['render']).save('image_0005.png')
# print(obs['pick'])
