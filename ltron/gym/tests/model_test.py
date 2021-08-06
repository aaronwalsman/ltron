import gym
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
from ltron.gym.components.build_tower import (PickandPlace, TallestTower)
import math
import collections
import numpy
from ltron.gym.tower_env import TowerEnv

bricklist = '/home/nanami/.cache/ltron/collections/omr/ldraw/bricklist.mpd'
components = collections.OrderedDict()
components['scene'] = SceneComponent(dataset_component = None,
                                     initial_scene_path = bricklist,
                                     track_snaps = True)
components['episode'] = MaxEpisodeLengthComponent(100)
components['camera'] = FixedAzimuthalViewpointComponent(
                components['scene'],
                azimuth = math.radians(-135.),
                elevation = math.radians(-30.),
                aspect_ratio = 1,
                distance = (2, 2))
components['neg_snap'] = SnapRenderComponent(256, 256, components['scene'], polarity = "-")
components['pos_snap'] = SnapRenderComponent(256, 256, components['scene'], polarity = '+')
components['pick'] = PickandPlace(components['scene'], components['pos_snap'], components['neg_snap'])
components['tower'] = TallestTower(components['scene'])
components['render'] = ColorRenderComponent(256, 256, components['scene'])

env = TowerEnv(components)

from stable_baselines3 import (PPO, DQN, A2C)
from stable_baselines3.common.env_checker import check_env

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if reward > 34:
        print(reward)
    if i % 100 == 0:
        print(action)
        print(reward)
    # env.render()
    if done:
      obs = env.reset()

env.close()


