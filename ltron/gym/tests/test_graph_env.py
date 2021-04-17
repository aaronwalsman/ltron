#!/usr/bin/env python
import numpy
import brick_gym.envs.standard_envs as standard_envs

train_graph_env = standard_envs.graph_env(
        'random_stack',
        'train_mpd',
        train=True)

observations = []
observation = train_graph_env.reset()
observations.append(observation)

terminal = False
step = 0
import time
t0 = time.time()
while not terminal:
    prediction = {
            'nodes' : numpy.zeros(8, dtype=numpy.long),
            'edges' : numpy.zeros((8,8), dtype=numpy.float)}
    action = {'visibility':step, 'graph':prediction}
    observation, reward, terminal, info = train_graph_env.step(action)
    step += 1
    observations.append(observation)

print(len(observations))
