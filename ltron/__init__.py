try:
    from gym.envs.registration import register
    gym_available = True
except ModuleNotFoundError:
    gym_available = False

# do this once we finalize the envs for the first paper
'''
register(
    id='brick-graph-8-v0',
    entry_point='brick_gym.envs:TrainRandomStackGraphEnvVec8'
)
'''
