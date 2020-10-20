from gym.envs.registration import register

import brick_gym.config as config

register(
    id='viewpoint-v0',
    entry_point='brick_gym.envs:ViewpointEnv'
)
