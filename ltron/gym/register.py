from gymnasium.envs.registration import register

def register_ltron_envs():
    register(
        id='LTRON/SelectConnectionPoint-v0',
        entry_point='ltron.gym.envs:SelectConnectionPointEnv',
    )
    
    register(
        id='LTRON/Break-v0',
        entry_point='ltron.gym.envs:BreakEnv',
    )
    
    register(
        id='LTRON/IdentifyRedBrick-v0',
        entry_point='ltron.gym.envs:ColoredBrickPrediction',
    )
