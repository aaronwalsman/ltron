from gymnasium.envs.registration import register

def register_ltron_envs():
    #register(
    #    id='LTRON/SelectConnectionPoint-v0',
    #    entry_point='ltron.gym.envs:SelectConnectionPointEnv',
    #)
    
    register(
        id='LTRON/Freebuild-v0',
        entry_point='ltron.gym.envs:InterfaceEnv',
    )
    
    register(
        id='LTRON/Detect-v0',
        entry_point='ltron.gym.envs:DetectEnv',
    )
    
    register(
        id='LTRON/Break-v0',
        entry_point='ltron.gym.envs:BreakEnv',
    )
    
    register(
        id='LTRON/BreakWithExpert-v0',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={'env_name':'LTRON/Break-v0'},
    )
    
    #register(
    #    id='LTRON/BreakEval-v0',
    #    entry_point='ltron.gym.envs:BreakEnv',
    #    kwargs={'train':False},
    #)
    #register(
    #    id='LTRON/BreakTrain-v0',
    #    entry_point='ltron.gym.wrappers.build_expert:wrapped_build_expert',
    #    kwargs={'env_name':'LTRON/Break-v0', 'train':True},
    #)
    
    register(
        id='LTRON/Make-v0',
        entry_point='ltron.gym.envs:MakeEnv',
    )
    
    register(
        id='LTRON/MakeWithExpert-v0',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={'env_name':'LTRON/Make-v0'},
    )
    
    #register(
    #    id='LTRON/IdentifyRedBrick-v0',
    #    entry_point='ltron.gym.envs:ColoredBrickPrediction',
    #)
