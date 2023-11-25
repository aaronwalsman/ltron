from gymnasium.envs.registration import register

def register_ltron_envs():
    #register(
    #    id='LTRON/SelectConnectionPoint-v0',
    #    entry_point='ltron.gym.envs:SelectConnectionPointEnv',
    #)
    
    register(
        id='LTRON/Freebuild-v0',
        entry_point='ltron.gym.envs:FreebuildEnv',
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
    
    register(
        id='LTRON/BreakAndMake-v1',
        entry_point='ltron.gym.envs.break_and_make_env:BreakAndMakeEnv',
    )
    
    #register(
    #    id='LTRON/BreakAndMakeStepWrapper-v1',
    #    entry_point='ltron.gym.wrappers.break_and_make_step_wrapper:'
    #        'break_and_make_step_wrapper_env',
    #)
    
    register(
        id='LTRON/SteppedBreakAndMake-v1',
        entry_point=
            'ltron.gym.envs.stepped_break_and_make_env:SteppedBreakAndMakeEnv'
    )
    
    register(
        id='LTRON/SteppedBreakAndMakeWithExpert-v1',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={'env_name':'LTRON/SteppedBreakAndMake-v1'},
    )
    
    register(
        id='LTRON/MakeAndBreakAutoStepPhaseClick-v1',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={
            'env_name':'LTRON/SteppedBreakAndMake-v1',
            'execute_expert_primitives':(
                'remove',
                'assemble_step',
                'phase',
                'rotate',
                'translate',
                'pick_and_place',
            ),
        },
    )
    
    register(
        id='LTRON/MakeAndBreakAutoStepPhaseInsert-v1',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={
            'env_name':'LTRON/SteppedBreakAndMake-v1',
            'execute_expert_primitives':(
                'assemble_step',
                'phase',
                'insert',
            ),
        },
    )
    
    register(
        id='LTRON/SteppedBreakAndMakeAutoPhase-v1',
        entry_point='ltron.gym.wrappers.build_step_expert:'
            'wrapped_build_step_expert',
        kwargs={
            'env_name':'LTRON/SteppedBreakAndMake-v1',
            'execute_expert_primitives':{
                'assemble_step',
                'remove',
                'phase',
            },
            'execute_expert_phase':0,
        },
    )
    
    #register(
    #    id='LTRON/IdentifyRedBrick-v0',
    #    entry_point='ltron.gym.envs:ColoredBrickPrediction',
    #)
