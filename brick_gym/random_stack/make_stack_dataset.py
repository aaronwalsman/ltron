#!/usr/bin/env python
import os

import brick_gym.config as config
import brick_gym.random_stack.random_stack as random_stack

random_stack.sample_dataset(config.paths['random_stack'])
