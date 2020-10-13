#!/usr/bin/env python
import os
import json

import ldraw_renderpy as lr

import brick_gym.config as config

carbon_star_path = os.path.join(
        config.paths['omd'], '8661-1 - Carbon Star.mpd')
slave_one_path = os.path.join(
        config.paths['omd'], '75060 - Slave I.mpd')

renderpy_data = lr.mpd_to_renderpy(
        open(slave_one_path),
        image_light_directory = '/home/awalsman/Development/renderpy/renderpy/example_image_lights/marienplatz')

with open('./slave_one.json', 'w') as f:
    json.dump(renderpy_data, f)
