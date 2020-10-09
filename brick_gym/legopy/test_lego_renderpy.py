#!/usr/bin/env python
import lego_renderpy as lr
import json

'''
renderpy_data = lr.mpd_to_renderpy(open('../OMD/8661-1 - Carbon Star.mpd'),
        '../obj',
        '/home/awalsman/Development/renderpy/renderpy/example_image_lights/marienplatz')
'''

'''
renderpy_data = lr.mpd_to_renderpy(open('../OMD/75060 - Slave I.ldr'),
        '../obj',
        '/home/awalsman/Development/renderpy/renderpy/example_image_lights/marienplatz')
'''

renderpy_data = lr.mpd_to_renderpy(open('../simple_set_generator/test.mpd'),
        '../obj',
        '/home/awalsman/Development/renderpy/renderpy/example_image_lights/marienplatz')

with open('./stack.json', 'w') as f:
    json.dump(renderpy_data, f)
