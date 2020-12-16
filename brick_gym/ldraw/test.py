#!/usr/bin/env python
import os
import re

import brick_gym.config as config

files = os.listdir(config.paths['omr'])

for f in files:
    file_path = os.path.join(config.paths['omr'], f)
    _, ext = os.path.splitext(f)
    if ext == '.mpd':
        with open(file_path, 'r', encoding='utf-8') as ff:
            lines = list(ff.readlines())
            line = lines[0]
            line = re.sub('[^!-~]+', ' ', line).strip()
            if line[:6] != '0 FILE':
                print('Bad mpd start', f)
                print(line)
