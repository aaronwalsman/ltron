#!/usr/bin/env python
import time
import os

import brick_gym.config as config
from brick_gym.ldraw.documents import *
import brick_gym.ldraw.snap as snap

import brick_gym.ldraw.mpd as mpd

t0 = time.time()
d = LDrawDocument.parse_document(os.path.join(
        config.paths['omr'], '10030-1 - Imperial Star Destroyer - UCS.mpd'))
print('load elapsed: %f'%(time.time() - t0))

t0 = time.time()
parts = mpd.parts_from_mpd(open(os.path.join(
        config.paths['omr'], '10030-1 - Imperial Star Destroyer - UCS.mpd')),
        os.listdir(os.path.join(config.paths['ldraw'], 'parts')))
print('old load elapsed: %f'%(time.time() - t0))

#d = LDrawDocument.parse_document('3003.dat')
t0 = time.time()
snap_points = snap.snap_points_from_document(d)
print('snap elapsed: %f'%(time.time() - t0))

'''
for p in snap_points:
    print(p)
    print(p.id)
    #print(p.transform)
'''

'''
print(d)
print(d.reference_table['shadow'].keys())
for command in d.reference_table['shadow']['s/3003s01.dat'].commands:
    print(command)
'''
