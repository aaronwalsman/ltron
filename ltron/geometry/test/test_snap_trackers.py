import time
import os

import numpy

from scipy.spatial import cKDTree

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.grid_bucket import GridBucket

destroyer_path = os.path.join(settings.collections['omr'], 'ldraw',
    '10030-1 - Imperial Star Destroyer - UCS.mpd')
    #'7657-1 - AT-ST.mpd')

scene = BrickScene()
scene.import_ldraw(destroyer_path)

snaps = []
for i, instance in scene.instances.items():
    snaps.extend(instance.snaps)

snap_positions = [snap.transform[:3,3] for snap in snaps]

min_box = numpy.min(snap_positions, axis=0)
max_box = numpy.max(snap_positions, axis=0)

print(min_box, max_box)
b = 10
r = 10
n = 50

t0 = time.time()
bucket = GridBucket(b)
bucket.insert_many(range(len(snaps)), snap_positions)
t1 = time.time()
print('GridBucket build elapsed: %f'%(t1-t0))

t0 = time.time()
for i in range(10000):
    p = numpy.random.uniform(min_box, max_box)
    values = bucket.lookup(p, r)
t1 = time.time()
print('GridBucket query elapsed: %f'%(t1-t0))

t0 = time.time()
for snap_position in snap_positions:
    p = snap_position + numpy.random.uniform([-n,-n,-n], [n,n,n])
    values = bucket.lookup(p, r)
t1 = time.time()
print('GridBucket nearby query elapsed: %f'%(t1-t0))

t0 = time.time()
tree = cKDTree(snap_positions)
t1 = time.time()
print('KDTree build elapsed: %f'%(t1-t0))

t0 = time.time()
for i in range(10000):
    p = numpy.random.uniform(min_box, max_box)
    values = tree.query_ball_point(p, r, eps=0.5)
t1 = time.time()
print('KDTree query elapsed: %f'%(t1-t0))

t0 = time.time()
for snap_position in snap_positions:
    p = snap_position + numpy.random.uniform([-n,-n,-n], [n,n,n])
    values = tree.query_ball_point(p, r)
t1 = time.time()
print('KDTree nearby query elapsed: %f'%(t1-t0))
