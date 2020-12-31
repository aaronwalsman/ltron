#!/usr/bin/env python
import time
import random

from brick_gym.geometry.grid_bucket import GridBucket

bucket = GridBucket(cell_size = 4.0)

t0 = time.time()
for i in range(50000):
    xyz = tuple(random.random() * 100 for _ in range(3))
    bucket.insert('thing_%i'%i, xyz)
    
    xyz_eps = tuple(x + 1 for x in xyz)
    values = bucket.lookup(xyz, 0.0000001)
    assert 'thing_%i'%i in values
t1 = time.time()
print('elapsed: %.04f'%(t1-t0))
