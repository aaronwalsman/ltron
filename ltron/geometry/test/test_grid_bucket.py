#!/usr/bin/env python
import time
import random

from ltron.geometry.grid_bucket import GridBucket

bucket = GridBucket(cell_size = 4.0)

t0 = time.time()
for i in range(50000):
    xyz = tuple(random.random() * 100 for _ in range(3))
    bucket.insert('thing_%i'%i, xyz)
t1 = time.time()
print('build elapsed: %.04f'%(t1-t0))

for i in range(50000):
    xyz = tuple(random.random() * 100 for _ in range(3))
    values = bucket.lookup(xyz, 1.)
t2 = time.time()
print('query_elapsed: %.04f'%(t2-t1))
