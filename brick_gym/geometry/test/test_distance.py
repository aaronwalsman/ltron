#!/usr/bin/env python
import time
import numpy

t0 = time.time()
for _ in range(100000):
    a = numpy.random.rand(3)
    b = numpy.random.rand(3)
    c = numpy.random.rand()
    d = numpy.linalg.norm(a-b) < c
print(time.time() - t0)

# wow, ok, this is actually quite a bit faster
t0 = time.time()
for _ in range(100000):
    a = numpy.random.rand(3)
    b = numpy.random.rand(3)
    c = numpy.random.rand()
    d = sum(aa-bb for aa,bb in zip(a,b)) < c**2
print(time.time() - t0)
