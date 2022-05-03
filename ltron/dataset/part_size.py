import tqdm

import numpy

from ltron.ldraw.parts import LDRAW_PARTS, LDRAW_BLACKLIST_ALL

from ltron.bricks.brick_shape import BrickShape

max_size = 0
iterate = tqdm.tqdm(LDRAW_PARTS - LDRAW_BLACKLIST_ALL)
for part in iterate:
    shape = BrickShape(part)
    bbox = shape.bbox
    size = bbox[1] - bbox[0]
    max_size = max(max_size, numpy.max(numpy.abs(size)))
    iterate.set_description('size: %.01f'%max_size)

print(max_size)
# 1602
