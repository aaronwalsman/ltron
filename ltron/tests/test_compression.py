#!/usr/bin/env python
import glob

import numpy

import PIL.Image as Image

import ltron.compression as compression

'''
frames = [
    numpy.array(Image.open('helicopter_disassembly/image_%06i.png'%i))
    for i in range(1, 6)
]
image_size=512
'''

frames = numpy.stack([
    numpy.array(Image.open(path))
    for path in sorted(glob.glob('carbon_star_disassembly/brick_viewer*.png'))
], axis=0)
image_size=256

print(frames.shape)

sequence, coordinates = compression.deduplicate_tiled_seq(
    frames, 16, 16, background=255)

n, h, w, c = sequence.shape

stitched_images = {}

for i in range(n):
    tile = sequence[i]
    t,s = coordinates[i]
    folder_name = './tiles_%04i'%i
    Image.fromarray(tile).save('./tile_%i_%i.png'%(t,s))
    
    if t not in stitched_images:
        stitched_images[t] = numpy.zeros(
            (image_size,image_size,4), dtype=numpy.uint8)
    
    y = s // (image_size // 16)
    x = s % (image_size // 16)
    stitched_images[t][y*16:(y+1)*16, x*16:(x+1)*16,:3] = tile
    stitched_images[t][y*16:(y+1)*16, x*16:(x+1)*16,3] = 255

for t, stitched_image in stitched_images.items():
    Image.fromarray(stitched_image).save('stitched_%04i.png'%t)
