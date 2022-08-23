#!/usr/bin/env python
import glob

import numpy

import PIL.Image as Image

import ltron.compression as compression

frames = numpy.stack([
    numpy.array(Image.open(path))
    for path in sorted(glob.glob('carbon_star_disassembly/brick_viewer*.png'))
], axis=0)
h, w, c = frames.shape[-3:]

frames = numpy.concatenate(
    (frames, numpy.ones((1, h, w, c), dtype=numpy.uint8)*255), axis=0)
frames = frames.reshape(4, 11, h, w, c)
frames = frames.transpose(1, 0, 2, 3, 4)

pad = numpy.array([11,11,11,10])

sequence, coordinates, pad = compression.batch_deduplicate_tiled_seqs(
    frames, pad, 16, 16, background=255)

n, b = sequence.shape[:2]

stitched_images = {}

for i in range(n):
    for j in range(b):
        if i >= pad[j]:
            continue
        tile = sequence[i,j]
        s,y,x = coordinates[i,j]
        Image.fromarray(tile).save('./tile_%i_%i_%i.png'%(s,y,x))
        
        if (s,j) not in stitched_images:
            stitched_images[s,j] = numpy.zeros(
                (h,w,4), dtype=numpy.uint8)
        
        #y = s // (h // 16)
        #x = s % (w // 16)
        stitched_images[s,j][y*16:(y+1)*16, x*16:(x+1)*16,:3] = tile
        stitched_images[s,j][y*16:(y+1)*16, x*16:(x+1)*16,3] = 255

for (s,j), stitched_image in stitched_images.items():
    Image.fromarray(stitched_image).save('stitched_%04i_%04i.png'%(j,s))
