#!/usr/bin/env python
import os
import time
#import sys

import tqdm

import ltron.ldraw.documents as documents
from ltron.bricks.brick_scene import BrickScene

t0 = time.time()

folder = '/home/awalsman/.cache/ltron/collections/omr/ldraw'
file_names = os.listdir(folder)
file_times = []
for file_name in tqdm.tqdm(file_names[:1000]):
    scene = BrickScene()
    file_path = os.path.join(folder, file_name)
    
    t1 = time.time()
    #scene.import_ldraw(file_path)
    doc = documents.LDrawDocument.parse_document(file_path)
    t2 = time.time()
    
    file_times.append(t2-t1)

print('Total time: %f'%(time.time() - t0))
print('Max file time: %f (%s)'%max(zip(file_times, file_names)))
print('Cached dat files: %i'%len(documents.dat_cache))
#print('Cached ldraw files: %i'%len(documents.ref_cache['ldraw']))
#print('Cached shadow files: %i'%len(documents.ref_cache['shadow']))
