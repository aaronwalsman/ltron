#!/usr/bin/env python
import os

import tqdm

import numpy

from splendor.obj_mesh import load_mesh

import ltron.settings as settings
from ltron.ldraw.documents import LDrawDocument

def bbox(vertices):
    vertex_min = numpy.min(vertices, axis=0)
    vertex_max = numpy.max(vertices, axis=0)
    
    return numpy.concatenate((vertex_min, vertex_max), axis=0)

def doc_bbox(doc):
    pass

ldraw_part_directory = os.path.join(settings.paths['ldraw'], 'parts')
obj_directory = os.path.join(settings.paths['splendor'], 'meshes')
obj_files = [f for f in os.listdir(obj_directory) if f.endswith('.obj')]

for obj_file in tqdm.tqdm(obj_files):
    obj_path = os.path.join(obj_directory, obj_file)
    obj_mesh = load_mesh(obj_path)
    try:
        obj_bbox = bbox(obj_mesh['vertices'])
    except ValueError:
        print('-'*80)
        print('bad obj')
        print(obj_file)
    
    dat_file = obj_file.replace('.obj', '.dat')
    dat_path = os.path.join(ldraw_part_directory, dat_file)
    dat_doc = LDrawDocument.parse_document(dat_path)
    try:
        dat_bbox = bbox(dat_doc.get_all_vertices()[:3].T)
    except ValueError:
        print('-'*80)
        print('bad dat')
        print(dat_file)
        continue
    
    bbox_diff = dat_bbox - obj_bbox
    bbox_max_abs = numpy.max(numpy.abs(bbox_diff))
    if bbox_max_abs > 1.0:
        print('-------')
        print(obj_file)
        print(obj_bbox)
        print(dat_bbox)
