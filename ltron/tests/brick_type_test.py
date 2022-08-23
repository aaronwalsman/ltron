import time

import tqdm

from splendor.json_numpy import NumpyEncoder

from ltron.bricks.brick_shape import BrickShape

from ltron.dataset.paths import get_dataset_info
from ltron.dataset.parts import all_ldraw_parts

t0 = time.time()

info = get_dataset_info('omr_clean')

brick_data = {}

for brick_shape_name in tqdm.tqdm(all_ldraw_parts()):
    bt = BrickShape(brick_shape_name)
    data = {}
    data['reference_name'] = bt.reference_name
    data['mesh_name'] = bt.mesh_name
    data['bbox'] = bt.bbox
    data['bbox_vertices'] = bt.bbox_vertices
    data['snaps'] = [snap.raw_data() for snap in bt.snaps]
    brick_data[brick_shape_name] = data

print(time.time() - t0)

import pdb
pdb.set_trace()
