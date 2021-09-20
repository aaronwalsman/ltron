import re
import os
import glob

import tqdm

import ltron.settings as settings

from ltron.bricks.brick_scene import BrickScene

omr_path = os.path.join(settings.collections['omr'], 'ldraw')
all_files = glob.glob(os.path.join(omr_path, '*.mpd'))

all_brick_types = set()

for file_path in tqdm.tqdm(all_files):
    scene = BrickScene(renderable=False, track_snaps=False)
    scene.import_ldraw(file_path)
    brick_types = set(str(brick_type) for brick_type in scene.brick_library)
    all_brick_types |= brick_types

def is_variant(brick_type):
    name, ext = brick_type.split('.')
    return not name.isdigit()

def non_variant(brick_type):
    return re.sub('[a-z].*', '', brick_type)

deduplicated_brick_types = {}
for brick_type in all_brick_types:
    non_variant_type = non_variant(brick_type)
    deduplicated_brick_types.setdefault(non_variant_type, [])
    deduplicated_brick_types[non_variant_type].append(brick_type)

import pdb
pdb.set_trace()
