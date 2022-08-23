import json

import tqdm

from splendor.json_numpy import NumpyEncoder

from ltron.ldraw.parts import LDRAW_PARTS, LDRAW_PATHS, LDRAW_BLACKLIST_ALL
from ltron.bricks.brick_shape import BrickShape

detail_types = {}
subtype_ids = {}

try:
    with open('snap_types.json', 'r') as f:
        data = json.load(f)
        detail_types = data['detail']
        subtype_ids = data['subtype']

except FileNotFoundError:
    parts = list(LDRAW_PARTS - LDRAW_BLACKLIST_ALL)
    iterate = tqdm.tqdm(parts)
    for part in iterate:
        bt = BrickShape(part)
        
        for i, snap in enumerate(bt.snaps):
            raw_data = snap.raw_data()
            del(raw_data['transform'])
            detail_string = json.dumps(raw_data, cls=NumpyEncoder)
            if detail_string not in detail_types:
                detail_types[detail_string] = []
            detail_types[detail_string].append((part, i))
            
            if snap.subtype_id not in subtype_ids:
                subtype_ids[snap.subtype_id] = []
            subtype_ids[snap.subtype_id].append((part, i))
        
        iterate.set_description(
            'd:%i, s:%i'%(len(detail_types), len(subtype_ids)))

    with open('snap_types.json', 'w') as f:
        json.dump({'detail': detail_types, 'subtype': subtype_ids}, f)

detail_counts = list(sorted((len(v), k) for k,v in detail_types.items()))
subtype_counts = list(sorted((len(v), k) for k,v in subtype_ids.items()))

#scale_things = set()
#group_things = set()
#non_generic_group_things = set()
snap_types = {}
cylinder_pieces = {}
for detail_string in detail_types:
    detail_data = json.loads(detail_string)
    #if detail_data['scale'] != 'none':
    #    scale_things.add(detail_string)
    #if detail_data['group'] is not None:
    #    group_things.add(detail_string)
    #if (detail_data['group'] is not None and 
    #   detail_data['snap_type'] != 'generic'):
    #    non_generic_group_things.add(detail_string)
    snap_type = detail_data['snap_type']
    if snap_type not in snap_types:
        snap_types[snap_type] = set()
    snap_types[snap_type].add(detail_string)
    
    if detail_data['snap_type'] == 'cylinder':
        for t, r, l in zip(
            detail_data['sec_type'],
            detail_data['sec_radius'],
            detail_data['sec_length']):
            cylinder_description = (
                t, r, l, detail_data['polarity'], detail_data['slide'])
            if cylinder_description not in cylinder_pieces:
                cylinder_pieces[cylinder_description] = []
            cylinder_pieces[cylinder_description].extend(
                detail_types[detail_string])

for snap_type, values in snap_types.items():
    print('%s: %i'%(snap_type, len(values)))

order = sorted(
    (len(pieces), cylinder) for cylinder, pieces in cylinder_pieces.items())
for i, c in order:
    print(c, i)

import pdb
pdb.set_trace()
