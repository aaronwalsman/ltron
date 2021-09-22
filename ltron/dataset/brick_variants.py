import re
import os
import glob
import json

import tqdm

from ltron.home import get_ltron_home
import ltron.settings as settings

from ltron.bricks.brick_scene import BrickScene

def is_variant(brick_type):
    name, ext = brick_type.split('.')
    return not name.isdigit()

def non_variant(brick_type, all_brick_types):
    #nv = re.sub('[a-z].*', '', brick_type).replace('.', '') + '.dat'
    try:
        nv  = re.search('[a-z]*[0-9]+', brick_type)[0] + '.dat'
    except TypeError:
        return brick_type
    if nv in all_brick_types:
        return nv
    else:
        return brick_type

def make_brick_variant_tables():
    # get all parts
    parts_path = os.path.join(settings.paths['ldraw'], 'parts', '*.dat')
    all_brick_types = set(os.path.split(p)[-1] for p in glob.glob(parts_path))
    
    # build the tables
    non_variant_to_variant = {}
    variant_to_non_variant = {}
    for brick_type in all_brick_types:
        non_variant_type = non_variant(brick_type, all_brick_types)
        non_variant_to_variant.setdefault(non_variant_type, [])
        non_variant_to_variant[non_variant_type].append(brick_type)
        variant_to_non_variant[brick_type] = non_variant_type
    
    variant_tables = {
        'non_variant_to_variant':non_variant_to_variant,
        'variant_to_non_variant':variant_to_non_variant,
    }
    
    # write the tables to disk
    variant_table_path = os.path.join(get_ltron_home(), 'variant_tables.json')
    with open(variant_table_path, 'w') as f:
        json.dump(variant_tables, f, indent=2)

def remap_variants(original_scene, variant_tables):
    remapped_scene = BrickScene()
    for i, instance in original_scene.instances.items():
        brick_type = str(instance.brick_type)
        remapped_brick_type = variant_tables[
            'variant_to_non_variant'][brick_type]
        remapped_scene.add_instance(
            remapped_brick_type, instance.color, instance.transform)
    
    return remapped_scene

def remap_variant_paths(paths, out_path, variant_tables=None, overwrite=False):
    if variant_tables is None:
        variant_table_path = os.path.join(
            get_ltron_home(), 'variant_tables.json')
        variant_tables = json.load(open(variant_table_path))
    
    if os.path.isdir(paths):
        #paths = [os.path.join(paths, p) for p in os.listdir(paths)]
        paths = (
            glob.glob(os.path.join(paths, '*.ldr')) +
            glob.glob(os.path.join(paths, '*.mpd'))
        )
    
    for path in tqdm.tqdm(paths):
        file_name = os.path.split(path)[-1]
        write_path = os.path.join(out_path, file_name)
        if not overwrite and os.path.exists(write_path):
            continue
        original_scene = BrickScene()
        original_scene.import_ldraw(path)
        remapped_scene = remap_variants(original_scene, variant_tables)
        remapped_scene.export_ldraw(write_path)

if __name__ == '__main__':
    #make_brick_variant_tables()
    
    src_path = (
        '/gscratch/raivn/muruz/ltron/ltron/dataset/subcomponents4')
    dest_path = (
        '/gscratch/raivn/awalsman/.cache/ltron/collections/omr_split_4/ldraw')
    
    remap_variant_paths(src_path, dest_path)
    
