import os
import glob
from pathlib import Path

import numpy

from ltron.home import get_ltron_home
from ltron import settings
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.brick_shape import BrickShape
#from ltron.dataset.submodel_extraction import blacklist_computation
import json
import os

import tqdm

def get_blacklist_path():
    return os.path.join(get_ltron_home(), 'blacklist.json')

def add_large_bricks_to_blacklist(threshold):
    #path = Path("~/.cache/ltron/ldraw/parts").expanduser()
    #partlist = list(path.glob("*.dat"))
    part_path = os.path.join(settings.paths['ldraw'], 'parts')
    part_list = glob.glob(os.path.join(part_path, '*.dat'))
    
    blacklist = []
    print('-'*80)
    print('Finding Large Bricks to Blacklist')
    for part in tqdm.tqdm(part_list):
        part = str(part)
        if "30520.dat" in part:
            continue
        bshape = BrickShape(part)
        max_dim = numpy.max(bshape.bbox[1] - bshape.bbox[0])
        if max_dim > threshold:
            blacklist.append(bshape.reference_name)
    
    blacklist_path = get_blacklist_path()
    blacklist_data = json.load(open(blacklist_path))
    if 'large_%i'%threshold not in blacklist_data:
        blacklist_data['large_%i'%threshold] = []
    
    for part in blacklist:
        if part not in blacklist_data['large_%i'%threshold]:
            blacklist_data['large_%i'%threshold].append(os.path.split(part)[-1])
    
    with open(blacklist_path, 'w') as f:
        json.dump(blacklist_data, f, indent=2)

def remove_blacklisted_parts(
    source_directory,
    dest_directory,
    #blacklist_dest,
    threshold=400,
    #blacklist=None,
    overwrite=False,
):
    
    #if blacklist is None:
    #    blacklist = []
    
    '''
    blacklist_path = os.path.join(blacklist_dest, 'blacklist_%i.json'%threshold)
    if os.path.exists(blacklist_path):
        with open(blacklist_path) as f:
            blacklist = blacklist + json.load(f)
    else:
        blacklist = blacklist + blacklist_computation(threshold)
    '''
    #with open(blacklist_path, 'w') as f:
    #    json.dump(blacklist, f)
    
    blacklist_path = get_blacklist_path()
    blacklist_data = json.load(open(blacklist_path))
    if 'large_%i'%threshold not in blacklist_data:
        add_large_bricks_to_blacklist(threshold)
    blacklist_data = json.load(open(blacklist_path))
    blacklisted_bricks = set(
        blacklist_data['all'] + blacklist_data['large_%i'%threshold])
    
    source_directory = Path(source_directory).expanduser()
    model_list = list(source_directory.rglob('*'))
    print('-'*80)
    print('Removing Blacklisted Bricks From Scenes')
    scene = BrickScene(track_snaps=False)
    for model_src_path in tqdm.tqdm(model_list):
        model_src_path = str(model_src_path)
        model_name = os.path.split(model_src_path)[-1]
        model_dest_path = os.path.join(dest_directory, model_name)
        if os.path.exists(model_dest_path) and not overwrite:
            continue
        
        scene.clear_instances()
        try:
            scene.import_ldraw(model_src_path)
        except:
            print("Can't open: " + model_src_path + " during blacklisting")
            continue

        #keep = [i+1 for i in range(len(scene.instances)) if scene.instances.instances[i+1].brick_shape.reference_name not in blacklist]
        keep = [
            instance for i, instance in scene.instances.items()
            if str(instance.brick_shape) not in blacklisted_bricks
        ]
        # for i in range(len(scene.instances)):
        #     try:
        #         if scene.instances.instances[i+1].brick_shape.reference_name in blacklist:
        #             scene.remove_instance(i+1)
        #     except:
        #         print(scene.instances.instances)
        #         print(i)
        #         print(scene.instances.instances[i+1].brick_shape)
        scene.export_ldraw(
            model_dest_path, instances=keep)
