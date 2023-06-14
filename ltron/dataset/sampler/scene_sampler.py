import time
from itertools import product
import json
import os
import tarfile
from io import BytesIO

import numpy
random = numpy.random.default_rng(1234567890)

import tqdm

import ltron.settings as settings
from ltron.exceptions import LtronMissingDatasetException
from ltron.dataset.info import get_dataset_info
from ltron.dataset.sampler.subassembly_sampler import get_all_brick_shapes
from ltron.bricks.brick_scene import BrickScene

#def sample_dataset(
#    name,
#    subassembly_samplers,
#    colors,
#    min_max_instances_per_scene,
#    train_scenes,
#    test_scenes,
#):
#    dataset_metadata = {
#        'splits':{}
#    }
#    
#    colors = sorted(colors)
#    
#    max_instances = 0
#    max_edges = 0
#    train_collections = []
#    test_collections = []
#    all_collections = []
#    for min_instances, max_instances in min_max_instances_per_scene:
#        size_collections = []
#        for split, num_scenes in ('train', train_scenes), ('test', test_scenes):
#            split_name = '%i_%i_%s'%(min_instances, max_instances, split)
#            collection_name = '%s_%s'%(name, split_name)
#            size_collections.append(collection_name)
#            all_collections.append(collection_name)
#            if split == 'train':
#                train_collections.append(collection_name)
#            elif split == 'test':
#                test_collections.append(collection_name)
#            collection_max_instances, collection_max_edges = sample_collection(
#                collection_name,
#                subassembly_samplers,
#                colors,
#                min_instances,
#                max_instances,
#                num_scenes,
#            )
#            max_instances = max(max_instances, collection_max_instances)
#            max_edges = max(max_edges, collection_max_edges)
#            
#            dataset_metadata['splits'][split_name] = {
#                'sources':[collection_name],
#            }
#        
#        size_all_split_name = '%i_%i_all'%(min_instances, max_instances)
#        dataset_metadata['splits'][size_all_split_name] = {
#            'sources':size_collections,
#        }
#        
#    dataset_metadata['splits']['train_all'] = train_collections
#    dataset_metadata['splits']['test_all'] = test_collections
#    dataset_metadata['splits']['all'] = all_collections
#    
#    dataset_metadata['max_instances_per_scene'] = max_instances
#    dataset_metadata['max_edges_per_scene'] = max_edges
#    all_shapes = sorted(get_all_brick_shapes(subassembly_samplers))
#    num_shapes = len(all_shapes)
#    dataset_metadata['shape_ids'] = dict(zip(all_shapes, range(1, num_shapes)))
#    dataset_metadata['color_ids'] = dict(zip(colors, range(1, len(colors)+1)))
#    
#    dataset_path = os.path.join(settings.PATHS['datasets'], '%s.json'%name)
#    with open(dataset_path, 'w') as f:
#        json.dump(dataset_metadata, f, indent=2)

def sample_shard(
    dataset_name,
    split_name,
    subassembly_samplers,
    colors,
    min_instances,
    max_instances,
    num_scenes,
    compress=False,
    shard_id=None,
):
    size_name = '%i_%i'%(min_instances, max_instances)
    if shard_id is None:
        shard_name = '%s_%s_%s'%(dataset_name, size_name, split_name)
    else:
        shard_name = '%s_%s_%s_%i'%(
            dataset_name, size_name, split_name, shard_id)
    print('-'*80)
    print('Building Shard: %s'%shard_name)
    
    # retrieve the dataset info
    try:
        dataset_info = get_dataset_info(dataset_name)
    except LtronMissingDatasetException:
        dataset_info = {}
    
    # this is a global class list now
    '''
    # add the color_ids and shape_ids to the dataset info
    colors = sorted(colors)
    shapes = sorted(get_all_brick_shapes(subassembly_samplers))
    for color_shape, values in ('color', colors), ('shape', shapes):
        ids = dict(zip(values, range(1, len(values)+1)))
        # if the ids exist in dataset_info, make sure they match the new ids
        if '%s_ids'%color_shape in dataset_info:
            if dataset_info['%s_ids'%color_shape] != ids:
                raise ValueError('New %ss (%s) does not match existing (%s)'%(
                    color_shape, colors, existing_colors))
        
        # if ids do not exist, add them
        else:
            dataset_info['%s_ids'%color_shape] = ids
    '''
    
    # open the shard tar file
    if compress:
        extension = 'tar.gz'
        mode = 'w:gz'
    else:
        extension = 'tar'
        mode = 'w'
    tar_path = os.path.join(
        settings.PATHS['shards'], '%s.%s'%(shard_name, extension))
    tar = tarfile.open(tar_path, mode)
    
    # build the ldraw scenes and add them to the tar file
    padding = len(str(num_scenes))
    max_instances_per_scene = 0
    max_edges_per_scene = 0
    scene = BrickScene(
        renderable=True,
        track_snaps=True,
        collision_checker=True,
    )
    for i in tqdm.tqdm(range(num_scenes)):
        scene.clear_instances()
        sample_scene(
            scene,
            subassembly_samplers,
            colors,
            min_instances,
            max_instances,
        )
        
        si = str(i).rjust(padding, '0')
        mpd_name = ('%s_%s.mpd')%(shard_name, si)
        
        text = scene.export_ldraw_text(mpd_name)
        text_bytes = text.encode('utf8')
        io = BytesIO(text_bytes)
        info = tarfile.TarInfo(name=mpd_name)
        info.size = len(text_bytes)
        tar.addfile(tarinfo=info, fileobj=io)
        
        max_instances_per_scene = max(
            max_instances_per_scene, len(scene.instances))
        scene_connections = scene.get_all_snap_connections()
        scene_edges = sum(scene_connections.values(), [])
        max_edges_per_scene = max(max_edges_per_scene, len(scene_edges))
        
    tar.close()
    
    # update the max instances/edges per scene
    for ie, m in (
        ('instances', max_instances_per_scene), ('edges', max_edges_per_scene)
    ):
        max_name = 'max_%s_per_scene'%ie
        if max_name in dataset_info:
            if m > dataset_info[max_name]:
                print('Updating "%s" from %i to %i'%(
                    max_name, dataset_info[max_name], m))
                dataset_info[max_name] = m
        else:
            print('Setting "%s" to %i'%(max_name, m))
            dataset_info[max_name] = m
    
    # add the splits dictionary to the dataset_info if it does not exist
    if 'splits' not in dataset_info:
        dataset_info['splits'] = {}
    
    # add the split for this shard, overwrite if it already exists
    size_split_name = '%s_%s'%(size_name, split_name)
    if size_split_name in dataset_info['splits']:
        print('Overwriting split "%s"'%size_split_name)
    else:
        print('Adding split "%s"'%size_split_name)
    dataset_info['splits'][size_split_name] = {
        'shards': [shard_name],
    }
    
    # add this shard to the size_all, split_all and all splits
    for all_name in '%s_all'%size_name, '%s_all'%split_name, 'all':
        if all_name not in dataset_info['splits']:
            print('Adding split "%s"'%all_name)
            dataset_info['splits'][all_name] = {'shards':[]}
        if shard_name not in dataset_info['splits'][all_name]['shards']:
            print('Adding shard "%s" to Split "%s"'%(
                shard_name, all_name))
            dataset_info['splits'][all_name]['shards'].append(shard_name)
    
    dataset_path = os.path.join(
        settings.PATHS['datasets'], '%s.json'%dataset_name)
    with open(dataset_path, 'w') as f:
        dataset_info = json.dump(dataset_info, f, indent=2)

def sample_scene(
        scene,
        subassembly_samplers,
        colors,
        min_instances,
        max_instances,
        retries=20,
        debug=False,
        timeout=None):
    
    t_start = time.time()
    
    num_bricks = random.integers(min_instances, max_instances, endpoint=True)
    
    scene.load_colors(colors)
    
    for i in range(num_bricks):
        if timeout is not None:
            if time.time() - t_start > timeout:
                print('TIMEOUT')
                return scene
        
        if len(scene.instances):
            
            unoccupied_snaps = scene.get_unoccupied_snaps()
            
            if not len(unoccupied_snaps):
                print('no unoccupied snaps!')
            
            while True:
                if timeout is not None:
                    if time.time() - t_start > timeout:
                        print('TIMEOUT')
                        return scene
                
                # import the sub-assembly
                sub_assembly_sampler = random.choice(subassembly_samplers)
                sub_assembly = sub_assembly_sampler.sample()
                sub_assembly_snaps = []
                new_instances = []
                for brick_shape, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_shape(brick_shape)
                    new_instance = scene.add_instance(
                            brick_shape, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.snaps
                    sub_assembly_snaps.extend(new_snaps)
                
                # TMP to handle bad sub-assemblies
                if len(sub_assembly_snaps) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                # try to find a valid connection
                # get all pairs
                pairs = get_all_snap_pairs(sub_assembly_snaps, unoccupied_snaps)
                if len(pairs) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                # try to find a pair that is not in collision
                for j in range(retries):
                    pick, place = random.choice(pairs)
                    transforms = scene.all_pick_and_place_transforms(
                        pick, place, check_collision=True)
                    
                    if len(transforms):
                        transform = random.choice(transforms)
                        scene.move_instance(pick.brick_instance, transform)
                        
                        if debug:
                            print(i,j)
                            scene.export_ldraw('%i_%s.ldr'%(i,j))
                        
                        break
                
                # if we tried many times and didn't find a good connection,
                # loop back and try a new sub-assembly
                else:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                # if we did find a good connection, break out and move on
                # to the next piece
                break
                    
        else:
            while True:
                sub_assembly_sampler = random.choice(subassembly_samplers)
                sub_assembly = sub_assembly_sampler.sample()
                new_instances = []
                sub_assembly_snaps = []
                for brick_shape, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_shape(brick_shape)
                    new_instance = scene.add_instance(
                            brick_shape, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.snaps
                    sub_assembly_snaps.extend(new_snaps)
                
                if len(sub_assembly_snaps):
                    break
                else:
                    for instance in new_instances:
                        scene.remove_instance(instance)

def get_all_snap_pairs(instance_snaps_a, instance_snaps_b):
    snap_pairs = [
        (snap_a, snap_b)
        for (snap_a, snap_b)
        in product(instance_snaps_a, instance_snaps_b)
        if (snap_a != snap_b and snap_a.compatible(snap_b))
    ]

    return snap_pairs
