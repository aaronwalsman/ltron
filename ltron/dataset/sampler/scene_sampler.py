import math
import time
from itertools import product
import tarfile
from StringIO import StringIO

import numpy
random = numpy.random.default_rng(1234567890)

from pyquaternion import Quaternion

from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.snap import SnapCylinder
from ltron.geometry.collision_sampler import get_all_transformed_snap_pairs
from ltron.geometry.collision import check_snap_collision

def sample_dataset(
    name,
    subassembly_samplers,
    colors,
    min_max_bricks_per_scene,
    train_scenes,
    test_scenes,
):
    dataset_metadata = {
        'splits':{}
    }
    
    for min_bricks, max_bricks in min_max_bricks_per_scene:
        for split, num_scenes in ('train', train_scenes), ('test', test_scenes):
            sample_collection(
                '%s_%i_%i_%s'%(name, min_bricks, max_bricks, split),
                subassembly_samplers,
                colors,
                min_bricks,
                max_bricks,
                num_scenes,
            )

def sample_collection(
    name,
    subassembly_samplers,
    colors,
    min_bricks,
    max_bricks,
    num_scenes,
    compress=False
):
    print('Building Collection: %s'%name)
    if compress:
        extension = '.tar.gz'
        mode = 'w:gz'
    else:
        extension = '.tar'
        mode = 'w'
    tar_path = os.path.join(settings.paths['collections'], name + extension)
    tar = tarfile.open(tar_path, mode)
    
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
            min_bricks,
            max_bricks,
        )
        
        text = scene.export_ldraw_text()
        io = StringIO()
        io.write(text)
        io.seek(0)
        info = tarfile.TarInfo(name=file_name)
        info.size = len(io.buf)
        tar.addfile(tarinfo=info, fileobj=io)
        
    tar.close()

def sample_scene(
        scene,
        subassembly_samplers,
        colors,
        min_bricks,
        max_bricks,
        retries=20,
        debug=False,
        timeout=None):
    
    t_start = time.time()
    
    num_bricks = random.integers(min_bricks, max_bricks, endpoint=True)
    
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
