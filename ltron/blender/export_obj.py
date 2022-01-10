import os
import argparse

import bpy

import io_scene_importldraw.loadldraw.loadldraw as loadldraw

import ltron.settings as settings
import ltron.dataset.paths as dataset_paths
#from ltron.ldraw.documents import LDrawDocument
from ltron.bricks.brick_scene import BrickScene
import splendor.assets as assets

# to fix a light/lamp naming issue
loadldraw.globalLightBricks = {}

part_directory = os.path.join(settings.paths['ldraw'], 'parts')
#obj_directory = os.path.join(settings.paths['splendor'], 'meshes')
ltron_assets = assets.AssetLibrary('ltron_assets')
obj_directory = ltron_assets['meshes'].paths[0]

if not os.path.exists(obj_directory):
    os.makedirs(obj_directory)

def clear_scene(scene_name):
    
    # if this scene name already exists somewhere, get rid of it
    for scene in bpy.data.scenes:
        if scene.name == scene_name:
            clear_scene = bpy.data.scenes.new('CLEAR')
            for scene in bpy.data.scenes:
                if scene.name != clear_scene.name:
                    for obj in scene.objects:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    bpy.data.scenes.remove(scene, do_unlink=True)
    
    # create the new scene we want
    scene = bpy.data.scenes.new(scene_name)
    scene.world = bpy.data.worlds[0]

    # get rid of everything else
    for scene in bpy.data.scenes:
        if scene.name != scene_name:
            bpy.data.scenes.remove(scene, do_unlink=True)
    
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    
    # I wonder what else I might have to delete to get a fresh scene?
    
    # HAHAHAHA, this just stops script execution.
    # OOPS, it also forgets all your plugins!
    #bpy.ops.wm.read_factory_settings(use_empty=True)

def clear_scene_nope(name):
    # this doesn't work... halts the python execution
    bpy.ops.wm.open_mainfile(
            filepath='/media/awalsman/data_drive/brick-gym/data/blank.blend')

medium_settings = {
        'primitive_resolution' : 'Standard',
        'smooth_parts' : True,
        'curved_walls' : True,
        'bevel_edges' : False,
        'add_gaps' : True,
        'gap_size' : 0.25,
        'use_normals' : True
}

high_settings = {
        'primitive_resolution' : 'Standard',
        'smooth_parts' : True,
        'curved_walls' : True,
        'bevel_edges' : True,
        'bevel_width' : 0.5,
        'add_gaps' : False,
        'use_normals' : True
}

def export_brick(brick,
        directory=obj_directory,
        overwrite=False,
        import_scale = 1.0,
        primitive_resolution = 'Standard',
        smooth_parts = True,
        curved_walls = True,
        use_logo_studs = False,
        bevel_edges = False,
        bevel_width = 0.5,
        add_gaps = True,
        gap_size = 0.25,
        use_normals = True,
        axis_forward = 'Z',
        axis_up = '-Y'):
    brick_name, brick_ext = os.path.splitext(brick)
    export_brick_path = os.path.join(directory, brick_name + '.obj')
    if overwrite or not os.path.isfile(export_brick_path):
    
        # new blender scene
        clear_scene(brick_name)

        # load brick
        import_brick_path = os.path.join(part_directory, brick)
        bpy.ops.import_scene.importldraw(
                filepath = import_brick_path,
                ldrawPath = settings.paths['ldraw'],
                importScale = import_scale,
                resPrims = primitive_resolution,
                smoothParts = smooth_parts,
                bevelEdges = bevel_edges,
                bevelWidth = bevel_width,
                addGaps = add_gaps,
                gapsSize = gap_size,
                curvedWalls = curved_walls,
                importCameras = False,
                useLogoStuds = use_logo_studs,
                positionOnGround = False,
                addEnvironment = False)
        
        # export obj
        bpy.ops.export_scene.obj(
                filepath = export_brick_path,
                check_existing = False,
                use_normals = use_normals,
                axis_forward = axis_forward,
                axis_up = axis_up)

def export_scene_bricks(scene_path, **kwargs):
    #scene_document = LDrawDocument.parse_document(scene_path)
    brick_scene = BrickScene()
    brick_scene.import_ldraw(scene_path)
    for brick_name, brick_shape in brick_scene.shape_library.items():
        print(brick_name)
        export_brick(brick_name, **kwargs)
    #parts = scene_document.get_all_parts()
    #bricks = [part[0] for part in parts]
    #for brick in bricks:
    #    print(brick)
    #    export_brick(brick, **kwargs)

def export_dataset_bricks(dataset, **kwargs):
    info = dataset_paths.get_dataset_info(dataset)
    bricks = info['shape_ids'].keys()
    for brick in bricks:
        print(brick)
        export_brick(brick, **kwargs)

def export_carbon_star(**kwargs):
    for brick in (
            '6157.dat',
            '3031.dat',
            '6157.dat',
            '3710.dat',
            '2436a.dat',
            '6141.dat',
            '6141.dat',
            '2436a.dat',
            '3795.dat',
            '2436a.dat',
            '6141.dat',
            '6141.dat',
            '6141.dat',
            '6141.dat',
            '2412b.dat',
            '2412b.dat',
            '3069b.dat',
            '3069b.dat',
            '2412b.dat',
            '2412b.dat',
            '4590.dat',
            '4589.dat',
            '4589.dat',
            '3021.dat',
            '50947.dat',
            '50947.dat',
            '2412b.dat',
            '50947.dat',
            '50943.dat',
            '50947.dat',
            '50950.dat',
            '50950.dat',
            '3023.dat',
            '30602.dat',
            '47458.dat',
            '51719.dat',
            '51719.dat',
            '51719.dat',
            '51719.dat',
            '50951.dat',
            '50951.dat',
            '50951.dat',
            '50951.dat'):
        export_brick(brick, **kwargs)

def export_n(n, **kwargs):
    raise Exception('Deprecated, use batch_export instead')
    bricks = sorted(os.listdir(part_directory))[-n:]
    import random
    random.shuffle(bricks)
    for brick in bricks:
        import_brick_path = os.path.join(part_directory, brick)
        if os.path.isdir(import_brick_path):
            continue
        export_brick(brick, **kwargs)

def export_all(**kwargs):
    raise Exception('Deprecated, use batch_export instead')
    bricks = sorted(os.listdir(part_directory))
    for brick in bricks:
        import_brick_path = os.path.join(part_directory, brick)
        if os.path.isdir(import_brick_path):
            continue
        export_brick(brick, **kwargs)
