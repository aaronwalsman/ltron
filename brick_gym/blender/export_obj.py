import os
import argparse

import bpy

import io_scene_importldraw.loadldraw.loadldraw as loadldraw

import brick_gym.config as config
from brick_gym.ldraw.documents import LDrawDocument

# to fix a light/lamp naming issue
loadldraw.globalLightBricks = {}

part_directory = os.path.join(config.paths['ldraw'], 'parts')
obj_directory = os.path.join(config.paths['renderpy'], 'meshes')

if not os.path.exists(obj_directory):
    os.makedirs(obj_directory)

def clear_scene(scene_name):
    
    # if this scene name already exists somewhere, get rid of it
    for scene in bpy.data.scenes:
        if scene.name == scene_name:
            clear_scene = bpy.data.scenes.new('CLEAR')
            for scene in bpy.data.scenes:
                if scene.name != clear_scene.name:
                    bpy.data.scenes.remove(scene)
    
    # create the new scene we want
    scene = bpy.data.scenes.new(scene_name)
    scene.world = bpy.data.worlds[0]

    # get rid of everything else
    for scene in bpy.data.scenes:
        if scene.name != scene_name:
            bpy.data.scenes.remove(scene)
    
    # HAHAHAHA, this just stops script execution.
    # OOPS, it also forgets all your plugins!
    #bpy.ops.wm.read_factory_settings(use_empty=True)

def export_brick(brick,
        overwrite=False,
        import_scale = 1.0,
        primitive_resolution = 'Standard',
        smooth_parts = True,
        curved_walls = True,
        use_logo_studs = False,
        bevel_edges = True,
        bevel_width = 0.5,
        use_normals = True,
        axis_forward = 'Z',
        axis_up = '-Y'):
    brick_name, brick_ext = os.path.splitext(brick)
    export_brick_path = os.path.join(obj_directory, brick_name + '.obj')
    if overwrite or not os.path.isfile(export_brick_path):
    
        # new blender scene
        clear_scene(brick_name)

        # load brick
        import_brick_path = os.path.join(part_directory, brick)
        bpy.ops.import_scene.importldraw(
                filepath = import_brick_path,
                ldrawPath = config.paths['ldraw'],
                importScale = import_scale,
                resPrims = primitive_resolution,
                smoothParts = smooth_parts,
                curvedWalls = curved_walls,
                importCameras = False,
                useLogoStuds = use_logo_studs,
                positionOnGround = False,
                bevelEdges = bevel_edges,
                bevelWidth = bevel_width,
                addEnvironment = False)
        
        # export obj
        bpy.ops.export_scene.obj(
                filepath = export_brick_path,
                check_existing = False,
                use_normals = use_normals,
                axis_forward = axis_forward,
                axis_up = axis_up)

def export_scene_bricks(scene_path, **kwargs):
    scene_document = LDrawDocument.parse_document(scene_path)
    parts = scene_document.get_all_parts()
    bricks = [part[0] for part in parts]
    print(bricks)
    for brick in bricks:
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

def export_all(**kwargs):
    bricks = sorted(os.listdir(part_directory))
    for brick in bricks:
        import_brick_path = os.path.join(part_directory, brick)
        if os.path.isdir(import_brick_path):
            continue
        export_brick(brick, **kwargs)
