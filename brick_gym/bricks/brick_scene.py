import time
import math

import numpy

#import renderpy.buffer_manager_egl as buffer_manager_egl
#import renderpy.buffer_manager_glut as buffer_manager_glut
from renderpy.core import Renderpy
import renderpy.camera as camera
import renderpy.assets as drpy_assets

import brick_gym.config as config
from brick_gym.ldraw.documents import LDrawDocument
#from brick_gym.bricks.snap_manager import SnapManager
from brick_gym.bricks.brick_type import BrickLibrary
from brick_gym.bricks.brick_instance import BrickInstanceTable
from brick_gym.bricks.brick_color import BrickColorLibrary
from brick_gym.bricks.snap import *
from brick_gym.geometry.grid_bucket import GridBucket

class BrickScene:
    
    upright = numpy.array([
            [-1, 0, 0, 0],
            [ 0,-1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]])
    
    def __init__(self, renderable=False, track_snaps=False):
        # modes
        self.renderable = renderable
        self.track_snaps = track_snaps
        
        # bricks
        self.brick_library = BrickLibrary()
        self.instances = BrickInstanceTable(self.brick_library)
        self.color_library = BrickColorLibrary()
        
        # renderpy
        if self.renderable:
            config_paths = '%s:%s'%(
                    config.paths['renderpy_assets_cfg'],
                    drpy_assets.default_assets_path)
            self.renderer = Renderpy(config_paths,
                    default_camera_projection = camera.projection_matrix(
                        math.radians(60.),
                        1.0,
                        1, 5000))
        else:
            self.renderer = None
        
        # snap
        if self.track_snaps:
            self.snap_tracker = GridBucket(cell_size=8)
            if self.renderable:
                self.renderer.load_material(
                        'M_snap',
                        color = (0, 0, 255),
                        ka = 1.0,
                        kd = 0.0,
                        ks = 0.0,
                        shine = 1.0,
                        image_light_kd = 1.0,
                        image_light_ks = 0.0,
                        image_light_blur_reflection = 4.0)
                self.renderer.load_material(
                        'F_snap',
                        color = (255, 0, 0),
                        ka = 1.0,
                        kd = 0.0,
                        ks = 0.0,
                        shine = 1.0,
                        image_light_kd = 1.0,
                        image_light_ks = 0.0,
                        image_light_blur_reflection = 4.0)
        else:
            self.snap_tracker = None
    
    #===========================================================================
    # scene manipulation
    #===========================================================================
    def clear_instances(self):
        self.instances.clear()
        if self.snap_tracker is not None:
            self.snap_tracker.clear()
        if self.renderer is not None:
            self.renderer.clear_instances()
    
    def clear_assets(self):
        self.brick_library.clear()
        self.color_library.clear()
        if self.renderer is not None:
            self.renderer.clear_meshes()
            self.renderer.clear_materials()
            self.clear_image_lights()
    
    def import_ldraw(self, path):
        #t0 = time.time()
        document = LDrawDocument.parse_document(path)
        new_types = self.brick_library.import_document(document)
        new_instances = self.instances.import_document(
                document, transform=self.upright)
        new_colors = self.color_library.load_from_instances(new_instances)
        #print('ldraw: %f'%(time.time() - t0))
        
        if self.renderable:
            # load meshes
            #t0 = time.time()
            for brick_type in new_types:
                if not self.renderer.mesh_exists(brick_type.mesh_name):
                    self.renderer.load_mesh(
                            brick_type.mesh_name,
                            **brick_type.renderpy_mesh_args())
            #print('meshes: %f'%(time.time() - t0))
            
            # load materials
            #t0 = time.time()
            for brick_color in new_colors:
                self.renderer.load_material(
                        brick_color.material_name,
                        **brick_color.renderpy_material_args())
            #print('materials: %f'%(time.time() - t0))
            
            # add instances
            #t0 = time.time()
            for brick_instance in new_instances:
                self.renderer.add_instance(
                        brick_instance.instance_name,
                        **brick_instance.renderpy_instance_args())
            #print('instances: %f'%(time.time() - t0))
        if self.track_snaps:
            for brick_instance in new_instances:
                self.update_instance_snaps(brick_instance)
            
    #===========================================================================
    # instance manipulation
    #===========================================================================
    def get_instance(self, instance):
        return self.instances[instance]
    
    def hide_instance(self, brick_instance):
        self.renderer.hide_instance(str(brick_instance))
    
    def show_instance(self, brick_instance):
        self.renderer.show_instance(str(brick_instance))
    
    def show_all_instances(self):
        for instance_id, instance in self.instances.items():
            self.show_instance(instance_id)
    
    def renderable_snap(self, snap):
        return isinstance(snap, SnapCylinder)
    
    def update_instance_snaps(self, instance):
        for i, snap in enumerate(instance.get_snaps()):
            if not self.renderable_snap(snap):
                continue
            snap_id = (str(instance), i)
            self.snap_tracker.remove(snap_id)
            snap_position = numpy.dot(snap.transform, [0,0,0,1])[:3]
            self.snap_tracker.insert(snap_id, snap_position)
            
            subtype_id = snap.subtype_id
            if not self.renderer.mesh_exists(subtype_id):
                snap_mesh = snap.get_snap_mesh()
                self.renderer.load_mesh(
                        subtype_id,
                        mesh_data = snap_mesh)
            snap_name = '%s_%i'%(str(instance), i)
            self.renderer.add_instance(
                    snap_name,
                    mesh_name = subtype_id,
                    material_name = '%s_snap'%snap.gender,
                    transform = snap.transform,
                    mask_color = (0,0,0),
                    hidden = True)
    
    def move_instance(self, instance, transform):
        instance = self.instances[instance]
        instance.transform = transform
        if self.track_snaps:
            self.update_instance_snaps()
    
    def get_snap_connections(self, instance, radius=1):
        instance = self.instances[instance]
        other_snaps = []
        for i, snap in enumerate(instance.get_snaps()):
            snap_id = (str(instance), i)
            snap_position = numpy.dot(snap.transform, [0,0,0,1])[:3]
            snaps_in_radius = self.snap_tracker.lookup(snap_position, radius)
            other_snaps.extend(
                    [s for s in snaps_in_radius if s[0] != str(instance)])
        return other_snaps
    
    def get_snap_graph(self):
        graph_edges = {}
        for instance in self.instances:
            connections = self.get_snap_connections(instance)
            graph_edges[str(instance)] = connections
        
        return graph_edges
    
    def show_all_snaps(self):
        for instance in self.instances.values():
            for i, snap in enumerate(instance.get_snaps()):
                if self.renderable_snap(snap):
                    snap_name = '%s_%i'%(str(instance), i)
                    self.renderer.show_instance(snap_name)
    
    def hide_all_snaps(self):
        for instance in self.instances.values():
            for i, snap in enumerate(instance.get_snaps()):
                if self.renderable_snap(snap):
                    snap_name = '%s_%i'%(str(instance), i)
                    self.renderer.hide_instance(snap_name)
    
    #===========================================================================
    # pass-through to renderer
    #===========================================================================
    pass_through_functions = set((
            'set_ambient_color',
            'set_background_color',
            'set_active_image_light',
            
            'reset_camera',
            'set_projection',
            'get_projection',
            'set_camera_pose',
            'get_camera_pose',
            'camera_frame_scene',
            
            'load_image_light',
            'remove_image_light',
            'clear_image_lights',
            'list_image_lights',
            'image_light_exists',
            'get_image_light',
            
            'add_point_light',
            'remove_point_light',
            'clear_point_lights',
            
            'add_direction_light',
            'remove_direction_light',
            'clear_direction_lights',
            
            'color_render',
            'mask_render'))
    def __getattr__(self, attr):
        if attr in self.pass_through_functions:
            return getattr(self.renderer, attr)
        else:
            raise AttributeError('Unknown attribute: %s'%attr)
    
    #===========================================================================
    # individual brick manipulations
    #===========================================================================
    def add_brick(self, brick_name, color, transform):
        raise NotImplementedError
    
    def remove_brick(self, brick_instance):
        raise NotImplementedError
    
    def hide_brick(self, brick_instance):
        raise NotImplementedError
