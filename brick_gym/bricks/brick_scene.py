import time
import math

import numpy

#import renderpy.buffer_manager_egl as buffer_manager_egl
#import renderpy.buffer_manager_glut as buffer_manager_glut
try:
    from renderpy.core import Renderpy
    import renderpy.camera as camera
    import renderpy.assets as drpy_assets
    renderpy_available = True
except ImportError:
    renderpy_available = False

import brick_gym.config as config
from brick_gym.dataset.paths import resolve_subdocument
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
    
    def __init__(self,
            default_image_light = None,
            opengl_mode='none',
            renderable=False,
            track_snaps=False):
        
        self.default_image_light = default_image_light
        
        # renderable
        self.opengl_mode = opengl_mode
        self.renderable = False
        self.renderer = None
        if renderable:
            self.make_renderable()
        
        # track_snaps
        self.track_snaps = False
        self.snap_tracker = None
        if track_snaps:
            self.make_track_snaps()
        
        # bricks
        self.brick_library = BrickLibrary()
        self.instances = BrickInstanceTable(self.brick_library)
        self.color_library = BrickColorLibrary()
        
    def make_renderable(self):
        if not renderpy_available:
            raise Exception('Renderpy not available')
        if not self.renderable:
            if self.opengl_mode == 'egl':
                import renderpy.buffer_manager_egl as buffer_manager_egl
                manager = buffer_manager_egl.initialize_shared_buffer_manager()
            #import renderpy.glut as glut
            #glut.initialize_glut()
            #self.window = glut.GlutWindowWrapper()
            self.renderable = True
            config_paths = '%s:%s'%(
                    config.paths['renderpy_assets_cfg'],
                    drpy_assets.default_assets_path)
            self.renderer = Renderpy(config_paths,
                    default_camera_projection = camera.projection_matrix(
                        math.radians(60.),
                        1.0,
                        1, 5000))
            self.make_snap_materials()
    
    def make_snap_materials(self):
        self.renderer.load_material(
                'M_snap',
                flat_color = (0, 0, 1),
                ka = 1.0,
                kd = 0.0,
                ks = 0.0,
                shine = 1.0,
                image_light_kd = 1.0,
                image_light_ks = 0.0,
                image_light_blur_reflection = 4.0)
        self.renderer.load_material(
                'F_snap',
                flat_color = (1, 0, 0),
                ka = 1.0,
                kd = 0.0,
                ks = 0.0,
                shine = 1.0,
                image_light_kd = 1.0,
                image_light_ks = 0.0,
                image_light_blur_reflection = 4.0)
        if self.default_image_light is not None:
            self.load_default_image_light()
        
    def make_track_snaps(self):
        if not self.track_snaps:
            self.snap_tracker = GridBucket(cell_size=8)
            self.track_snaps = True
    
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
            self.make_snap_materials()
            self.clear_image_lights()
            if self.default_image_light is not None:
                self.load_default_image_light()
    
    def add_brick_type(self, brick_type):
        new_type = self.brick_library.add_type(brick_type)
        if self.renderable:
            if not self.renderer.mesh_exists(new_type.mesh_name):
                self.renderer.load_mesh(
                        new_type.mesh_name,
                        **new_type.renderpy_mesh_args())
    
    def add_colors(self, colors):
        new_colors = self.color_library.load_colors(colors)
        if self.renderable:
            for new_color in new_colors:
                self.renderer.load_material(
                        brick_color.material_name,
                        **brick_color.renderpy_material_args())
    
    def add_instance(self, brick_type, brick_color, transform):
        brick_instance = self.instances.add_instance(
                brick_type, brick_color, transform)
        if self.renderable:
            self.renderer.add_instance(
                    brick_instance.instance_name,
                    **brick_instance.renderpy_instance_args())
        if self.track_snaps:
            self.update_instance_snaps(brick_instance)
    
    def import_ldraw(self, path):
        #t0 = time.time()
        path, subdocument = resolve_subdocument(path)
        
        document = LDrawDocument.parse_document(path)
        if subdocument is not None:
            document = document.reference_table['ldraw'][subdocument]
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
        if str(brick_instance) in self.instances:
            self.renderer.hide_instance(str(brick_instance))
    
    def show_instance(self, brick_instance):
        self.renderer.show_instance(str(brick_instance))
    
    def instance_hidden(self, brick_instance):
        return self.renderer.instance_hidden(str(brick_instance))
    
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
            
            if self.renderable:
                subtype_id = snap.subtype_id
                if not self.renderer.mesh_exists(subtype_id):
                    snap_mesh = snap.get_snap_mesh()
                    self.renderer.load_mesh(
                            subtype_id,
                            mesh_data = snap_mesh,
                            color_mode = 'flat_color')
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
    
    def get_instance_snap_connections(self, instance, radius=1):
        instance = self.instances[instance]
        other_snaps = []
        for i, snap in enumerate(instance.get_snaps()):
            snap_id = (str(instance), i)
            snap_position = numpy.dot(snap.transform, [0,0,0,1])[:3]
            snaps_in_radius = self.snap_tracker.lookup(snap_position, radius)
            other_snaps.extend(
                    [s for s in snaps_in_radius if s[0] != str(instance)])
        return other_snaps
    
    def get_all_snap_connections(self):
        snap_connections = {}
        for instance in self.instances:
            connections = self.get_instance_snap_connections(instance)
            snap_connections[str(instance)] = connections
        
        return snap_connections
    
    def get_all_edges(self, unidirectional=False):
        snap_connections = self.get_all_snap_connections()
        all_edges = set()
        for instance_a_name in snap_connections:
            instance_a_id = int(instance_a_name)
            for instance_b_name, snap_id in snap_connections[instance_a_name]:
                instance_b_id = int(instance_b_name)
                if instance_a_id < instance_b_id or not unidirectional:
                    all_edges.add((instance_a_id, instance_b_id))
        all_edges = numpy.array(list(all_edges)).T
        return all_edges
    
    '''
    def get_edge_dict(self):
        graph_edges = self.get_all_snap_connections()
        for instance_a in graph_edges:
            for instance_b, snap_id in graph_edges[instance_a]:
                if int(instance_a) < int(instance_b):
                    class_a = 
                    class_b = 
    '''
    
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
    
    def set_instance_color(self, instance, new_color):
        instance = self.instances[instance]
        instance.color = new_color
        self.renderer.set_instance_material(
                instance.instance_name,
                new_color)
    
    #===========================================================================
    # materials
    #===========================================================================
    def load_colors(self, colors):
        new_colors = self.color_library.load_colors(colors)
        for new_color in new_colors:
            self.renderer.load_material(
                    new_color.material_name,
                    **new_color.renderpy_material_args())
        return new_colors
    
    #===========================================================================
    # rendering
    #===========================================================================
    def load_default_image_light(self):
        self.renderer.load_image_light(
                'default',
                texture_directory = self.default_image_light)
        self.renderer.set_active_image_light('default')
    
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
            
            'get_instance_center_bbox',
            
            'color_render',
            'mask_render'))
    def __getattr__(self, attr):
        if attr in self.pass_through_functions and self.renderable:
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
