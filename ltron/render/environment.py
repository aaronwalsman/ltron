import math

import splendor.contexts.egl as egl
import splendor.contexts.glut as glut
from splendor.core import SplendorRender
import splendor.camera as camera
import splendor.masks as masks

import ltron.settings as settings

default_projection = camera.projection_matrix(
    math.radians(60.),
    aspect_ratio=1.,
    near_clip=10,
    far_clip=50000,
)

default_asset_paths = 'ltron_assets,default_assets'

class RenderEnvironment:
    
    # initialization ===========================================================
    
    def __init__(self,
        asset_paths=default_asset_paths,
        opengl_mode='egl',
        egl_device=None,
        window_width=128,
        window_height=128,
        window_visible=True,
        window_anti_alias=True,
        window_anti_alias_samples=8,
        load_scene=None,
    ):
        if opengl_mode == 'egl':
            egl.initialize_plugin()
            egl.initialize_device(device=egl_device)
            self.window = None
        
        elif opengl_mode == 'glut':
            glut.initialize()
            self.window = glut.GlutWindowWrapper(
                    'LTRON',
                    width = window_width,
                    height = window_height,
                    anti_alias = window_anti_alias,
                    anti_alias_samples = window_anti_alias_samples)
            if window_visible:
                self.window.show_window()
            else:
                self.window.hide_window()
        elif opengl_mode == 'ignore':
            self.window = None
        else:
            raise Exception(
                    'Unknown opengl_mode: %s (expected "egl" or "glut")')
        
        self.renderer = SplendorRender(
                asset_paths,
                default_camera_projection=default_projection,
        )
        self.load_scene = load_scene
        if self.load_scene is not None:
            self.renderer.load_scene(self.load_scene)
        self.make_snap_materials()
    
    # materials ================================================================
    
    def make_snap_materials(self):
        self.renderer.load_material(
                'snap+',
                flat_color = (0, 0, 1),
                ambient = 1.0,
                metal = 0.0,
                rough = 0.0,
                base_reflect = 0.0)
        self.renderer.load_material(
                'snap-',
                flat_color = (1, 0, 0),
                ambient = 1.0,
                metal = 0.0,
                rough = 0.0,
                base_reflect = 0.0)
    
    def load_color_material(self, color):
        if not self.renderer.material_exists(color.color_name):
            self.renderer.load_material(
                color.color_name,
                **color.splendor_material_args(),
            )
    
    def clear_materials(self):
        self.renderer.clear_materials()
        self.make_snap_materials()
    
    # meshes ===================================================================
    
    def load_brick_mesh(self, brick_shape):
        if not self.renderer.mesh_exists(brick_shape.mesh_name):
            self.renderer.load_mesh(
                brick_shape.mesh_name,
                **brick_shape.splendor_mesh_args(),
            )    
    
    # instances ================================================================
    
    def add_instance(self, brick_instance):
        # load mesh if necessary
        self.load_brick_mesh(brick_instance.brick_shape)
        
        # load the color material if necessary
        self.load_color_material(brick_instance.color)
        
        # add the splendor instance
        self.renderer.add_instance(
            brick_instance.instance_name,
            **brick_instance.splendor_instance_args(),
        )
        
        # add the snap instances
        for i, snap in enumerate(brick_instance.snaps):
            self.add_snap_instance(snap)
    
    def add_snap_instance(self, snap):
        if self.window is not None:
            self.window.set_active()
        # create the mesh if it doesn't exist
        if not self.renderer.mesh_exists(snap.subtype_id):
            self.renderer.load_mesh(
                snap.subtype_id,
                mesh_data=snap.get_snap_mesh(),
                color_mode='flat_color',
            )
        
        # add the splendor instance
        self.renderer.add_instance(
            str(snap),
            mesh_name=snap.subtype_id,
            material_name='snap%s'%snap.polarity,
            transform=snap.transform,
            mask_color=(0,0,0),
            hidden=False,
        )
    
    def remove_instance(self, brick_instance):
        # remove the instance
        self.renderer.remove_instance(str(brick_instance))
        
        # remove all snap instances
        for snap in brick_instance.snaps:
            self.renderer.remove_instance(str(snap))
    
    def update_instance(self, brick_instance):
        self.renderer.set_instance_transform(
            str(brick_instance),
            brick_instance.transform,
        )
        self.renderer.set_instance_material(
            str(brick_instance),
            brick_instance.color.color_name,
        )
        self.renderer.set_instance_mesh(
            str(brick_instance),
            brick_instance.brick_shape.mesh_name,
        )
        for snap in brick_instance.snaps:
            self.renderer.set_instance_transform(str(snap), snap.transform)
    
    def instance_hidden(self, brick_instance):
        return self.renderer.instance_hidden(str(brick_instance))
    
    def get_all_brick_instances(self):
        return [
            instance for instance in self.renderer.list_instances()
            if '_' not in instance
        ]
    
    def get_all_snap_instances(self):
        return [
            instance for instance in self.renderer.list_instances()
            if '_' in instance
        ]
    
    def hide_all_brick_instances(self):
        brick_instances = self.get_all_brick_instances()
        for brick_instance in brick_instances:
            self.renderer.hide_instance(brick_instance)
    
    def show_all_brick_instances(self):
        brick_instances = self.get_all_brick_instances()
        for brick_instance in brick_instances:
            self.renderer.show_instance(brick_instance)
    
    def hide_all_snap_instances(self):
        snap_instances = self.get_all_snap_instances()
        for snap_instance in snap_instances:
            self.renderer.hide_instance(snap_instance)
    
    def show_all_snap_instances(self):
        snap_instances = self.get_all_snap_instances()
        for snap_instance in snap_instances:
            self.renderer.show_instance(snap_instance)
    
    def hide_snap_instance(self, snap):
        self.hide_instance(str(snap))
    
    def show_snap_instance(self, snap):
        self.show_instance(str(snap))
    
    def color_render(self, instances=None, **kwargs):
        if instances is None:
            instances = self.get_all_brick_instances()
        self.renderer.color_render(instances=instances, **kwargs)
    
    def mask_render(self, instances=None, **kwargs):
        if instances is None:
            instances = self.get_all_brick_instances()
        
        background_color = self.renderer.get_background_color()
        self.renderer.set_background_color((0,0,0))
        self.renderer.mask_render(instances=instances, **kwargs)
        self.renderer.set_background_color(background_color)
    
    def snap_render_instance_id(self, snaps=None, **kwargs):
        if snaps is None:
            snaps = self.get_all_snap_instances()
        
        snap_names = [str(snap) for snap in snaps]
        
        background_color = self.renderer.get_background_color()
        self.renderer.set_background_color((0,0,0))
        self.set_snap_masks_to_instance_id(snaps)
        
        self.renderer.mask_render(instances=snap_names, **kwargs)
        self.renderer.set_background_color(background_color)
    
    def snap_render_snap_id(self, snaps=None, **kwargs):
        if snaps is None:
            snaps = self.get_all_snap_instances()
        
        snap_names = [str(snap) for snap in snaps]
        
        background_color = self.renderer.get_background_color()
        self.renderer.set_background_color((0,0,0))
        
        self.set_snap_masks_to_snap_id(snaps)
        self.renderer.mask_render(instances=snap_names, **kwargs)
        self.renderer.set_background_color(background_color)
    
    def set_snap_masks_to_instance_id(self, snaps):
        if snaps is None:
            snaps = self.get_all_snap_instances()
        mask_lookup = {
            str(snap):int(snap.brick_instance) for snap in snaps}
        self.renderer.set_instance_masks_to_instance_indices(mask_lookup)
    
    def set_snap_masks_to_snap_id(self, snaps):
        if snaps is None:
            snaps = self.get_all_snap_instances()
        mask_lookup = {
            str(snap):int(snap.snap_style) for snap in snaps}
        self.renderer.set_instance_masks_to_instance_indices(mask_lookup)
    
    def __getattr__(self, attr):
        try:
            return getattr(self.renderer, attr)
        except AttributeError:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr
                )
            )
