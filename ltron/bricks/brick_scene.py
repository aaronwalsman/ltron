import time
import math
import os

import numpy

from pyquaternion import Quaternion

import splendor.masks as masks

from ltron.dataset.paths import resolve_subdocument
from ltron.ldraw.documents import LDrawDocument
from ltron.bricks.brick_type import BrickLibrary
from ltron.bricks.brick_instance import BrickInstanceTable
from ltron.bricks.brick_color import BrickColorLibrary
from ltron.bricks.snap import SnapCylinder
from ltron.render.environment import RenderEnvironment
from ltron.geometry.grid_bucket import GridBucket

class BrickScene:
    
    upright = numpy.array([
            [-1, 0, 0, 0],
            [ 0,-1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]])
    
    # initialization and high level settings ===================================
    
    def __init__(self,
            #default_image_light = None,
            renderable=False,
            render_args=None,
            track_snaps=False):
        
        #self.default_image_light = default_image_light
        
        # renderable
        self.renderable = False
        self.render_environment = None
        if renderable:
            if render_args is None:
                render_args = {}
            self.make_renderable(**render_args)
        
        # track_snaps
        self.track_snaps = False
        self.snap_tracker = None
        if track_snaps:
            self.make_track_snaps()
        
        # bricks
        self.brick_library = BrickLibrary()
        self.color_library = BrickColorLibrary()
        self.instances = BrickInstanceTable(
                self.brick_library,
                self.color_library,
        )
        
    def make_renderable(self, **render_args):
        if not self.renderable:
            self.render_environment = RenderEnvironment(**render_args)
            self.renderable = True
    
    def make_track_snaps(self):
        if not self.track_snaps:
            self.snap_tracker = GridBucket(cell_size=8)
            self.track_snaps = True
    
    # scene manipulation =======================================================
    
    # ldraw i/o ----------------------------------------------------------------
    
    def import_ldraw(self, path):
        # convert the path to a path and subdocument
        path, subdocument = resolve_subdocument(path)
        
        # read the document and pull out the subdocument
        document = LDrawDocument.parse_document(path)
        if subdocument is not None:
            document = document.reference_table['ldraw'][subdocument]
        
        # load brick types, instances and colors
        new_types = self.brick_library.import_document(document)
        new_colors = self.color_library.import_document(document)
        new_instances = self.instances.import_document(
                document, transform=self.upright)
        
        if self.renderable:
            # adding instances will automatically load the appropriate assets
            for new_instance in new_instances:
                self.render_environment.add_instance(new_instance)
        
        if self.track_snaps:
            for brick_instance in new_instances:
                self.update_instance_snaps(brick_instance)
    
    def export_ldraw(self, path):
        directory, file_name = os.path.split(path)
        lines = [
                '0 FILE %s'%file_name,
                '0 Main',
                '0 Name: %s'%file_name,
                '0 Author: LTRON',
        ]
        for instance in self.instances.values():
            color = instance.color
            t = self.upright @ instance.transform
            str_transform = (' '.join(['%f']*12))%(
                    t[0,3], t[1,3], t[2,3],
                    t[0,0], t[0,1], t[0,2],
                    t[1,0], t[1,1], t[1,2],
                    t[2,0], t[2,1], t[2,2])
                    
            brick_type_name = str(instance.brick_type)
            line = '1 %s %s %s'%(color, str_transform, brick_type_name)
            lines.append(line)
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    
    # assets -------------------------------------------------------------------
    
    def clear_assets(self):
        self.clear_instances()
        self.brick_library.clear()
        self.color_library.clear()
        if self.renderable:
            self.render_environment.clear_meshes()
            self.render_environment.clear_materials()
            self.render_environment.clear_image_lights()
    
    # brick types --------------------------------------------------------------
    
    def add_brick_type(self, brick_type):
        new_type = self.brick_library.add_type(brick_type)
        if self.renderable:
            self.render_environment.load_brick_mesh(new_type)
        return new_type
    
    # instances ----------------------------------------------------------------
    
    def add_instance(self, brick_type, brick_color, transform):
        self.brick_library.add_type(brick_type)
        self.color_library.load_colors([brick_color])
        brick_instance = self.instances.add_instance(
                brick_type, brick_color, transform)
        if self.renderable:
            self.render_environment.add_instance(brick_instance)
        if self.track_snaps:
            self.update_instance_snaps(brick_instance)
        
        return brick_instance
    
    def move_instance(self, instance, transform):
        instance = self.instances[instance]
        instance.transform = transform
        if self.renderable:
            self.render_environment.update_instance(instance)
        if self.track_snaps:
            self.update_instance_snaps(instance)
    
    def hide_instance(self, instance):
        self.renderer.hide_instance(str(instance))
    
    def show_instance(self, instance):
        self.renderer.show_instance(str(instance))
    
    def clear_instances(self):
        self.instances.clear()
        if self.snap_tracker is not None:
            self.snap_tracker.clear()
        if self.renderable:
            self.render_environment.clear_instances()
    
    def is_instance_removable(
            self, instance, direction_space='scene', radius=1):
        assert self.track_snaps
        
        instance = self.instances[instance]
        other_snaps = self.get_instance_snap_connections(instance, radius)
        
        instance_snaps = instance.get_snaps()
        snap_polarities = []
        snap_axes = []
        for other_instance_id, other_snap_id, this_snap_id in other_snaps:
            if self.renderable and self.instance_hidden(other_instance_id):
                continue
            
            snap = instance_snaps[this_snap_id]
            snap_polarities.append(snap.polarity)
            snap_axis = numpy.dot(
                    snap.transform, numpy.array([[0],[-1],[0],[0]]))[:,0]
            if snap.polarity == '+':
                snap_axis = -snap_axis
            if direction_space == 'camera':
                snap_axis = numpy.dot(self.get_view_matrix(), snap_axis)
            snap_axis = snap_axis / numpy.linalg.norm(snap_axis)
            snap_axes.append(snap_axis[:3])
        
        if len(snap_axes) == 0:
            return True, None
        
        if len(snap_axes) == 1:
            return True, snap_axes[0]
        
        if not all(snap_polarities[0] == g for g in snap_polarities[1:]):
            return False, None
        
        if not all(numpy.dot(snap_axes[0], a) > 0.95 for a in snap_axes[1:]):
            return False, None
        
        return True, snap_axes[0]
    
    def set_instance_color(self, instance, new_color):
        self.load_colors([new_color])
        new_color = self.color_library[new_color]
        
        instance = self.instances[instance]
        instance.color = new_color
        if self.renderable:
            self.render_environment.update_instance(instance)
    
    def set_instance_transform(self, instance, transform):
        instance = self.instances[instance]
        instance.transform = transform
        if self.renderable:
            self.render_environment.update_instance(instance)
    
    def remove_instance(self, instance):
        instance = self.instances[instance]
        if self.renderable:
            self.render_environment.remove_instance(instance)
        if self.track_snaps:
            for i, snap in enumerate(instance.get_snaps()):
                snap_id = (str(instance), i)
                self.snap_tracker.remove(snap_id)
            
        del(self.instances[instance])
    
    # instance snaps -----------------------------------------------------------
    
    def update_instance_snaps(self, instance):
        assert self.track_snaps
        for i, snap in enumerate(instance.get_snaps()):
            snap_id = (str(instance), i)
            self.snap_tracker.remove(snap_id)
            snap_position = numpy.dot(snap.transform, [0,0,0,1])[:3]
            self.snap_tracker.insert(snap_id, snap_position)
    
    def get_snaps(
        self,
        instances=None,
        polarity=None,
        style=None,
        visible=True
    ):
        if instances is None:
            instances = self.instances
        matching_snaps = []
        for instance in instances:
            if self.renderable and self.instance_hidden(str(instance)):
                continue
            instance = self.instances[instance]
            for i, snap in enumerate(instance.get_snaps()):
                if polarity is not None and snap.polarity != polarity:
                    continue
                if style is not None and snap.style not in style:
                    continue
                #if renderable and not self.renderable_snap(snap):
                #    continue
                matching_snaps.append((str(instance), i))
        
        return matching_snaps
    
    def snap_names_to_snaps(self, snap_names):
        return [self.instances[i].get_snap(s) for i,s in snap_names]
    
    def get_instance_snap_connections(self, instance, radius=1):
        assert self.track_snaps
        
        instance = self.instances[instance]
        other_snaps = []
        for i, snap in enumerate(instance.get_snaps()):
            snap_position = numpy.dot(snap.transform, [0,0,0,1])[:3]
            snaps_in_radius = self.snap_tracker.lookup(snap_position, radius)
            #for j,s in snaps_in_radius:
            #    if j != str(instance)
            
            other_snaps.extend(
                [(j,s,i) for j,s in snaps_in_radius
                    if j != str(instance)
                    and snap.connected(self.instances[j].get_snap(s))])
            
            '''
            other_snaps.extend(
                    [s + (i,) for s in snaps_in_radius
                        if s[0] != str(instance)])
            '''
        return other_snaps
    
    def get_all_snap_connections(self):
        assert self.track_snaps
        
        snap_connections = {}
        for instance in self.instances:
            connections = self.get_instance_snap_connections(instance)
            snap_connections[str(instance)] = connections
        
        return snap_connections
    
    def get_all_edges(self, unidirectional=False):
        assert self.track_snaps
        snap_connections = self.get_all_snap_connections()
        all_edges = set()
        for instance_a_name in snap_connections:
            instance_a_id = int(instance_a_name)
            connections = snap_connections[instance_a_name]
            for instance_b_name, snap_id_b, snap_id_a in connections:
                instance_b_id = int(instance_b_name)
                if instance_a_id < instance_b_id or not unidirectional:
                    all_edges.add(
                        (instance_a_id, instance_b_id, snap_id_a, snap_id_b))
        num_edges = len(all_edges)
        all_edges = numpy.array(list(all_edges)).T.reshape(4, num_edges)
        return all_edges
    
    def get_all_snaps(self):
        assert self.track_snaps
        
        all_snaps = set()
        for instance_id, instance in self.instances.items():
            brick_type = instance.brick_type
            all_snaps |= set(
                (instance_id, i)
                for i in range(len(brick_type.snaps)))
        
        return all_snaps
    
    def get_unoccupied_snaps(self):
        assert self.track_snaps
        
        '''
        # get all snaps
        all_scene_snaps = set()
        for instance_id, instance in self.instances.items():
            brick_type = instance.brick_type
            all_scene_snaps |= set(
                    (instance_id, i)
                    for i in range(len(brick_type.snaps)))
        '''
        all_snaps = self.get_all_snaps()
        
        # build a list of occupied snaps
        all_snap_connections = self.get_all_snap_connections()
        occupied_snaps = set()
        for a_id, connections in all_snap_connections.items():
            for b_id, b_snap, a_snap in connections:
                occupied_snaps.add((a_id, a_snap))
                occupied_snaps.add((b_id, b_snap))
        # build a list of unoccupied snaps
        unoccupied_snaps = all_snaps - occupied_snaps
        unoccupied_snaps = [
                self.instances[instance_id].get_snap(snap_id)
                for instance_id, snap_id in unoccupied_snaps]
        '''
        # filter for studs
        unoccupied_snaps = [
                snap for snap in unoccupied_snaps
                if (isinstance(snap, SnapCylinder) and
                        snap.contains_stud_radius())]
        '''
        
        return unoccupied_snaps
    
    def pick_and_place_snap_transform(self, pick, place):
        pick_instance_id, pick_snap_id = pick
        pick_instance = self.instances[pick_instance_id]
        pick_snap = pick_instance.get_snap(pick_snap_id)
        place_instance_id, place_snap_id = place
        place_instance = self.instances[place_instance_id]
        place_snap = self.instances[place_instance_id].get_snap(place_snap_id)
        
        best_transform = None
        best_pseudo_angle = -float('inf')
        for i in range(4):
            angle = i * math.pi/2.
            rotation = Quaternion(axis=(0,1,0), angle=angle)
            candidate_transform = (
                place_snap.transform @
                rotation.transformation_matrix @
                numpy.linalg.inv(pick_snap.transform) @
                pick_instance.transform
            )
            offset = (
                candidate_transform @
                numpy.linalg.inv(pick_instance.transform)
            )
            pseudo_angle = numpy.trace(offset[:3,:3])
            if pseudo_angle > best_pseudo_angle:
                best_transform = candidate_transform
                best_pseudo_angle = pseudo_angle
        
        return best_transform
    
    def pick_and_place_snap(self, pick, place):
        pick_instance = self.instances[pick[0]]
        transform = self.pick_and_place_snap_transform(pick, place)
        self.move_instance(pick_instance, transform)
    
    # materials ----------------------------------------------------------------
    
    def load_colors(self, colors):
        new_colors = self.color_library.load_colors(colors)
        if self.renderable:
            for new_color in new_colors:
                self.render_environment.load_color_material(new_color)
        return new_colors
    
    # rendering ----------------------------------------------------------------
    '''
    def load_default_image_light(self):
        self.renderer.load_image_light(
                'default',
                diffuse_texture = self.default_image_light + '_dif.png',
                reflect_texture = self.default_image_light + '_ref.png')
                #texture_directory = self.default_image_light)
        self.renderer.set_active_image_light('default')
    '''
    def removable_render(self, *args, **kwargs):
        # needs update
        raise NotImplementedError
        for instance_id, instance in self.instances.items():
            instance_data = self.renderer.scene_description[
                    'instances'][instance.instance_name]
            removable, axis = self.is_instance_removable(instance)
            mask_color = (float(removable),)*3
            instance_data['mask_color'] = mask_color
        
        self.mask_render(*args, **kwargs)
        
        for instance_id, instance in self.instances.items():
            instance_data = self.renderer.scene_description[
                    'instances'][instance.instance_name]
            mask_color = masks.color_index_to_byte(int(instance_id)) / 255.
            instance_data['mask_color'] = mask_color
    
    def __getattr__(self, attr):
        if self.renderable:
            try:
                return getattr(self.render_environment, attr)
            except AttributeError:
                pass
         
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )
