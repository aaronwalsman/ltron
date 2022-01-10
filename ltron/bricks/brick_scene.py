import time
import math
import os

import numpy

try:
    from pyquaternion import Quaternion
    pyquaternion_available=True
except ImportError:
    pyquaternion_available=False

#import splendor.masks as masks

from ltron.dataset.paths import resolve_subdocument
from ltron.ldraw.documents import LDrawDocument
from ltron.bricks.brick_shape import BrickShapeLibrary
from ltron.bricks.brick_instance import BrickInstanceTable
from ltron.bricks.brick_color import BrickColorLibrary
from ltron.bricks.snap import SnapCylinder
try:
    from ltron.render.environment import RenderEnvironment
    render_available = True
except ImportError:
    render_available = False
from ltron.geometry.grid_bucket import GridBucket
try:
    from ltron.geometry.collision import CollisionChecker
    collision_available = True
except ImportError:
    collision_available = False
from ltron.geometry.utils import unscale_transform
from ltron.exceptions import LtronException

class MissingClassError(LtronException):
    pass

class MissingColorError(LtronException):
    pass

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
            track_snaps=False,
            collision_checker=False,
            collision_checker_args=None,
            shape_library=None):
        
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
        
        # collision_checker
        self.collision_checker = None
        if collision_checker:
            if collision_checker_args is None:
                collision_checker_args = {}
            self.make_collision_checker(**collision_checker_args)
        
        # bricks
        if shape_library is None:
            shape_library = BrickShapeLibrary()
        self.shape_library = shape_library
        self.color_library = BrickColorLibrary()
        self.instances = BrickInstanceTable(
                self.shape_library,
                self.color_library,
        )
    
    def make_renderable(self, **render_args):
        assert render_available
        if not self.renderable:
            self.render_environment = RenderEnvironment(**render_args)
            self.renderable = True
    
    def make_track_snaps(self):
        if not self.track_snaps:
            self.snap_tracker = GridBucket(cell_size=8)
            self.track_snaps = True
    
    def make_collision_checker(self, **collision_checker_args):
        assert collision_available
        if self.collision_checker is None:
            self.collision_checker = CollisionChecker(
                self, **collision_checker_args)
    
    # scene manipulation =======================================================
    
    # ldraw i/o ----------------------------------------------------------------
    
    def import_ldraw(self, path):
        #import time
        #a = time.time()
        # convert the path to a path and subdocument
        path, subdocument = resolve_subdocument(path)
        
        #b = time.time()
        # read the document and pull out the subdocument
        document = LDrawDocument.parse_document(path)
        if subdocument is not None:
            document = document.reference_table['ldraw'][subdocument]
        
        #c = time.time()
        # load brick shapes, instances and colors
        new_shapes = self.shape_library.import_document(document)
        new_colors = self.color_library.import_document(document)
        new_instances = self.instances.import_document(
                document, transform=self.upright)
        
        #d = time.time()
        if self.renderable:
            # adding instances will automatically load the appropriate assets
            for new_instance in new_instances:
                self.render_environment.add_instance(new_instance)
        
        #e = time.time()
        if self.track_snaps:
            for brick_instance in new_instances:
                self.update_instance_snaps(brick_instance)
        
        #f = time.time()
        #print('ab', b-a)
        #print('bc', c-b)
        #print('cd', d-c)
        #print('de', e-d)
        #print('ef', f-e)
        #print('total', f-a)
    
    def export_ldraw(self, path, instances=None):
        if instances is None:
            instances = self.instances
        
        directory, file_name = os.path.split(path)
        lines = [
                '0 FILE %s'%file_name,
                '0 Main',
                '0 Name: %s'%file_name,
                '0 Author: LTRON',
        ]
        for instance in instances:
            instance = self.instances[int(instance)]
            color = instance.color
            t = self.upright @ instance.transform
            str_transform = (' '.join(['%f']*12))%(
                    t[0,3], t[1,3], t[2,3],
                    t[0,0], t[0,1], t[0,2],
                    t[1,0], t[1,1], t[1,2],
                    t[2,0], t[2,1], t[2,2])
                    
            brick_shape_name = str(instance.brick_shape)
            line = '1 %s %s %s'%(color, str_transform, brick_shape_name)
            lines.append(line)
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    
    def set_assembly(self, assembly, shape_ids, color_ids):
        self.clear_instances()
        self.import_assembly(
            assembly, shape_ids, color_ids, match_instance_ids=True)
    
    def import_assembly(
        self,
        assembly,
        shape_ids,
        color_ids,
        match_instance_ids=False,
    ):
        for i in range(len(assembly['shape'])):
            instance_shape = assembly['shape'][i]
            if instance_shape == 0:
                continue
            instance_color = assembly['color'][i]
            instance_pose = assembly['pose'][i]
            shape_labels = {
                value:key for key, value in shape_ids.items()}
            color_labels = {
                value:key for key, value in color_ids.items()}
            try:
                brick_shape = shape_labels[instance_shape]
            except KeyError:
                raise MissingClassError(instance_shape)
            try:
                color = color_labels[instance_color]
            except KeyError:
                raise MissingColorError
            
            if match_instance_ids:
                instance_id = i
            else:
                instance_id = None
            self.add_instance(
                brick_shape, color, instance_pose, instance_id=instance_id)
    
    def make_shape_ids(self):
        brick_shapes = [str(bt) for bt in self.shape_library.values()]
        shape_ids = {bt:i+1 for i, bt in enumerate(brick_shapes)}
        return shape_ids
    
    def make_color_ids(self):
        colors = [int(c) for c in self.color_library.values()]
        color_ids = {str(c):i for i, c in enumerate(sorted(colors))}
        return color_ids
    
    def get_assembly(
        self,
        shape_ids=None,
        color_ids=None,
        max_instances=None,
        max_edges=None,
        unidirectional=False,
    ):
        assembly = {}
        
        if shape_ids is None:
            shape_ids = self.make_shape_ids()
        if color_ids is None:
            color_ids = self.make_color_ids()
        if max_instances is None:
            if len(self.instances.keys()):
                max_instances = max(self.instances.keys())
            else:
                max_instances = 0
        else:
            if len(self.instances):
                assert max(self.instances.keys()) <= max_instances, (
                    'Instance ids %s larger than max_instances: %i'%(
                    list(self.instances.keys()), max_instances))
        assembly['shape'] = numpy.zeros((max_instances+1,), dtype=numpy.long)
        assembly['color'] = numpy.zeros((max_instances+1,), dtype=numpy.long)
        assembly['pose'] = numpy.zeros((max_instances+1, 4, 4))
        for instance_id, instance in self.instances.items():
            try:
                assembly['shape'][instance_id] = shape_ids[
                    str(instance.brick_shape)]
            except KeyError:
                raise MissingClassError(instance.brick_shape)
            try:
                assembly['color'][instance_id] = color_ids[str(instance.color)]
            except KeyError:
                raise MissingColorError
            assembly['pose'][instance_id] = instance.transform
        
        all_edges = self.get_all_edges(unidirectional=unidirectional)
        num_edges = all_edges.shape[1]
        if max_edges is not None:
            assert all_edges.shape[1] <= max_edges, 'Too many edges'
            extra_edges = numpy.zeros(
                (4, max_edges - num_edges), dtype=numpy.long)
            all_edges = numpy.concatenate((all_edges, extra_edges), axis=1)
        assembly['edges'] = all_edges
        
        return assembly
    
    
    # assets -------------------------------------------------------------------
    def clear_assets(self):
        self.clear_instances()
        self.shape_library.clear()
        self.color_library.clear()
        if self.renderable:
            self.render_environment.clear_meshes()
            self.render_environment.clear_materials()
            self.render_environment.clear_image_lights()
    
    
    # brick shapes -------------------------------------------------------------
    def add_brick_shape(self, brick_shape):
        new_shape = self.shape_library.add_shape(brick_shape)
        if self.renderable:
            self.render_environment.load_brick_mesh(new_shape)
        return new_shape
    
    
    # instances ----------------------------------------------------------------
    def add_instance(
        self, brick_shape, brick_color, transform, instance_id=None
    ):
        # TODO: what is this about?
        if self.renderable and self.render_environment.window is not None:
            self.render_environment.window.set_active()
        self.shape_library.add_shape(brick_shape)
        self.color_library.load_colors([brick_color])
        brick_instance = self.instances.add_instance(
                brick_shape, brick_color, transform, instance_id=instance_id)
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
    
    '''
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
    '''
    
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
    
    def get_scene_bbox(self):
        vertices = []
        for i, instance in self.instances.items():
            vertices.append(instance.bbox_vertices())
        if len(vertices):
            vertices = numpy.concatenate(vertices, axis=1)
            vmin = numpy.min(vertices[:3], axis=1)
            vmax = numpy.max(vertices[:3], axis=1)
        else:
            vmin = numpy.zeros(3)
            vmax = numpy.zeros(3)
        return vmin, vmax
    
    
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
            
        return other_snaps
    
    def get_all_snap_connections(self, instances=None):
        assert self.track_snaps
        if instances is None:
            instances = self.instances
        
        snap_connections = {}
        for instance in instances:
            connections = self.get_instance_snap_connections(instance)
            snap_connections[str(instance)] = connections
        
        return snap_connections
    
    def get_all_edges(self, instances=None, unidirectional=False):
        assert self.track_snaps
        snap_connections = self.get_all_snap_connections(instances=instances)
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
        return all_edges.astype(numpy.long)
    
    def get_brick_neighbors(self, instances=None):
        if instances is None:
            instances = self.instances
        edges = self.get_all_edges(instances=instances, unidirectional=True)
        neighbor_ids = {int(brick):set() for brick in instances}
        for i in range(edges.shape[1]):
            instance_a, instance_b, snap_a, snap_b = edges[:,i]
            neighbor_ids[instance_a].add(instance_b)
            neighbor_ids[instance_b].add(instance_a)
        
        brick_order = list(sorted(neighbor_ids.keys()))
        bricks = [self.instances[brick_id] for brick_id in brick_order]
        neighbors = [
            [self.instances[n_id] for n_id in neighbor_ids[brick_id]]
            for brick_id in brick_order
        ]
        
        return bricks, neighbors
    
    def get_all_snaps(self):
        assert self.track_snaps
        
        all_snaps = set()
        for instance_id, instance in self.instances.items():
            brick_shape = instance.brick_shape
            all_snaps |= set(
                (instance_id, i)
                for i in range(len(brick_shape.snaps)))
        
        return all_snaps
    
    def get_unoccupied_snaps(self):
        assert self.track_snaps
        
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
        
        return unoccupied_snaps
    
    def pick_and_place_snap_transform(self,
        pick,
        place,
        check_collisions=False
    ):
        assert pyquaternion_available
        
        if check_collisions:
            assert self.collision_checker is not None
        
        pick_instance_id, pick_snap_id = pick
        pick_instance = self.instances[pick_instance_id]
        pick_snap = pick_instance.get_snap(pick_snap_id)
        pick_transform = unscale_transform(pick_snap.transform.copy())
        inv_pick_transform = numpy.linalg.inv(pick_transform)
        pick_instance_transform = pick_instance.transform.copy()
        if place is None:
            place_transform = self.upright
        else:
            place_instance_id, place_snap_id = place
            place_instance = self.instances[place_instance_id]
            place_snap = place_instance.get_snap(place_snap_id)
            place_transform = unscale_transform(place_snap.transform.copy())
        
        best_transform = None
        best_pseudo_angle = -float('inf')
        for i in range(4):
            angle = i * math.pi/2.
            rotation = Quaternion(axis=(0,1,0), angle=angle)
            candidate_transform = (
                place_transform @
                rotation.transformation_matrix @
                inv_pick_transform @
                pick_instance_transform
            )
            
            offset = (
                candidate_transform @
                numpy.linalg.inv(pick_instance_transform)
            )
            pseudo_angle = numpy.trace(offset[:3,:3])
            if pseudo_angle > best_pseudo_angle:
                if check_collisions:
                    self.move_instance(pick_instance, candidate_transform)
                    collision = self.check_snap_collision(
                        [pick_instance],
                        pick_instance.get_snap(pick_snap_id),
                    )
                    self.move_instance(pick_instance, pick_instance_transform)
                    if collision:
                        continue
                best_transform = candidate_transform
                best_pseudo_angle = pseudo_angle
        
        return best_transform
    
    def pick_and_place_snap(self, pick, place, check_collisions=False):
        pick_instance = self.instances[pick[0]]
        transform = self.pick_and_place_snap_transform(
            pick, place, check_collisions=check_collisions)
        if transform is None:
            return False
        else:
            self.move_instance(pick_instance, transform)
            return True
    
    def transform_about_snap(self, instances, snap, local_transform):
        offset = (
            snap.transform @
            local_transform @
            numpy.linalg.inv(snap.transform)
        )
        for instance in instances:
            self.move_instance(instance, offset @ instance.transform)
    
    
    # materials ----------------------------------------------------------------
    def load_colors(self, colors):
        new_colors = self.color_library.load_colors(colors)
        if self.renderable:
            for new_color in new_colors:
                self.render_environment.load_color_material(new_color)
        return new_colors
    
    
    # rendering ----------------------------------------------------------------
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
    
    # collision checking -------------------------------------------------------
    def check_collision(
        self, target_instances, render_transform, scene_instances=None
    ):
        assert self.collision_checker is not None
        if self.render_environment.window is not None:
            self.render_environment.window.set_active()
        return self.collision_checker.check_collision(
            target_instances, render_transform, scene_instances=scene_instances)
    
    def check_snap_collision(
        self, target_instances, snap, *args, **kwargs
    ):
        assert self.collision_checker is not None
        if self.render_environment.window is not None:
            self.render_environment.window.set_active()
        return self.collision_checker.check_snap_collision(
            target_instances, snap, *args, **kwargs)
