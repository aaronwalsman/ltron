import copy

try:
    import splendor.primitives as primitives
    splendor_available = True
except ImportError:
    splendor_available = False

from ltron.ldraw.commands import *
from ltron.exceptions import LtronException
from ltron.geometry.utils import metric_close_enough, default_allclose
#from ltron.geometry.epsilon_array import EpsilonArray
from ltron.geometry.deduplicate_spatial import (
    deduplicate, rotation_doublecheck_function)

gender_to_polarity = {
    'M':'+',
    'm':'+',
    'F':'-',
    'f':'-',
}

def str_to_bool(s):
    return s.lower() == 'true'

class BadGridException(LtronException):
    pass

def griderate(grid, transform):
    if grid is None:
        return [transform]
    
    # 3D grids are a thing
    grid_parts = []
    grid_tokens = grid.split()
    while grid_tokens:
        if grid_tokens[0] == 'C':
            grid_parts.append((grid_tokens[1], True))
            grid_tokens = grid_tokens[2:]
        else:
            grid_parts.append((grid_tokens[0], False))
            grid_tokens = grid_tokens[1:]
    
    if len(grid_parts) == 4:
        grid_x = int(grid_parts[0][0])
        center_x = grid_parts[0][1]
        grid_y = 1
        center_y = False
        grid_z = int(grid_parts[1][0])
        center_z = grid_parts[1][1]
        grid_spacing_x = float(grid_parts[2][0])
        grid_spacing_y = 0.
        grid_spacing_z = float(grid_parts[3][0])
    
    elif len(grid_parts) == 6:
        grid_x = int(grid_parts[0][0])
        center_x = grid_parts[0][1]
        grid_y = int(grid_parts[1][0])
        center_y = grid_parts[1][1]
        grid_z = int(grid_parts[2][0])
        center_z = grid_parts[2][1]
        grid_spacing_x = float(grid_parts[3][0])
        grid_spacing_y = float(grid_parts[4][0])
        grid_spacing_z = float(grid_parts[5][0])
    
    else:
        raise BadGridException('LDCad grid info must be either 2d or 3d')
    
    if center_x:
        x_width = grid_spacing_x * (grid_x-1)
        x_offset = -x_width/2.
    else:
        x_offset = 0.
    
    if center_y:
        y_width = grid_spacing_y * (grid_y-1)
        y_offset = -y_width/2.
    else:
        y_offset = 0.
    
    if center_z:
        z_width = grid_spacing_z * (grid_z-1)
        z_offset = -z_width/2.
    else:
        z_offset = 0.

    grid_transforms = []
    for x_index in range(grid_x):
        x = x_index * grid_spacing_x + x_offset
        for y_index in range(grid_y):
            y = y_index * grid_spacing_y + y_offset
            for z_index in range(grid_z):
                z = z_index * grid_spacing_z + z_offset
                translate = numpy.eye(4)
                translate[0,3] = x
                translate[1,3] = y
                translate[2,3] = z
                grid_transforms.append(numpy.dot(transform, translate))

    return grid_transforms

class Snap:
    @staticmethod
    def construct_snaps(command, reference_transform):
        if isinstance(command, LDCadSnapClearCommand):
            return [SnapClear(command)]
        if isinstance(command, LDCadSnapStyleCommand):
            return SnapStyle.construct_snaps(command, reference_transform)
    
    def __init__(self, command):
        self.type_id = command.id

class SnapClear(Snap):
    pass

class SnapStyle(Snap):
    @staticmethod
    def construct_snaps(command, reference_transform):
        def construct_snap(command, transform):
            if isinstance(command, LDCadSnapCylCommand):
                return SnapCylinder(command, transform)
            elif isinstance(command, LDCadSnapClpCommand):
                return SnapClip(command, transform)
            elif isinstance(command, LDCadSnapFgrCommand):
                return SnapFinger(command, transform)
            elif isinstance(command, LDCadSnapGenCommand):
                return SnapGeneric(command, transform)
            elif isinstance(command, LDCadSnapSphCommand):
                return SnapSphere(command, transform)
        
        snap_transform = numpy.dot(reference_transform, command.transform)
        snaps = []
        if 'grid' in command.flags:
            grid_transforms = griderate(command.flags['grid'], snap_transform)
            for grid_transform in grid_transforms:
                snaps.append(construct_snap(command, grid_transform))
        else:
            snaps.append(construct_snap(command, snap_transform))
        
        return snaps
    
    def __init__(self, command, transform):
        super(SnapStyle, self).__init__(command)
        self.center = command.flags.get('center', 'false').lower() == 'true'
        self.mirror = command.flags.get('mirror', 'none')
        self.scale = command.flags.get('scale', 'none')
        self.group = command.flags.get('group', None)
        
        # do checks here on center/mirror/scale to change the transform
        
        self.transform = transform
    
    def raw_data(self):
        return {
            'center' : self.center,
            'mirror' : self.mirror,
            'scale' : self.scale,
            'group' : self.group,
            'transform' : self.transform,
        }
    
    def is_upright(self):
        return False
    
    def connected2(self, other):
        return True
    
    def groups_match(self, other):
        if self.group is None and other.group is None:
            return True
        else:
            return self.group == other.group
    
    def transformed_copy(self, transform):
        copied_snap = copy.copy(self)
        copied_snap.transform = numpy.dot(transform, self.transform)
        return copied_snap
    
    # you know when people tell you something is a bad idea and you're like,
    # "no, it's cool man, I know what I'm doing, I got this."
    '''
    def __hash__(self):
        # WARNING: Instances of this class are totally 100% mutable.
        # This hash function is designed for 1-time deduplication using a set
        # that will not persist.  Do not use this class in a set or as a
        # key for a dictionary unless you know for certain that the instance
        # will not change during the lifetime of the set/dictionary.
        return hash((self.subtype_id, EpsilonArray(self.transform)))
    
    def __eq__(self, other):
        return (
            (self.subtype_id == other.subtype_id) and
            #numpy.allclose(self.transform, other.transform)
            default_allclose(self.transform, other.transform)
        )
    '''
    
class SnapCylinder(SnapStyle):
    style='cylinder'
    def __init__(self, command, transform):
        super(SnapCylinder, self).__init__(command, transform)
        self.polarity = gender_to_polarity[command.flags['gender']]
        self.secs = command.flags['secs']
        sec_parts = self.secs.split()
        self.sec_type = sec_parts[0::3]
        self.sec_radius = [float(r) for r in sec_parts[1::3]]
        self.sec_length = [float(h) for h in sec_parts[2::3]]
        
        self.caps = command.flags.get('caps', 'one')
        self.slide = str_to_bool(command.flags.get('slide', 'false'))
        center_string = ('uncentered', 'centered')[self.center]
        #self.subtype_id = 'cyl|%s|%s|%s'%(self.secs, self.caps, self.polarity)
        self.subtype_id = 'cylinder(%s,%s)'%(self.secs, center_string)
    
    def raw_data(self):
        raw_data = {
            'snap_type' : 'cylinder',
            'polarity' : self.polarity,
            'sec_type' : self.sec_type,
            'sec_radius' : self.sec_radius,
            'sec_length' : self.sec_length,
            'caps' : self.caps,
            'slide' : self.slide,
        }
        raw_data.update(super(SnapCylinder, self).raw_data())
        return raw_data
    
    def is_upright(self):
        axis = self.transform[:3,1]
        #if self.polarity == '+' and numpy.allclose(axis, (0,-1,0)):
        if self.polarity == '+' and default_allclose(axis, (0,-1,0)):
            return True
        #elif self.polarity == '-' and numpy.allclose(axis, (0,1,0)):
        elif self.polarity == '-' and default_allclose(axis, (0,1,0)):
            return True
        else:
            return False
    
    def bbox(self):
        r = max(self.sec_radii)
        h = sum(self.sec_heights)
        
        if self.center:
            half_h = h/2.
            y_h = self.transform[:3,1] * half_h
            # slightly faster than a matrix multiply
            p0 = -y_h + self.transform[:3,3]
            p1 = y_h + self.transform[:3,3]
        else:
            # slightly faster than a matrix multiply
            p0 = numpy.array([0,0,0])
            p1 = self.transform[:3,1] * h + self.transform[:3,3]
        
        # slightly slower than what I'm doing above
        #p0 = (self.transform @ p0)[:3]
        #p1 = (self.transform @ p1)[:3]
        box_min = numpy.min([p0, p1], axis=0) - r
        box_max = numpy.max([p0, p1], axis=0) + r
        
        #box_min = numpy.zeros(3)
        #box_max = numpy.ones(3)
        
        return box_min, box_max
    
    def connected(
        self,
        other,
        alignment_tolerance=0.95,
        distance_tolerance=0.5,
    ):
        if not self.groups_match(other):
            return False
        
        if not isinstance(other, (SnapCylinder, SnapClip)):
            return False
        
        if self.polarity == other.polarity:
            return False
        
        alignment = numpy.dot(self.transform[:3,1], other.transform[:3,1])
        if abs(alignment) < alignment_tolerance:
            return False
        
        # TODO: replace this with more appropriate cylinder matching ASAP
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3] #woops!
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        '''
        # This is half-way correct cylinder matching.
        # It still uses matching locations, which is bad.
        # It just makes sure that this snap and the other share at least one
        # common cross/section radius.
        self_center = self.transform[:3,3]
        other_center = other.transform[:3,3]
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        if isinstance(other, SnapClip):
            other_radius_sections = {('R', other.radius)}
        else:
            other_sec_parts = other.get_sec_parts()
            other_radius_sections = {
                (sec_part[0], sec_part[1]) for sec_part in other_sec_parts}
        self_sec_parts = self.get_sec_parts()
        self_radius_sections = {
            (sec_part[0], sec_part[1]) for sec_part in self_sec_parts}
        if not len(self_radius_sections & other_radius_sections):
            return False
        '''
        
        return True
    
    def get_sec_parts(self):
        sec_parts = self.secs.split()
        sec_parts = zip(sec_parts[::3], sec_parts[1::3], sec_parts[2::3])
        return list(sec_parts)
    
    def get_snap_mesh(self):
        assert splendor_available
        sec_parts = self.get_sec_parts()
        sections = []
        previous_length = 0
        if self.center:
            previous_length += sum(
                    float(sec_part[-1]) for sec_part in sec_parts)/2.
        start_height = previous_length
        for cross_section, radius, length in sec_parts:
            radius = float(radius)
            length = -float(length)
            sections.append((radius, length + previous_length))
            previous_length += length
        
        return primitives.multi_cylinder(
                start_height=start_height,
                sections=sections,
                radial_resolution=16,
                start_cap=True,
                end_cap=True)

class SnapClip(SnapStyle):
    style='clip'
    def __init__(self, command, transform):
        super(SnapClip, self).__init__(command, transform)
        self.polarity = '-'
        self.radius = float(command.flags.get('radius', 4.0))
        self.length = float(command.flags.get('length', 8.0))
        self.slide = str_to_bool(command.flags.get('slide', 'false'))
        center_string = ('uncentered', 'centered')[self.center]
        self.subtype_id = 'clip(%s,%s,%s)'%(
            self.radius, self.length, center_string)
    
    def raw_data(self):
        raw_data = {
            'snap_type' : 'clip',
            'polarity' : self.polarity,
            'radius' : self.radius,
            'length' : self.length,
            'slide' : self.slide,
        }
        raw_data.update(super(SnapClip, self).raw_data())
        return raw_data
    
    def bbox(self):
        r = self.radius
        h = self.length
        
        if self.center:
            p0 = numpy.array([0,-h/2.,0, 1])
            p1 = numpy.array([0, h/2.,0, 1])
        else:
            p0 = numpy.array([0,0,0,1])
            p1 = numpy.array([0,h,0,1])
        
        p0 = (self.transform @ p0)[:3]
        p1 = (self.transform @ p1)[:3]
        
        box_min = numpy.min([p0, p1], axis=0) - r
        box_max = numpy.max([p0, p1], axis=0) + r
        
        return box_min, box_max
    
    def connected(
        self,
        other,
        alignment_tolerance=0.95,
        distance_tolerance=0.5,
    ):
        if not self.groups_match(other):
            return False
        
        if not isinstance(other, (SnapCylinder)):
            return False
        
        if self.polarity == other.polarity:
            return False
        
        alignment = numpy.dot(self.transform[:3,1], other.transform[:3,1])
        if abs(alignment) < alignment_tolerance:
            return False
        
        # TODO: replace with proper cylinder matching
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3]
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        return True
    
    def get_snap_mesh(self):
        assert splendor_available
        
        start_height = 0
        end_height = -self.length
        if self.center:
            start_height = self.length/2.
            end_height = -self.length/2.
        
        return primitives.multi_cylinder(
                start_height=start_height,
                sections=[[self.radius, end_height]],
                radial_resolution=16,
                start_cap=True,
                end_cap=True)

class SnapFinger(SnapStyle):
    style='finger'
    def __init__(self, command, transform):
        super(SnapFinger, self).__init__(command, transform)
        self.polarity = gender_to_polarity[command.flags.get('genderofs', 'm')]
        self.seq = [float(s) for s in command.flags['seq'].split()]
        self.radius = float(command.flags.get('radius', 4.0))
        # center defaulting to true is not documented, but seems to be correct
        self.center = str_to_bool(command.flags.get('center', 'true'))
        center_string = ('uncentered', 'centered')[self.center]
        self.subtype_id = 'finger(%s,%i,%s)'%(
            self.radius, sum(self.seq), center_string)
    
    def raw_data(self):
        raw_data = {
            'snap_type' : 'finger',
            'polarity' : self.polarity,
            'seq' : self.seq,
            'radius' : self.radius,
        }
        raw_data.update(super(SnapFinger, self).raw_data())
        return raw_data
    
    def bbox(self):
        r = self.radius
        h = sum(self.seq)
        
        if self.center:
            p0 = numpy.array([0,-h/2.,0, 1])
            p1 = numpy.array([0, h/2.,0, 1])
        else:
            p0 = numpy.array([0,0,0,1])
            p1 = numpy.array([0,h,0,1])
        
        p0 = (self.transform @ p0)[:3]
        p1 = (self.transform @ p1)[:3]
        
        box_min = numpy.min([p0, p1], axis=0) - r
        box_max = numpy.max([p0, p1], axis=0) + r
        
        return box_min, box_max
    
    def connected(
        self,
        other,
        alignment_tolerance=0.95,
        distance_tolerance=0.5
    ):
        if not self.groups_match(other):
            return False
        
        if not isinstance(other, SnapFinger):
            return False
        
        #if self.polarity == other.polarity:
        #    return False
        
        if self.radius != other.radius:
            return False
        
        alignment = numpy.dot(self.transform[:3,1], other.transform[:3,1])
        if abs(alignment) < alignment_tolerance:
            return False
        
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3]
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        return True
    
    def get_snap_mesh(self):
        assert splendor_available
        
        length = sum(self.seq)
        start_height = 0
        end_height = -length
        if self.center:
            start_height = length/2.
            end_height = -length/2.
        
        return primitives.multi_cylinder(
            start_height = start_height,
            sections=[(self.radius, end_height)],
            radial_resolution=16,
            start_cap=True,
            end_cap=True,
        )

class SnapGeneric(SnapStyle):
    style='generic'
    def __init__(self, command, transform):
        super(SnapGeneric, self).__init__(command, transform)
        self.polarity = gender_to_polarity[command.flags.get('genderofs', 'm')]
        bounding = command.flags['bounding'].split()
        self.bounding = (bounding[0],) + tuple(float(b) for b in bounding[1:])
        self.subtype_id = 'generic(%s)'%command.flags['bounding']
    
    def raw_data(self):
        raw_data = {
            'snap_type' : 'generic',
            'polarity' : self.polarity,
            'bounding' : self.bounding,
        }
        raw_data.update(super(SnapGeneric, self).raw_data())
        return raw_data
    
    def bbox(self):
        raise NotImplementedError
    
    def connected(
        self,
        other,
        alignment_tolerance=0.95,
        distance_tolerance=0.5
    ):
        if not self.groups_match(other):
            return False
        
        if not isinstance(other, SnapFinger):
            return False
        
        if self.polarity == other.polarity:
            return False
        
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3]
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        return True
    
    def get_snap_mesh(self):
        assert splendor_available
        
        bounding_type, *bounding_args = self.bounding
        if bounding_type == 'pnt':
            return primitives.sphere(radius=1)
        
        elif bounding_type == 'box':
            x, y, z = bounding_args
            return primitives.cube(
                x_extents = [-x, x],
                y_extents = [-y, y],
                z_extents = [-z, z])
        
        elif bounding_type == 'cube':
            xyz, = bounding_args
            return primitives.cube(
                x_extents=[-xyz, xyz],
                y_extents=[-xyz, xyz],
                z_extents=[-xyz, xyz])
        
        elif bounding_type == 'cyl':
            radius, length = bounding_args
            return primitives.cylinder(
                start_height=length/2,
                end_height=-length/2,
                radius=radius,
                start_cap=True,
                end_cap=True)
        
        elif bounding_type == 'sph':
            radius, = bounding_args
            return primitives.sphere(radius=radius)
        
        else:
            raise NotImplementedError

class SnapSphere(SnapStyle):
    style='sphere'
    def __init__(self, command, transform):
        super(SnapSphere, self).__init__(command, transform)
        self.polarity = gender_to_polarity[command.flags['gender']]
        self.radius = command.flags['radius']
        self.subtype_id = 'sphere(%s)'%self.radius
        assert self.scale in ('none', 'ROnly')
    
    def raw_data(self):
        raw_data = {
            'snap_type' : 'sphere',
            'polarity' : self.polarity,
            'radius' : self.radius,
        }
        raw_data.update(super(SnapSphere, self).raw_data())
        return raw_data
    
    def bbox(self):
        r = self.radius
        
        p0 = self.transform[:3,3]
        
        box_min = p0 - r
        box_max = p0 + r
        
        return box_min, box_max
    
    def connected(
        self,
        other,
        alignment_tolerance=0.95,
        distance_tolerance=0.5,
    ):
        if not self.groups_match(other):
            return False
        
        if not isinstance(other, SnapSphere):
            return False
        
        if self.polarity == other.polarity:
            return False
        
        if self.radius != other.radius:
            return False
        
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3]
        if not metric_close_enough(
            self_center, other_center, distance_tolerance
        ):
            return False
        
        return True
    
    def get_snap_mesh(self):
        assert splendor_avaialable
        
        return primitives.sphere(radius=self.radius)

def filter_snaps(snaps, polarity=None, style=None):
    def f(snap):
        if polarity is not None and hasattr(snap, 'polarity'):
            if snap.polarity != polarity:
                return False
        if style is not None:
            if isinstance(style, str):
                style = (style,)
            if snap.style not in style:
                return False
        return True
    
    return filter(f, snaps)

def deduplicate_snaps(
    snaps,
    max_metric_distance=1.,
    max_angular_distance=0.08,
):
    points = [snap.transform[:3,3] for snap in snaps]
    rdf = rotation_doublecheck_function(max_angular_distance)
    def doublecheck_function(a, b):
        if a.subtype_id != b.subtype_id:
            return False
        return rdf(a.transform,b.transform)
    
    deduplicate_indices = deduplicate(
        points,
        max_metric_distance,
        doublecheck_values=snaps,
        doublecheck_function=doublecheck_function,
    )
    
    return [snaps[i] for i in deduplicate_indices]
