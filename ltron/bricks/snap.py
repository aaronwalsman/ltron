import copy

try:
    import splendor.primitives as primitives
    splendor_available = True
except ImportError:
    splendor_available = False

from ltron.ldraw.commands import *
from ltron.ldraw.exceptions import LDrawException
from ltron.geometry.utils import close_enough

gender_to_polarity = {
    'M':'+',
    'm':'+',
    'F':'-',
    'f':'-',
}

def str_to_bool(s):
    return s.lower() == 'true'

class BadGridException(LDrawException):
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
    
    def groups_match(self, other):
        if self.group is None and other.group is None:
            return True
        else:
            return self.group == other.group
    
    def transformed_copy(self, transform):
        copied_snap = copy.copy(self)
        copied_snap.transform = numpy.dot(transform, self.transform)
        return copied_snap
    
class SnapCylinder(SnapStyle):
    style='cylinder'
    def __init__(self, command, transform):
        super(SnapCylinder, self).__init__(command, transform)
        self.polarity = gender_to_polarity[command.flags['gender']]
        self.secs = command.flags['secs']
        self.caps = command.flags.get('caps', 'one')
        self.slide = str_to_bool(command.flags.get('slide', 'false'))
        #self.subtype_id = 'cyl|%s|%s|%s'%(self.secs, self.caps, self.polarity)
        self.subtype_id = 'cylinder(%s)'%self.secs
    
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
        other_center = self.transform[:3,3]
        if not close_enough(self_center, other_center, distance_tolerance):
            return False
        
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
        for cross_section, radius, length in sec_parts:
            radius = float(radius)
            length = -float(length)
            sections.append((radius, length + previous_length))
            previous_length += length
        
        return primitives.multi_cylinder(
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
        self.subtype_id = 'clip(%s,%s)'%(self.radius, self.length)
    
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
        if not close_enough(self_center, other_center, distance_tolerance):
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
        self.subtype_id = 'finger(%s,%i)'%(self.radius, sum(self.seq))
    
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
        if not close_enough(self_center, other_center, distance_tolerance):
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
        if not close_enough(self_center, other_center, distance_tolerance):
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
        if not close_enough(self_center, other_center, distance_tolerance):
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
