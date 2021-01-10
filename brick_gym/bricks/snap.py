import copy

try:
    import renderpy.primitives as primitives
    renderpy_available = True
except ImportError:
    renderpy_available = False

from brick_gym.ldraw.commands import *
from brick_gym.ldraw.exceptions import LDrawException
from brick_gym.geometry.utils import close_enough

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
    
    '''
    grid_parts = grid.split()
    if grid_parts[0] == 'C':
        center_x = True
        grid_parts = grid_parts[1:]
    else:
        center_x = False
    grid_x = int(grid_parts[0])
    if grid_parts[1] == 'C':
        center_z = True
        grid_parts = [grid_parts[0]] + grid_parts[2:]
    else:
        center_z = False
    grid_z = int(grid_parts[1])
    grid_spacing_x = float(grid_parts[2])
    grid_spacing_z = float(grid_parts[3])
    '''
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
        self.gender = command.gender
        self.center = command.flags.get('center', 'false').lower() == 'true'
        self.mirror = command.flags.get('mirror', 'none')
        self.scale = command.flags.get('scale', 'none')
        
        # do checks here on center/mirror/scale to change the transform
        
        self.transform = transform
    
    def transformed_copy(self, transform):
        copied_snap = copy.copy(self)
        copied_snap.transform = numpy.dot(transform, self.transform)
        return copied_snap
    
class SnapCylinder(SnapStyle):
    
    def __init__(self, command, transform):
        super(SnapCylinder, self).__init__(command, transform)
        self.secs = command.flags['secs']
        self.caps = command.flags.get('caps', 'one')
        self.slide = command.flags.get('slide', 'false')
        self.subtype_id = 'cyl|%s|%s|%s'%(self.secs, self.caps, self.gender)
    
    def connected(self, other, tolerance=0.5):
        if not isinstance(other, (SnapCylinder, SnapClip)):
            return False
        
        if self.gender == other.gender:
            return False
        
        self_center = self.transform[:3,3]
        other_center = self.transform[:3,3]
        if close_enough(self.center, other.center, tolerance):
            return True
    
    def get_snap_mesh(self):
        assert renderpy_available
        sec_parts = self.secs.split()
        sec_parts = zip(sec_parts[::3], sec_parts[1::3], sec_parts[2::3])
        sections = []
        previous_length = 0
        for cross_section, radius, length in sec_parts:
            radius = float(radius)
            if self.gender == 'F':
                radius *= 1.01
            length = -float(length)
            sections.append((radius, length + previous_length))
            previous_length = length
        return primitives.multi_cylinder(sections = sections)

class SnapClip(SnapStyle):
    pass

class SnapFinger(SnapStyle):
    pass

class SnapGeneric(SnapStyle):
    pass

class SnapSphere(SnapStyle):
    def __init__(self, command, transform):
        super(SnapSphere, self).__init__(command, transform)
        self.radius = command.flags['radius']
        assert self.scale in ('none', 'ROnly')

'''
def snap_points_from_command(command, reference_transform):
    # TODO:
    # center flag
    # scale flag
    # mirror flag
    # gender
    # secs flag (maybe don't care?)
    # caps flag (probably don't care)
    # slide flag (probably don't care)
    base_transform = numpy.dot(reference_transform, command.transform)
    if 'grid' in command.flags:
        snap_transforms = griderate(command.flags['grid'], base_transform)
    else:
        snap_transforms = [base_transform]
    
    snap_points = [SnapPoint(
                        command.id,
                        transform)
                for transform in snap_transforms]
    return snap_points

def snap_points_from_part_document(document):
    def snap_points_from_nested_document(document, transform):
        snap_points = []
        for command in document.commands:
            if isinstance(command, LDrawImportCommand):
                reference_name = command.clean_reference_name
                reference_document = (
                        document.reference_table['ldraw'][reference_name])
                reference_transform = numpy.dot(transform, command.transform)
                snap_points.extend(snap_points_from_nested_document(
                        reference_document, reference_transform))
            elif isinstance(command, LDCadSnapInclCommand):
                reference_name = command.clean_reference_name
                reference_document = (
                        document.reference_table['shadow'][reference_name])
                reference_transform = numpy.dot(transform, command.transform)
                snap_points.extend(snap_points_from_nested_document(
                        reference_document, reference_transform))
            elif isinstance(command, LDCadSnapStyleCommand):
                snap_points.extend(snap_points_from_command(command, transform))
            elif isinstance(command, LDCadSnapClearCommand):
                snap_points.append(SnapClear(command.id))
        
        if not document.shadow:
            if document.clean_name in document.reference_table['shadow']:
                shadow_document = (
                        document.reference_table['shadow'][document.clean_name])
                snap_points.extend(snap_points_from_nested_document(
                        shadow_document, transform))
        
        return snap_points
    
    snap_points = snap_points_from_nested_document(document, numpy.eye(4))
    
    resolved_snap_points = []
    for snap_point in snap_points:
        if isinstance(snap_point, SnapPoint):
            resolved_snap_points.append(snap_point)
        elif isinstance(snap_point, SnapClear):
            if snap_point.id == '':
                resolved_snap_points.clear()
            else:
                resolved_snap_points = [
                        p for p in resolved_snap_points
                        if p.id != snap_point.id]
    
    return resolved_snap_points
'''
