import math
import collections
#import copy

from pyquaternion import Quaternion

try:
    import splendor.primitives as primitives
    splendor_available = True
except ImportError:
    splendor_available = False

from ltron.ldraw.commands import *
from ltron.exceptions import LtronException
from ltron.geometry.utils import (
    metric_close_enough, default_allclose, translate_matrix)
#from ltron.geometry.epsilon_array import EpsilonArray
from ltron.geometry.deduplicate_spatial import (
    deduplicate, rotation_doublecheck_function)
from ltron.geometry.utils import unscale_transform

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
        def delegate(command, transform):
            if isinstance(command, LDCadSnapCylCommand):
                return SnapCylinder.construct_snaps(command, transform)
            #elif isinstance(command, LDCadSnapClpCommand):
            #    return SnapClip.construct_snaps(command, transform)
            elif isinstance(command, LDCadSnapFgrCommand):
                return SnapFinger.construct_snaps(command, transform)
            #elif isinstance(command, LDCadSnapGenCommand):
            #    return SnapGeneric.construct_snaps(command, transform)
            #elif isinstance(command, LDCadSnapSphCommand):
            #    return SnapSphere.construct_snaps(command, transform)
            else:
                return []
        
        snap_transform = numpy.dot(reference_transform, command.transform)
        snaps = []
        if 'grid' in command.flags:
            grid_transforms = griderate(command.flags['grid'], snap_transform)
            for grid_transform in grid_transforms:
                snaps.extend(delegate(command, grid_transform))
        else:
            snaps.extend(delegate(command, snap_transform))
        
        return snaps
    
    def __init__(self, command):
        super().__init__(command)
        self.group = command.flags.get('group', None)
    
    def __int__(self):
        return self.snap_id
    
    def compatible(self, other):
        if self.group is None and other.group is None:
            return True
        else:
            return self.group == other.group

class SnapCylinder(SnapStyle):
    @staticmethod
    def construct_snaps(command, transform):
        p = gender_to_polarity[command.flags['gender']]
        secs = command.flags['secs']
        center = command.flags.get('center', 'false').lower() == 'true'
        sec_parts = secs.split()
        cross_section = [c.lower() for c in sec_parts[0::3]]
        radius = [float(r) for r in sec_parts[1::3]]
        length = [float(h) for h in sec_parts[2::3]]
        total_length = sum(length)
        num_sections = len(length) - 1
        
        caps = command.flags.get('caps', 'one')
        slide = str_to_bool(command.flags.get('slide', 'false'))
        
        snaps = []
        cumulative_length = 0
        if p == '+':
            for i, (c, r, l) in enumerate(zip(cross_section, radius, length)):
                first = i == 0
                last = i == num_sections
                ty = -cumulative_length
                if center:
                    ty += total_length / 2
                
                # axle 4 12
                if (c == 'r' and
                    r == 4 and
                    l == 11 and
                    not last and
                    cross_section[i+1] == '_l' and
                    radius[i+1] == 4.25 and
                    length[i+1] == 1
                ):
                    snaps.append(Axle_4_12(command, transform))
                
                # pin
                if c == 'r' and r == 6 and l == 16 and caps != 'both':
                    ty -= 3
                    first_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPin(command, first_transform))
                    
                    ty -= 10
                    second_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPin(command, second_transform))
                
                # half pin
                elif c == 'r' and r == 6 and l == 6 and caps != 'both':
                    ty -= 3
                    halfpin_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPin(command, halfpin_transform))
                
                # stud
                elif c == 'r' and r == 6 and (first or last) and caps != 'both':
                    if not(caps == 'one' and first and num_sections > 1):
                        if caps == 'none' and first:
                            t = translate_matrix([0,ty-l,0])
                            flip = numpy.array([
                                [-1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1]]
                            )
                            stud_transform = transform @ t @ flip
                        else:
                            t = translate_matrix([0,ty,0])
                            stud_transform = transform @ t
                        snaps.append(Stud(command, l, stud_transform))
                
                # bar
                elif p == '+' and c == 'r' and r == 4 and l >= 8:
                    # skip bars for now
                    pass
                    #for section in whatever:
                    #    bar_transform = SOMETHING
                    #    snaps.append(Bar(length=s, transform=bar_transform))
                    #continue
                
                cumulative_length += l
        
        elif p == '-':
            for i, (c, r, l) in enumerate(zip(cross_section, radius, length)):
                first = i == 0
                last = i == num_sections
                ty = -cumulative_length
                if center:
                    ty += total_length / 2
                
                # axle 4 12
                if (c == 'r' and
                    r == 4 and
                    l == 11 and
                    not last and
                    radius[i+1] == 5 and
                    length[i+1] == 1
                ):
                    snaps.append(AxleHole_4_12(command, transform))
                
                # pin hole
                if (c == 'r' and
                    r == 6 and
                    l == 16 and
                    caps != 'both' and
                    not first and
                    not last
                ):
                    ty -= 3
                    first_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPinHole(command, first_transform))
                    
                    #ty -= 15
                    ty -= 10
                    second_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPinHole(command, second_transform))
                
                # half pin hole
                elif (
                    c == 'r' and
                    r == 6 and
                    l == 6 and
                    caps != 'both' and
                    not first and
                    not last
                ):
                    ty -= 3
                    halfpin_transform = transform @ translate_matrix([0,ty,0])
                    snaps.append(HalfPinHole(command, halfpin_transform))
                
                # stud hole
                elif r == 6 and (first or last) and caps != 'both':
                    if not(caps == 'one' and first and num_sections > 1):
                        hole_transform = transform @ translate_matrix([0,ty,0])
                        snaps.append(StudHole(command, l, hole_transform))
                
                cumulative_length += l
        
        return snaps
        
    def __init__(self, command):
        super().__init__(command)
    
    def is_upright(self):
        return unscale_transform(self.transform.copy())[:3,1] @ [0,1,0] >= 0.999
    
    def equivalent(self, other):
        if type(self) != type(other):
            return False
        
        if self.polarity != other.polarity:
            return False
        
        if self.radius != other.radius:
            return False
        
        if self.length != other.length:
            return False
        
        p1 = (self.transform @ [0,0,0,1])[:3]
        p2 = (other.transform @ [0,0,0,1])[:3]
        if numpy.linalg.norm(p1 - p2) > self.search_radius:
            return False
        
        n1 = (self.transform @ [0,1,0,0])[:3]
        n1 /= numpy.linalg.norm(n1)
        n2 = (other.transform @ [0,1,0,0])[:3]
        n2 /= numpy.linalg.norm(n2)
        if numpy.dot(n1, n2) < 0.99:
            return False
        
        return True
    
    def get_collision_direction_transforms(self):
        # return a list of directions that this snap can be pushed onto another
        if self.polarity == '+':
            sign = 1
        elif self.polarity == '-':
            sign = -1
        
        return [
            numpy.array([
                [ 1, 0,    0, 0],
                [ 0, 0, sign, 0],
                [ 0, 1,    0, 0],
                [ 0, 0,    0, 1]
            ])
        ]
    
    collision_direction_transforms = property(
        get_collision_direction_transforms)

class UniversalSnap(SnapStyle):
    group = None
    def __init__(self, transform):
        self.transform = transform
        self.snap_style = self

class Axle_4_12(SnapCylinder):
    polarity = '+'
    radius = 4
    search_radius = 1
    length = 12
    def __init__(self, command, transform):
        super().__init__(command)
        self.transform = transform
        self.subtype_id = 'cylinder(4,12,u)'
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
            start_height=0,
            sections=((self.radius, -self.length),),
            radial_resolution=16,
            start_cap=True,
            end_cap=True,
        )
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        return isinstance(other, AxleHole_4_12)
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        p = my_instance.transform[:3,3]
        q = other_instance.transform[:3,3]
        return metric_close_enough(p, q, 1.)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        return default_pick_and_place_transforms(my_instance, other_instance)

class AxleHole_4_12(SnapCylinder):
    polarity = '-'
    radius = 4
    search_radius = 1
    length = 12
    def __init__(self, command, transform):
        super().__init__(command)
        self.transform = transform
        self.subtype_id = 'cylinder(4,12,u)'
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
            start_height=0,
            sections=((self.radius, -self.length),),
            radial_resolution=16,
            start_cap=True,
            end_cap=True,
        )
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        return isinstance(other, Axle_4_12)
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        p = my_instance.transform[:3,3]
        q = other_instance.transform[:3,3]
        return metric_close_enough(p, q, 1.)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        t = default_pick_and_place_transforms(my_instance, other_instance)
        return t

class Stud(SnapCylinder):
    polarity = '+'
    radius = 6
    search_radius = 10
    def __init__(self, command, length, transform):
        super().__init__(command)
        self.length = length
        self.transform = transform
        self.subtype_id = 'cylinder(%.01f,%.01f,u)'%(
            round(self.radius, 1), round(self.length, 1))
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
            start_height=0,
            sections=((self.radius, -self.length),),
            radial_resolution=16,
            start_cap=True,
            end_cap=True,
        )
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        return isinstance(other, (StudHole, HalfPinHole, UniversalSnap))
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        if isinstance(other_instance.snap_style, StudHole):
            return stud_studhole_connected(my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPinHole):
            return stud_halfpinhole_connected(my_instance, other_instance)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        if isinstance(other_instance.snap_style, StudHole):
            return stud_studhole_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPinHole):
            return stud_halfpinhole_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, UniversalSnap):
            return default_pick_and_place_transforms(
                my_instance, other_instance)

class StudHole(SnapCylinder):
    polarity = '-'
    radius = 6
    search_radius = 10
    def __init__(self, command, length, transform):
        super().__init__(command)
        self.length = length
        self.transform = transform
        self.subtype_id = 'cylinder(6,%.01f,u)'%(round(self.length, 1))
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        
        return isinstance(other, (Stud, HalfPin, UniversalSnap))
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        if isinstance(other_instance.snap_style, Stud):
            return stud_studhole_connected(other_instance, my_instance)
        
        elif isinstance(other_instance.snap_style, HalfPin):
            return halfpin_studhole_connected(other_instance, my_instance)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        if isinstance(other_instance.snap_style, Stud):
            return studhole_stud_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPin):
            return studhole_halfpin_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, UniversalSnap):
            return default_pick_and_place_transforms(
                my_instance, other_instance)
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
                start_height=0,
                sections=((self.radius, -self.length),),
                radial_resolution=16,
                start_cap=True,
                end_cap=True)

class HalfPin(SnapCylinder):
    polarity = '+'
    radius = 6
    search_radius = 10
    length = 10
    subtype_id = 'cylinder(6,10,c)'
    
    def __init__(self, command, transform):
        super().__init__(command)
        self.transform = transform
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        return isinstance(other, (StudHole, HalfPinHole, UniversalSnap))
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        if isinstance(other_instance.snap_style, StudHole):
            return halfpin_studhole_connected(my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPinHole):
            return halfpin_halfpinhole_connected(my_instance, other_instance)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        if isinstance(other_instance.snap_style, StudHole):
            return halfpin_studhole_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPinHole):
            return halfpin_halfpinhole_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, UniversalSnap):
            return default_pick_and_place_transforms(
                my_instance, other_instance)
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
                start_height=5,
                sections=((self.radius, 5-self.length),),
                radial_resolution=16,
                start_cap=True,
                end_cap=True)

class HalfPinHole(SnapCylinder):
    polarity = '-'
    radius = 6
    search_radius = 10
    length = 10
    subtype_id = 'cylinder(6,10,c)'
    
    def __init__(self, command, transform):
        super().__init__(command)
        self.transform = transform
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        return isinstance(other, (Stud, HalfPin, UniversalSnap))
    
    def connected(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return False
        
        if isinstance(other_instance.snap_style, Stud):
            return stud_halfpinhole_connected(other_instance, my_instance)
        
        elif isinstance(other_instance.snap_style, HalfPin):
            return halfpin_halfpinhole_connected(other_instance, my_instance)
    
    def pick_and_place_transforms(self, my_instance, other_instance):
        if not self.compatible(other_instance.snap_style):
            return []
        
        if isinstance(other_instance.snap_style, Stud):
            return halfpinhole_stud_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, HalfPin):
            return halfpinhole_halfpin_pick_and_place_transforms(
                my_instance, other_instance)
        
        elif isinstance(other_instance.snap_style, UniversalSnap):
            return default_pick_and_place_transforms(
                my_instance, other_instance)
    
    def get_snap_mesh(self):
        assert splendor_available
        return primitives.multi_cylinder(
                start_height=5,
                sections=((self.radius, 5-self.length),),
                radial_resolution=16,
                start_cap=True,
                end_cap=True)

class SnapFinger(SnapStyle):
    @staticmethod
    def construct_snaps(command, transform):
        center = command.flags.get('center', 'false').lower() == 'true'
        seq = [float(s) for s in command.flags.get('seq', []).split()]
        g = command.flags.get('genderofs', 'm').lower()
        radius = command.flags.get('radius', 6.)
        
        snaps = []
        cumulative_length = 0
        
        if (command.flags.get('group', '').lower() == 'lckhng' or
            (radius == 6 and numpy.allclose(seq, [4,8,4])) or
            (radius == 6 and numpy.allclose(seq, [4.5,8,4.5])) or
            (radius == 6 and numpy.allclose(seq, [8]))
        ):
            
            # enforce the group
            command.flags['group'] = 'lckHng'
            
            if len(seq) == 1:
                snaps.append(InsideLockHinge(command, transform))
            elif len(seq) == 3:
                snaps.append(OutsideLockHinge(command, transform))
            else:
                raise Exception('bad lock hinge')
        
        elif command.flags.get('group','').lower() == 'hgbrc':
            if g == 'f':
                snaps.append(DoubleStudHingeHousing(command, transform))
            elif g == 'm':
                snaps.append(DoubleStudHingeInsert(command, transform))
            else:
                raise Exception('bad double snap hinge')
        
        elif len(seq) == 5 and numpy.allclose(seq, [4,4,4,4,4]):
            if g == 'f':
                snaps.append(Neg44444Finger(command, transform))
            elif g == 'm':
                snaps.append(Pos44444Finger(command, transform))
            else:
                raise Exception('bad 44444 finger')
        
        elif len(seq) == 3 and numpy.allclose(seq, [4,24,4]):
            if g == 'f':
                snaps.append(BoxCoverFinger(command, transform))
            elif g == 'm':
                snaps.append(BoxFinger(command, transform))
            else:
                raise Exception('bad box finger')
        
        elif len(seq) == 9 and numpy.allclose(seq, [4,14,4,16,4,16,4,14,4]):
            if g == 'f':
                snaps.append(NegQuadHinge(command, transform))
            elif g == 'm':
                snaps.append(PosQuadHinge(command, transform))
            else:
                raise Exception('bad quad hinge')
        
        return snaps
    
    def __init__(self, command, transform):
        super().__init__(command)
        self.transform = transform
    
    def is_upright(self):
        return unscale_transform(self.transform.copy())[:3,1] @ [0,1,0] >= 0.999
    
    def equivalent(self, other):
        if type(self) != type(other):
            return False
        
        if self.polarity != other.polarity:
            return False
        
        p1 = (self.transform @ [0,0,0,1])[:3]
        p2 = (other.transform @ [0,0,0,1])[:3]
        if numpy.linalg.norm(p1 - p2) > self.search_radius:
            return False
        
        #n1 = (self.transform @ [0,1,0,0])[:3]
        #n1 /= numpy.linalg.norm(n1)
        #n2 = (other.transform @ [0,1,0,0])[:3]
        #n2 /= numpy.linalg.norm(n2)
        #if numpy.dot(n1, n2) < 0.99:
        #    return False
        
        return True
    
    def get_collision_direction_transforms(self):
        # return a list of directions that this snap can be pushed onto another
        if self.polarity == '+':
            sign = 1
        elif self.polarity == '-':
            sign = -1

        return [
            numpy.array([
                [ 1, 0,    0, 0],
                [ 0, 1,    0, 0],
                [ 0, 0, sign, 0],
                [ 0, 0,    0, 1]
            ]),
            numpy.array([
                [ 0, 0,-sign, 0],
                [ 0, 1,    0, 0],
                [ 1, 0,    0, 0],
                [ 0, 0,    0, 1]
            ]),
            numpy.array([
                [-1, 0,    0, 0],
                [ 0, 1,    0, 0],
                [ 0, 0,-sign, 0],
                [ 0, 0,    0, 1]
            ]),
            numpy.array([
                [ 0, 0, sign, 0],
                [ 0, 1,    0, 0],
                [-1, 0,    0, 0],
                [ 0, 0,    0, 1]
            ]),
            ####
            #numpy.array([
            #    [-1, 0,    0, 0],
            #    [ 0,-1,    0, 0],
            #    [ 0, 0, sign, 0],
            #    [ 0, 0,    0, 1]
            #]),
            #numpy.array([
            #    [ 0, 0,-sign, 0],
            #    [ 0,-1,    0, 0],
            #    [-1, 0,    0, 0],
            #    [ 0, 0,    0, 1]
            #]),
            #numpy.array([
            #    [ 1, 0,    0, 0],
            #    [ 0,-1,    0, 0],
            #    [ 0, 0,-sign, 0],
            #    [ 0, 0,    0, 1]
            #]),
            #numpy.array([
            #    [ 0, 0, sign, 0],
            #    [ 0,-1,    0, 0],
            #    [ 1, 0,    0, 0],
            #    [ 0, 0,    0, 1]
            #]),
        ]

    collision_direction_transforms = property(
        get_collision_direction_transforms)

def make_finger_pair(radius, length, subtype_id):
    class SharedFinger(SnapFinger):
        search_radius = 2
        
        def __init__(self, command, transform):
            super().__init__(command, transform)
            self.radius = radius
            self.length = length
            self.subtype_id = subtype_id
        
        def connected(self, my_instance, other_instance):
            if not self.compatible(other_instance.snap_style):
                return False
            return generic_finger_connected(my_instance, other_instance)
        
        def pick_and_place_transforms(self, my_instance, other_instance):
            if not self.compatible(other_instance.snap_style):
                return []
            
            rotations = [
                numpy.array([
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [ 0, 0,-1, 0],
                    [ 0, 1, 0, 0],
                    [ 1, 0, 0, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [-1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                    [ 0, 0,-1, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [ 0, 0, 1, 0],
                    [ 0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [-1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [ 0, 0,-1, 0],
                    [ 0,-1, 0, 0],
                    [-1, 0, 0, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [ 1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0,-1, 0],
                    [ 0, 0, 0, 1]
                ]),
                numpy.array([
                    [ 0, 0, 1, 0],
                    [ 0,-1, 0, 0],
                    [ 1, 0, 0, 0],
                    [ 0, 0, 0, 1]
                ]),
            ]
            
            pick_transform = unscale_transform(
                my_instance.snap_style.transform.copy())
            inv_pick_transform = numpy.linalg.inv(pick_transform)
            place_transform = unscale_transform(other_instance.transform.copy())
            
            return [place_transform @ rotation @ inv_pick_transform
                for rotation in rotations]
        
        def get_snap_mesh(self):
            assert splendor_available
            return primitives.multi_cylinder(
                start_height=-self.length/2,
                sections=((self.radius, self.length/2),),
                radial_resolution=16,
                start_cap=True,
                end_cap=True,
            )
    
    class PosFinger(SharedFinger):
        polarity = '+'
        def compatible(self, other):
            if not super().compatible(other):
                return False
            return isinstance(other, NegFinger)
    
    class NegFinger(SharedFinger):
        polarity = '-'
        def compatible(self, other):
            if not super().compatible(other):
                return False
            return isinstance(other, PosFinger)

    return PosFinger, NegFinger

InsideLockHinge, OutsideLockHinge = make_finger_pair(6, 16, 'lock_hinge')
DoubleStudHingeInsert, DoubleStudHingeHousing = make_finger_pair(
    4, 40, 'double_stud_hinge')
Pos44444Finger, Neg44444Finger = make_finger_pair(4, 20, 'finger_44444')
BoxFinger, BoxCoverFinger = make_finger_pair(4, 32, 'box_finger')
PosQuadHinge, NegQuadHinge = make_finger_pair(4, 80, 'quad_hinge')

def default_pick_and_place_transforms(pick, place):
    pick_transform = unscale_transform(pick.snap_style.transform.copy())
    inv_pick_transform = numpy.linalg.inv(pick_transform)
    place_transform = unscale_transform(place.transform.copy())
    transforms = []
    for i in range(4):
        angle = i * math.pi/2.
        rotation = Quaternion(axis=(0,1,0), angle=angle)
        transform = (
            place_transform @
            rotation.transformation_matrix @
            inv_pick_transform #@
            #pick.brick_instance.transform
        )
        
        transforms.append(transform)
    
    return transforms

def stud_studhole_connected(stud, stud_hole):
    p = stud.transform[:3,3]
    q = stud_hole.transform[:3,3]
    return metric_close_enough(p, q, 2.)

def stud_studhole_pick_and_place_transforms(stud, studhole):
    return default_pick_and_place_transforms(stud, studhole)

def studhole_stud_pick_and_place_transforms(studhole, stud):
    return default_pick_and_place_transforms(studhole, stud)

def stud_halfpinhole_connected(stud, half_pin_hole):
    # check if the positions are close enough
    p = stud.transform[:3,3]
    a = (half_pin_hole.transform @ [0, 5,0,1])[:3]
    b = (half_pin_hole.transform @ [0,-5,0,1])[:3]
    if not (metric_close_enough(p, a, 2) or metric_close_enough(p, b, 2)):
        return False
    
    # make sure the stud is pointing towards the middle of the halfpinhole
    stud_direction = (stud.transform @ [0,1,0,0])[:3]
    center_to_stud = p - half_pin_hole.transform[:3,3]
    center_to_stud /= numpy.linalg.norm(center_to_stud)
    return numpy.dot(stud_direction, center_to_stud) > 0.99

def stud_halfpinhole_pick_and_place_transforms(stud, half_pin_hole):
    return default_pick_and_place_transforms(stud, half_pin_hole)

def halfpinhole_stud_pick_and_place_transforms(half_pin_hole, stud):
    return default_pick_and_place_transforms(half_pin_hole, stud)

def halfpin_studhole_connected(half_pin, stud_hole):
    p = stud_hole.transform[:3,3]
    a = (half_pin.transform @ [0, 5,0,1])[:3]
    b = (half_pin.transform @ [0,-5,0,1])[:3]
    if not (metric_close_enough(p, a, 2) or metric_close_enough(p, b, 2)):
        return False
    
    # make sure the stud hole is pointing toward the middle of the halfpin
    studhole_direction = (stud_hole.transform @ [0,1,0,0])[:3]
    center_to_studhole = p - half_pin.transform[:3,3]
    center_to_studhole /= numpy.linalg.norm(center_to_studhole)
    return numpy.dot(studhole_direction, center_to_studhole) > 0.99

def halfpin_studhole_pick_and_place_transforms(half_pin, stud_hole):
    return default_pick_and_place_transforms(half_pin, stud_hole)

def studhole_halfpin_pick_and_place_transforms(stud_hole, half_pin):
    return default_pick_and_place_transforms(stud_hole, half_pin)

def halfpin_halfpinhole_connected(half_pin, half_pin_hole):
    p = half_pin.transform[:3,3]
    q = half_pin_hole.transform[:3,3]
    return metric_close_enough(p, q, 1)

def halfpin_halfpinhole_pick_and_place_transforms(half_pin, half_pin_hole):
    return default_pick_and_place_transforms(half_pin, half_pin_hole)

def halfpinhole_halfpin_pick_and_place_transforms(half_pin_hole, half_pin):
    return default_pick_and_place_transforms(half_pin_hole, half_pin)

def generic_finger_connected(outside_lockhinge, inside_lockhinge):
    po = outside_lockhinge.transform[:3,3]
    pi = inside_lockhinge.transform[:3,3]
    if not metric_close_enough(po, pi, 2):
        print('not close enough')
        return False
    
    yo = outside_lockhinge.transform[:3,1]
    yi = inside_lockhinge.transform[:3,1]
    if numpy.abs(numpy.dot(yo, yi)) < 0.975:
        print('bad angle', numpy.dot(yo, yi))
        return False
    
    return True

def doublestudhinge_connected(housing, insert):
    return lockhinge_connected(housing, insert)

# OLD ==========================================================================

class SnapStyleOFF(Snap):
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
    
    def compatible(self, other):
        if not self.groups_match(other):
            return False
        
        return True
    
    def groups_match(self, other):
        if self.group is None and other.group is None:
            return True
        else:
            return self.group == other.group
    
    def __int__(self):
        return self.snap_id
    
    #def transformed_copy(self, transform):
    #    copied_snap = copy.copy(self)
    #    copied_snap.transform = numpy.dot(transform, self.transform)
    #    return copied_snap
    
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
    
class SnapCylinderOff(SnapStyle):
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
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        
        if not (isinstance(other, SnapCylinder) or isinstance(other, SnapClip)):
            return False
        
        if self.polarity == '+':
            if not other.polarity == '-':
                return False
        
        elif self.polarity == '-':
            if not other.polarity == '+':
                return False
        
        if isinstance(other, SnapCylinder):
            if self.sec_radius[0] != other.sec_radius[0]:
                return False
            
        elif isinstance(other, SnapClip):
            if self.sec_radius[0] != other.radius:
                return False
        
        return True
    
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
    
    def compatible(self, other):
        if not super().compatible(other):
            return False
        
        if not isinstance(other, SnapCylinder):
            return False
        
        if not other.polarity == '+':
            return False
        
        if self.radius != other.sec_radius[0]:
            return False
        
        return True
    
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

class SnapFingerOff(SnapStyle):
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
    #rdf = rotation_doublecheck_function(max_angular_distance)
    def doublecheck_function(a, b):
        return a.equivalent(b)
        #if a.subtype_id != b.subtype_id:
        #    return False
        #if type(a) != type(b):
        #    return False
        #return rdf(a.transform,b.transform)
    
    deduplicate_indices = deduplicate(
        points,
        max_metric_distance,
        doublecheck_values=snaps,
        doublecheck_function=doublecheck_function,
    )
    
    return [snaps[i] for i in deduplicate_indices]

class SnapStyleSequence(collections.abc.Sequence):
    def __init__(self, snap_styles=None):
        if snap_styles is None:
            snap_styles = []
        self.snap_styles = snap_styles
        
        # hack for now to get snap_ids
        for i, snap in enumerate(self.snap_styles):
            snap.snap_id = i
    
    def __getitem__(self, key):
        return self.snap_styles[int(key)]
    
    def __len__(self):
        return len(self.snap_styles)
    
    # this is not used yet, but needs to be where we go
    '''
    def extend_from_command(self, command, reference_transform):
        snap_styles = SnapStyle.construct_snaps(command, reference_transform)
        # this is a hack FOR NOW
        for i, snap_style in snap_styles:
            snap_style.snap_id = len(self.snap_styles + i)
        self.snap_styles.extend(snap_styles)
    '''

class SnapInstanceSequence(collections.abc.Sequence):
    def __init__(self, snap_styles, brick_instance):
        self.snap_instances = []
        for snap_style in snap_styles:
            self.snap_instances.append(SnapInstance(snap_style, brick_instance))
    
    def __getitem__(self, key):
        return self.snap_instances[int(key)]
    
    def __len__(self):
        return len(self.snap_instances)

class SnapInstance:
    def __init__(self, snap_style, brick_instance):
        self.snap_style = snap_style
        self.brick_instance = brick_instance
    
    def get_transform(self):
        return self.brick_instance.transform @ self.snap_style.transform
    
    transform = property(get_transform)
    
    def compatible(self, other):
        if isinstance(other, SnapInstance):
            return self.snap_style.compatible(other.snap_style)
        else:
            return self.snap_style.compatible(other)
    
    # for tuple conversion
    def __getitem__(self, i):
        if i == 0:
            return int(self.brick_instance)
        elif i == 1:
            return int(self.snap_style)
        else:
            raise IndexError
    
    def __hash__(self):
        return hash(tuple(self))
    
    def __eq__(self, other):
        return tuple(self) == tuple(other)
    
    def __str__(self):
        return '%i_%i'%tuple(self)
    
    def connected(self, other, unidirectional=False):
        if unidirectional and (
            int(self.brick_instance) > int(other.brick_instance)):
            return False
        
        return self.snap_style.connected(self, other)
    
    def __getattr__(self, attr):
        return getattr(self.snap_style, attr)
    
    def get_collision_direction_transforms(self):
        directions = self.snap_style.get_collision_direction_transforms()
        return [self.transform @ direction for direction in directions]
    
    collision_direction_transforms = property(
        get_collision_direction_transforms
    )
