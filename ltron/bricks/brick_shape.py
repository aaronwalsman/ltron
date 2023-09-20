import collections

import numpy

from ltron.ldraw.commands import (
    LDrawImportCommand,
    LDCadSnapInclCommand,
    LDCadSnapStyleCommand,
    LDCadSnapClearCommand,
    LDrawContentCommand,
)

from ltron.ldraw.parts import LDRAW_PARTS
from ltron.ldraw.documents import (
    LDrawDocument,
    LDrawMPDMainFile,
    LDrawMPDInternalFile,
    LDrawLDR,
    LDrawDAT,
)
from ltron.ldraw.commands import LDrawCommand
from ltron.bricks.snap import (
    Snap,
    SnapStyle,
    SnapStyleSequence,
    SnapClear,
    deduplicate_snaps,
    griderate,
    StudHole,
    Tire_10_16,
    Wheel_10_16,
    Tire_13_28,
    Wheel_13_28,
)

class BrickShapeLibrary(collections.abc.MutableMapping):
    def __init__(self, brick_shapes=None):
        if brick_shapes is None:
            brick_shapes = {}
        self.brick_shapes = brick_shapes
    
    def add_shape(self, new_shape):
        if new_shape in self:
            return self[new_shape]
        
        if not isinstance(new_shape, BrickShape):
            new_shape = BrickShape.construct_shape(new_shape)
        self[new_shape.reference_name] = new_shape
        return new_shape
    
    def import_document(self, document):
        new_shapes = []
        for command in document.commands:
            if isinstance(command, LDrawImportCommand):
                reference_name = command.reference_name
                reference_document = (
                        document.reference_table['ldraw'][reference_name])
                if isinstance(reference_document, LDrawDAT):
                    if reference_name in LDRAW_PARTS:
                        new_shapes.append(self.add_shape(reference_document))
                elif isinstance(
                    reference_document, 
                    (LDrawMPDMainFile, LDrawMPDInternalFile, LDrawLDR),
                ):
                    new_shapes.extend(self.import_document(reference_document))
                    
        return new_shapes
    
    def __getitem__(self, key):
        return self.brick_shapes[str(key)]
   
    def __setitem__(self, key, value):
        assert isinstance(value, BrickShape)
        assert str(key) == str(value)
        self.brick_shapes[str(key)] = value
    
    def __delitem__(self, key):
        del(self.brick_shapes[str(key)])
    
    def __iter__(self):
        return iter(self.brick_shapes)
    
    def __len__(self):
        return len(self.brick_shapes)

class BrickShape:
    @staticmethod
    def construct_shape(document):
        if str(document) in custom_brick_shapes:
            return custom_brick_shapes[str(document)](document)
        else:
            return BrickShape(document)
    
    def __init__(self, document):
        if isinstance(document, str):
            document = LDrawDocument.parse_document(document)
        self.reference_name = document.reference_name
        self.mesh_name = self.reference_name.replace('.dat', '')
        self.document = document
        self.construct_snaps_and_vertices()
    
    def __str__(self):
        return self.reference_name
    
    def splendor_mesh_args(self):
        mesh_entry = {
            'mesh_asset' : self.mesh_name,
            'scale' : 1.0,
            'color_mode' : 'flat_color'
        }
        return mesh_entry
    
    def construct_snaps_and_vertices(self):
        try:
            snaps, self.vertices = snaps_and_vertices_from_nested_document(
                self.document)
        except:
            print('Error while importing: %s'%self.document.reference_name)
            raise
        
        resolved_snaps = []
        for snap in snaps:
            if isinstance(snap, SnapStyle):
                resolved_snaps.append(snap)
            elif isinstance(snap, SnapClear):
                if snap.type_id == '':
                    resolved_snaps.clear()
                else:
                    resolved_snaps = [
                            p for p in resolved_snaps
                            if p.type_id != snap.type_id]
        
        self.snaps = SnapStyleSequence(deduplicate_snaps(resolved_snaps))
        
        try:
            bb = numpy.array([
                numpy.min(self.vertices[:3], axis=1),
                numpy.max(self.vertices[:3], axis=1),
            ])
            self.empty_shape=False
        except ValueError:
            bb = numpy.array([[0,0,0], [0,0,0]])
            self.empty_shape=True
        self.bbox = bb
        self.bbox_vertices = numpy.array([
            [bb[0][0], bb[0][0], bb[0][0], bb[0][0],
             bb[1][0], bb[1][0], bb[1][0], bb[1][0]],
            [bb[0][1], bb[0][1], bb[1][1], bb[1][1],
             bb[0][1], bb[0][1], bb[1][1], bb[1][1]],
            [bb[0][2], bb[1][2], bb[0][2], bb[1][2],
             bb[0][2], bb[1][2], bb[0][2], bb[1][2]],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])
    
    def get_upright_snaps(self):
        return [snap for snap in self.snaps if snap.is_upright()]

class BrickShape_3641(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        tire_transform = numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 8],
            [ 0, 0, 0, 1]
        ])
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=F] [caps=one] [secs=R 10 16]')
        self.snaps.append(Tire_10_16(command, tire_transform))

class BrickShape_4084(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        tire_transform = numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 8],
            [ 0, 0, 0, 1]
        ])
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=F] [caps=one] [secs=R 10 16]')
        self.snaps.append(Tire_10_16(command, tire_transform))

class BrickShape_4624(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        wheel_transform = numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 8],
            [ 0, 0, 0, 1]
        ])
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=M] [caps=one] [secs=R 10 16]')
        self.snaps.append(Wheel_10_16(command, wheel_transform))

class BrickShape_6014(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        wheel_transform = numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 8],
            [ 0, 0, 0, 1]
        ])
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=M] [caps=one] [secs=R 13 28]')
        self.snaps.append(Wheel_13_28(command, wheel_transform))

class BrickShape_6015(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        tire_transform = numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 14],
            [ 0, 0, 0, 1]
        ])
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=F] [caps=one] [secs=R 13 28]')
        self.snaps.append(Tire_13_28(command, tire_transform))

class BrickShape_98138(BrickShape):
    def construct_snaps_and_vertices(self):
        super().construct_snaps_and_vertices()
        #self.snaps[0].transform[1,:3] *= -1
        command = LDrawCommand.parse_command(
            '0 !LDCAD SNAP_CYL [gender=F] [caps=one] [secs=R 6 4]')
        transform = numpy.array([
            [1,0,0,0],
            [0,1,0,8],
            [0,0,1,0],
            [0,0,0,1],
        ])
        
        resolved_snaps = [StudHole(command, 4, transform)]
        self.snaps = SnapStyleSequence(deduplicate_snaps(resolved_snaps))

custom_brick_shapes = {}
custom_brick_shapes['4624.dat'] = BrickShape_4624
custom_brick_shapes['3641.dat'] = BrickShape_3641
custom_brick_shapes['4084.dat'] = BrickShape_4084
custom_brick_shapes['6014.dat'] = BrickShape_6014
custom_brick_shapes['6015.dat'] = BrickShape_6015
custom_brick_shapes['98138.dat'] = BrickShape_98138

def snaps_and_vertices_from_nested_document(document, transform=None):
    # Due to how snap clearing works, everything in this function
    # must be computed from scratch for each part.  Do not attempt
    if transform is None:
        transform = numpy.eye(4)
    reference_table = document.reference_table
    snaps = []
    vertices = [numpy.zeros((4,0))]
    for command in document.commands:
        if isinstance(command, LDrawImportCommand):
            reference_name = command.reference_name
            reference_document = (
                    reference_table['ldraw'][reference_name])
            reference_transform = transform @ command.transform
            try:
                s, v = snaps_and_vertices_from_nested_document(
                        reference_document, reference_transform)
                snaps.extend(s)
                vertices.append(v)
            except:
                print('Error while importing: %s'%reference_name)
                raise
        elif isinstance(command, LDCadSnapInclCommand):
            reference_name = command.reference_name
            try:
                reference_document = reference_table['shadow'][reference_name]
            except:
                print('Could not find shadow file %s'%
                    reference_name)
                raise
            #reference_transform = transform @ command.transform
            if 'grid' in command.flags:
                reference_transforms = griderate(
                    command.flags['grid'],
                    transform @ command.transform
                )
            else:
                reference_transforms = [transform @ command.transform]
            try:
                for reference_transform in reference_transforms:
                    s, v = snaps_and_vertices_from_nested_document(
                        reference_document, reference_transform)
                    snaps.extend(s)
                    vertices.append(v)
            except:
                print('Error while importing: %s'%reference_name)
                raise
        elif isinstance(command, (LDCadSnapStyleCommand,LDCadSnapClearCommand)):
            new_snaps = Snap.construct_snaps(command, transform)
            snaps.extend(new_snaps)
        elif isinstance(command, LDrawContentCommand):
            vertices.append(transform @ command.vertices)
    
    if not document.shadow:
        reference_name = document.reference_name
        if reference_name in reference_table['shadow']:
            shadow_document = reference_table['shadow'][reference_name]
            try:
                s,v = snaps_and_vertices_from_nested_document(
                    shadow_document, transform)
                snaps.extend(s)
                vertices.append(v)
            except:
                print('Error while importing shadow: %s'%reference_name)
                raise
    
    vertices = numpy.concatenate(vertices, axis=1)
    return snaps, vertices

