import collections

import numpy

try:
    import splendor.masks as masks
    splendor_available = True
except:
    splendor_available = False

#import ltron.ldraw.paths as ldraw_paths
from ltron.ldraw.parts import LDRAW_PARTS
from ltron.ldraw.commands import *
from ltron.ldraw.documents import *
from ltron.bricks.snap import Snap, SnapStyle, SnapClear, SnapInstanceSequence

class BrickInstanceTable(collections.abc.MutableMapping):
    def __init__(self, shape_library, color_library, instances = None):
        self.shape_library = shape_library
        self.color_library = color_library
        if instances is None:
            instances = {}
        self.instances = instances
        self.next_instance_id = 1
    
    def add_instance(
        self,
        brick_name,
        brick_color,
        brick_transform,
        mask_color=None,
        instance_id=None,
    ):
        if instance_id is None:
            instance_id = self.next_instance_id
            self.next_instance_id += 1
        else:
            assert instance_id not in self
            if instance_id >= self.next_instance_id:
                self.next_instance_id = instance_id + 1
            
        new_instance = BrickInstance(
                instance_id,
                self.shape_library[brick_name],
                self.color_library[brick_color],
                brick_transform,
                mask_color=mask_color)
        self[instance_id] = new_instance
        return new_instance
    
    def import_document(self, document, transform=None, color=None):
        new_instances = []
        try:
            if transform is None:
                transform = numpy.eye(4)
            
            for command in document.commands:
                if isinstance(command, LDrawImportCommand):
                    reference_name = command.reference_name
                    reference_document = (
                            document.reference_table['ldraw'][reference_name])
                    reference_transform = numpy.dot(
                            transform, command.transform)
                    reference_color = command.color
                    if isinstance(reference_document, LDrawDAT):
                        if reference_name in LDRAW_PARTS:
                            new_instance = self.add_instance(
                                    reference_name,
                                    reference_color,
                                    reference_transform)
                            new_instances.append(new_instance)
                    elif isinstance(reference_document, (
                            LDrawMPDMainFile,
                            LDrawMPDInternalFile,
                            LDrawLDR)):
                        new_instances.extend(self.import_document(
                                reference_document,
                                reference_transform,
                                reference_color))
                
        except:
            print('Error while importing instances from %s'%
                    document.reference_name)
            raise
        
        return new_instances
    
    def clear(self):
        super(BrickInstanceTable, self).clear()
        self.next_instance_id = 1
    
    def __getitem__(self, key):
        return self.instances[int(key)]
    
    def __setitem__(self, key, value):
        assert isinstance(value, BrickInstance)
        assert int(key) == int(value)
        self.instances[int(key)] = value
    
    def __delitem__(self, key):
        del(self.instances[int(key)])
    
    def __iter__(self):
        return iter(self.instances)
    
    def __len__(self):
        return len(self.instances)
        
class BrickInstance:
    def __init__(self,
            instance_id, brick_shape, color, transform, mask_color=None):
        self.instance_id = instance_id
        self.instance_name = str(self.instance_id)
        self.brick_shape = brick_shape
        self.color = color
        self.transform = transform
        self.mask_color = mask_color
        
        self.snaps = SnapInstanceSequence(self.brick_shape.snaps, self)
    
    def clone(self):
        return BrickInstance(
            self.instance_id,
            self.brick_shape,
            self.color,
            numpy.copy(self.transform),
            self.mask_color,
        )
    
    def __int__(self):
        return self.instance_id
    
    def __str__(self):
        return self.instance_name
    
    def get_upright_snaps(self):
        return [snap for snap in self.snaps if snap.is_upright()]
    
    def splendor_instance_args(self):
        instance_args = {
            'mesh_name' : self.brick_shape.mesh_name,
            'material_name' : self.color.color_name,
            'transform' : self.transform,
        }
        if splendor_available:
            if self.mask_color is None:
                instance_args['mask_color'] = masks.color_index_to_byte(
                        self.instance_id)/255.
            else:
                instance_args['mask_color'] = self.mask_color
        return instance_args
    
    def bbox_vertices(self):
        return self.transform @ self.brick_shape.bbox_vertices
