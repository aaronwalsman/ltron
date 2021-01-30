import collections

try:
    import renderpy.masks as masks
    renderpy_available = True
except:
    renderpy_available = False

from brick_gym.ldraw.commands import *
from brick_gym.ldraw.documents import *
from brick_gym.bricks.snap import Snap, SnapStyle, SnapClear

class BrickInstanceTable(collections.abc.MutableMapping):
    def __init__(self, library, instances = None):
        self.library = library
        if instances is None:
            instances = {}
        self.instances = instances
        self.next_instance_id = 1
    
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
                        if reference_name in ldraw_paths.LDRAW_PARTS:
                            new_instance = BrickInstance(
                                    self.next_instance_id,
                                    self.library[reference_name],
                                    reference_color,
                                    reference_transform)
                            self[self.next_instance_id] = new_instance
                            new_instances.append(new_instance)
                            self.next_instance_id += 1
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
        self.instances[int(key)] = value
    
    def __delitem__(self, key):
        del(self.instances[int(key)])
    
    def __iter__(self):
        return iter(self.instances)
    
    def __len__(self):
        return len(self.instances)
        
class BrickInstance:
    def __init__(self, instance_id, brick_type, color, transform):
        self.instance_id = instance_id
        self.instance_name = str(self.instance_id)
        self.brick_type = brick_type
        self.color = color
        self.transform = transform
    
    def __int__(self):
        return self.instance_id
    
    def __str__(self):
        return self.instance_name
    
    def get_snaps(self):
        snaps = []
        for snap in self.brick_type.snaps:
            snaps.append(snap.transformed_copy(self.transform))
        return snaps
    
    def renderpy_instance_args(self):
        instance_args = {
            'mesh_name' : self.brick_type.mesh_name,
            'material_name' : str(self.color),
            'transform' : self.transform,
        }
        if renderpy_available:
            instance_args['mask_color'] = masks.color_index_to_byte(
                    self.instance_id)/255.
        return instance_args