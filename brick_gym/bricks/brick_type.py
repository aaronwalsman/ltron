import collections

from brick_gym.ldraw.commands import *
from brick_gym.ldraw.documents import *
from brick_gym.bricks.snap import Snap, SnapStyle, SnapClear

class BrickLibrary(collections.abc.MutableMapping):
    def __init__(self, brick_types=None):
        if brick_types is None:
            brick_types = {}
        self.brick_types = brick_types
    
    def import_document(self, document):
        new_types = []
        '''
        for reference_name, reference_document in (
                    document.reference_table['ldraw'].items()):
            if reference_name not in self:
                if reference_name in ldraw_paths.LDRAW_PARTS:
                    new_type = BrickType(reference_document)
                    self[reference_name] = new_type
                    new_types.append(new_type)
        '''
        for command in document.commands:
            if isinstance(command, LDrawImportCommand):
                reference_name = command.reference_name
                reference_document = (
                        document.reference_table['ldraw'][reference_name])
                if isinstance(reference_document, LDrawDAT):
                    if reference_name in ldraw_paths.LDRAW_PARTS:
                        new_type = BrickType(reference_document)
                        self[reference_name] = new_type
                        new_types.append(new_type)
                elif isinstance(reference_document, (
                        LDrawMPDMainFile,
                        LDrawMPDInternalFile,
                        LDrawLDR)):
                    new_types.extend(self.import_document(
                            reference_document))
                    
        return new_types
    
    def __getitem__(self, key):
        return self.brick_types[str(key)]
    
    def __setitem__(self, key, value):
        assert isinstance(value, BrickType)
        self.brick_types[str(key)] = value
    
    def __delitem__(self, key):
        del(self.brick_types[str(key)])
    
    def __iter__(self):
        return iter(self.brick_types)
    
    def __len__(self):
        return len(self.brick_types)

class BrickType:
    def __init__(self, document):
        self.reference_name = document.reference_name
        self.mesh_name = self.reference_name.replace('.dat', '')
        self.document = document
        self.construct_snaps()
    
    def __str__(self):
        return self.reference_name
    
    def renderpy_mesh_args(self):
        mesh_entry = {
            'mesh_asset' : self.mesh_name,
            'scale' : 1.0,
            'create_uvs' : True
        }
        return mesh_entry
    
    def construct_snaps(self):
        def snaps_from_nested_document(document, transform=None):
            # Due to how snap clearing works, everything in this function
            # must be computed from scratch for each part.  Do not attempt
            # cache intermediate results for sub-files.
            if transform is None:
                transform = numpy.eye(4)
            reference_table = document.reference_table
            snaps = []
            for command in document.commands:
                #if isinstance(
                #        command, (LDrawImportCommand, LDCadSnapInclCommand)):
                if isinstance(command, LDrawImportCommand):
                    reference_name = command.reference_name
                    reference_document = (
                            reference_table['ldraw'][reference_name])
                    reference_transform = numpy.dot(
                            transform, command.transform)
                    try:
                        snaps.extend(snaps_from_nested_document(
                                reference_document, reference_transform))
                    except:
                        print('Error while importing: %s'%reference_name)
                        raise
                elif isinstance(command, LDCadSnapInclCommand):
                    reference_name = command.reference_name
                    reference_document = (
                            reference_table['shadow'][reference_name])
                    reference_transform = numpy.dot(
                            transform, command.transform)
                    try:
                        snaps.extend(snaps_from_nested_document(
                                reference_document, reference_transform))
                    except:
                        print('Error while importing: %s'%reference_name)
                        raise
                elif isinstance(
                        command,
                        (LDCadSnapStyleCommand, LDCadSnapClearCommand)):
                    new_snaps = Snap.construct_snaps(command, transform)
                    snaps.extend(new_snaps)
            
            if not document.shadow:
                reference_name = document.reference_name
                if reference_name in reference_table['shadow']:
                    shadow_document = reference_table['shadow'][reference_name]
                    try:
                        snaps.extend(snaps_from_nested_document(
                                shadow_document, transform))
                    except:
                        print('Error while importing shadow: %s'%reference_name)
                        raise
            
            return snaps
        
        try:
            snaps = snaps_from_nested_document(self.document)
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
        
        self.snaps = resolved_snaps
