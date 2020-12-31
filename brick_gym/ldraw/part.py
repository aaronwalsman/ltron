from brick_gym.ldraw.documents import LDrawDocument
from brick_gym.ldraw.commands import *

class BrickType:
    def __init__(self, document):
        self.document = document
        self.compute_snaps()
        
    def compute_snaps()
        def snaps_from_nested_document(document, transform=None):
            # Due to how snap clearing work, everything in this function
            # must be computed from scratch for each part.  Do not attempt
            # cache intermediate results for sub-files.
            if transform is None:
                transform = numpy.eye(4)
            reference_table = document.reference_table
            snaps = []
            for command in self.document.commands:
                if isinstance(command, LDrawImportCommand):
                    reference_name = command.clean_reference_name
                    reference_document = (
                            reference_table['ldraw'][reference_name])
                    reference_transform = numpy.dot(
                            transform, command.transform)
                    snaps.extend(snaps_from_nested_document(
                            reference_document, reference_transform)
                elif isinstance(command, LDCadSnapInclCommand):
                    reference_name  = command.clean_reference_name
                    reference_document = (
                            reference_table['shadow'][reference_name])
                    reference_transform = numpy.dot(
                            transform, command.transform)
                    snaps.extend(snaps_from_nested_document(
                            reference_document, reference_transform))
                elif isinstance(command, LDCadSnapStyleCommand):
                    snaps.extend(Snap(...))
                elif isinstance(command, LDCadSnapClearCommand):
                    snaps.append(SnapClear(...))
            
            if not document.shadow:
                clean_name = document.clean_name
                if clean_name in reference_table['shadow']:
                    shadow_document = reference_table['shadow'][clean_name]
                    snaps.extend(snaps_from_nested_document(
                            shadow_document, transform))
            
            return snaps
        
        snaps = snaps_from_nested_document(self.document)
        
        resolved_snaps = []
        for snap in snaps:
            if isinstance(snap, Snap):
                resolved_snap_points.append(snap)
            elif isinstance(snap, SnapClear):
                if snap.id == '':
                    resolved_snap_points.clear()
                else:
                    resolved_snap_points = [
                            p for p in resolved_snap_points
                            if p.id != snap.id]
        
        self.snaps = resolved_snaps

class BrickInstance:
    def __init__(self, instance_id, part_type, color, transform):
        self.instance_id = instance_id
        self.part_type = part_type
        self.color = color
        self.transform = transform
