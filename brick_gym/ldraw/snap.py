from brick_gym.ldraw.commands import *

class Snap:
    @staticmethod
    def construct_snaps(command, reference_transform, flags):
        
        def construct_snap(command, transform, flags):
            
        
        snap_transform = matrix_ldcad_to_numpy(flags)
        base_transform = numpy.dot(reference_transform, snap_transform)
        snaps = []
        if 'grid' in command.flags:
            grid_transforms = griderate(command.flags['grid'], base_transform)
            for grid_transform in grid_transforms:
                snaps.append(construct_snap(command, grid_transform, flags))
        else:
            snaps.append(construct_snap(command, base_transform, flags))
        
        return snaps
    
    @staticmethod
    def construct_snap(reference_transform, flags):
        

class SnapCylinder(Snap):
    def __init__(self, type_id, transform, gender, 

class SnapClear:
    def __init__(self, type_id):
        self.id = type_id

def griderate(grid, transform):
    if grid is None:
        return [transform]

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

    if center_x:
        x_width = grid_spacing_x * (grid_x-1)
        x_offset = -x_width/2.
    else:
        x_offset = 0.

    if center_z:
        z_width = grid_spacing_z * (grid_z-1)
        z_offset = -z_width/2.
    else:
        z_offset = 0.

    grid_transforms = []
    for x_index in range(grid_x):
        x = x_index * grid_spacing_x + x_offset
        for z_index in range(grid_z):
            z = z_index * grid_spacing_z + z_offset
            translate = numpy.eye(4)
            translate[0,3] = x
            translate[2,3] = z
            grid_transforms.append(numpy.dot(transform, translate))

    return grid_transforms

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
                pass
            else:
                resolved_snap_points = [
                        p for p in resolved_snap_points
                        if p.id != snap_point.id]
    
    return resolved_snap_points
