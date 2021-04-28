import re

import ltron.settings as settings
import ltron.ldraw.paths as ldraw_paths

SHADOW_FILES = ldraw_paths.get_ldraw_part_paths(settings.paths['shadow_ldraw'])

def get_ldcad_flags(arguments):
    ldcad_command, arguments = arguments.split(None, 1)
    
    flag_tokens = re.findall('\[[^\]]\]', arguments)
    flags = {}
    for flag_token in flag_tokens:
        flag, value = flag_token.split()
        flags[flag.strip()] = value.strip()
    
    return ldcad_command, flags

def parse_ldcad_line(line):
    line_parts = line.split(None, 2)
    if len(line_parts) != 2:
        return False, None, []
    
    command, arguments = line_parts
    if command != '0':
        return False, None, []
    
    argument_parts = arguments.split(None, 2)
    if len(argument_parts) != 2:
        return False, None, []

    LDCAD, arguments = argument_parts
    if LDCAD != '!LDCAD':
        return False, None, []

    ldcad_command, ldcad_arguments = get_ldcad_flags(arguments)
    return True, ldcad_command, ldcad_arguments

def griderate(grid, transform):
    if grid is None:
        return [transform]
    
    grid_parts = grid.split()
    if grid_parts[0] == 'C':
        center_x = True
        grid_parts = grid_parts[1:]
    grid_x = grid_parts[0]
    if grid_parts[1] == 'C':
        center_z = True
        grid_parts = grid_parts[0] + grid_parts[2:]
    grid_z = grid_parts[1]
    grid_spacing_x = grid_parts[2]
    grid_spacing_z = grid_parts[3]
    
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
    for x_index in grid_x:
        x = x_index * x_spacing + x_offset
        for z_index in grid_z:
            z = z_index * z_spacing + z_offset
            translate = numpy.eye(4)
            translate[0,3] = x
            translate[2,3] = z
            grid_transforms.append(numpy.dot(transform, translate))
    
    return grid_transforms

def import_shadow(name):
    clean_name = ldraw_paths.clean_path(name)
    shadow_file_name = SHADOW_FILES.get(clean_name, None)
    print(clean_name, shadow_file_name)
    if shadow_file_name is None:
        return []

    shadow_lines = open(shadow_file_name).readlines()
    resolved_lines = []
    for line in shadow_lines:
        is_ldcad, ldcad_command, ldcad_arguments = parse_ldcad_line(line)
        
        if is_ldcad and ldcad_command == 'SNAP_INCL':
            file_reference = ldcad_arguments['ref']
            referenced_contents = import_shadow_contents(file_reference)
        else:
            referenced_contents = []
        
        resolved_lines.append((line, referenced_contents))
    
    return resolved_lines
        
        
'''
def get_connection_points(
        ldraw_text,
        connection_points = None,
        connection_ids = (),
        transform = None):
    
    if connection_points is None:
        connection_points = []
    
    if transform is None:
        transform = numpy.eye(4)
    
    lines = ldraw_text.splitlines()
    for line in lines:
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            continue
        
        command, arguments = line_contents
        
        if command == '1':
            color, *matrix_elements, file_name = arguments.split(None, 13)
            ref_transform = ldraw.matrix_ldraw_to_numpy(matrix_elements)
            transform = numpy.dot(transform, ref_transform)
            file_path = ldraw.resolve_ldraw_part_filepath(file_name)
            with open(file_path, 'r') as f:
                referenced_ldraw_text = f.read()
            get_connection_points(
                    referenced_ldraw_text,
                    connection_points,
                    connection_ids,
                    transform)
        
        elif command == '0':
            ldcad_arguments = arguments.split(None, 1)
            if len(ldcad_arguments) != 2 or ldcad_arguments[0] != '!LDCAD':
                continue
            
            subcommand, flags = get_ldcad_flags(ldcad_arguments)
            try:
                ldcad_fn = globals()[subcommand]
            except KeyError:
                pass
            ldcad_fn(connection_points,
                    connection_ids,
                    transform,
                    **flags)
    
    return connection_points
'''

def SNAP_CLEAR(connection_points, id=None):
    if id is None:
        connection_points.clear()
    elif id in connection_points:
        connection_points[:] = [(ids, connections)
                for (ids, connections) in connection_points
                if id not in ids]

def SNAP_INCL(connection_points,
        transform,
        connection_ids,
        id=None,
        pos='0 0 0',
        ori='1 0 0 0 1 0 0 0 1',
        scale='1 1 1',
        ref=None,
        grid=None):
    
    assert ref is not None
    
    ref_path = ldraw.resolve_part_path(ref)
    with open(ref_path, 'r') as f:
        ref_text = f.read()
    
    # update ids
    if id is not None:
        connection_ids.append(id)
    
    # update transform
    pos = pos.split()
    ori = ori.split()
    ref_transform = ldcad.matrix_ldraw_to_numpy(pos + ori)
    scale_transform = numpy.eye(4)
    scale = [float(xyz) for xyz in scale.split()]
    scale_transform[0,0] = scale[0]
    scale_transform[1,1] = scale[1]
    scale_transform[2,2] = scale[2]
    transform = numpy.dot(numpy.dot(transform, ref_transform), scale_transform)
    
    for grid_transform in griderate(grid, grid_transform):
        get_connection_points(
                ref_text,
                connection_points = connection_points,
                connection_ids = connection_ids,
                transform = grid_transform)
