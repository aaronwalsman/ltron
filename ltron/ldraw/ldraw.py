import os

import numpy

import ltron.settings as settings
import ltron.ldraw.paths as ldraw_paths
import ltron.ldraw.ldcad as ldcad

ALL_COMMANDS = ('0', '1', '2', '3', '4', '5')
LDRAW_FILES = ldraw_paths.get_ldraw_part_paths(settings.paths['ldraw'])

def matrix_ldraw_to_list(elements):
    assert len(elements) == 12
    (x, y, z,
     xx, xy, xz,
     yx, yy, yz, 
     zx, zy, zz) = elements
    x, y, z = float(x), float(y), float(z)
    xx, xy, xz = float(xx), float(xy), float(xz)
    yx, yy, yz = float(yx), float(yy), float(yz)
    zx, zy, zz = float(zx), float(zy), float(zz)
    return [[xx, xy, xz, x],
            [yx, yy, yz, y],
            [zx, zy, zz, z],
            [ 0,  0,  0, 1]]

class LDrawReferenceNotFoundError(Exception):
    pass

def parse_mpd(
        ldraw_data,
        recursion_types = ldraw_paths.ALL_REFERENCE_TYPES,
        content_types = ALL_COMMANDS,
        include_shadow = False):
    if isinstance(ldraw_data, str):
        ldraw_lines = ldraw_data.splitlines()
    else:
        try:
            ldraw_lines = ldraw_data.readlines()
        except AttributeError:
            ldraw_lines = ldraw_data
    
    # first find all internal files
    nested_files = {}
    for i, line in enumerate(ldraw_lines):
        if i == 0 or line[:7] == '0 FILE ':
            if i == 0:
                file_name = 'main'
            else:
                file_name = line[7:].strip()
            current_file = []
            if not len(nested_files):
                main_file = current_file
            nested_files[file_name] = current_file
        
        current_file.append(line)
    
    ldraw_data = parse_ldraw(
            main_file,
            nested_files = nested_files,
            recursion_types = recursion_types,
            content_types = content_types,
            include_shadow = include_shadow)
    
    return ldraw_data

def parse_ldraw(
        ldraw_data,
        nested_files = None,
        recursion_types = ldraw_paths.ALL_REFERENCE_TYPES,
        content_types = ALL_COMMANDS,
        include_shadow = False):
    if isinstance(ldraw_data, str):
        ldraw_lines = ldraw_data.splitlines()
    else:
        try:
            ldraw_lines = ldraw_data.readlines()
        except AttributeError:
            ldraw_lines = ldraw_data
    
    if nested_files is None:
        nested_files = {}
    
    '''
    result = {
        'files' : [],
    }
    for reference_type in ldraw_paths.ALL_REFERENCE_TYPES:
        result[reference_type] = []
    for content_type in content_types:
        result[content_type] = []
    '''
    
    resolved_commands = []
    
    for line in ldraw_lines:
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            continue
        
        command, arguments = line_contents
        if command not in content_types:
            continue
        
        reference_lines = []
        if command == '1':
            #color, *matrix_elements, reference_name = arguments.split(None, 13)
            #transform = matrix_ldraw_to_list(matrix_elements)
            #reference_name = reference_name.strip()
            
            if reference_name in nested_files:
                reference_type = 'files'
                reference_contents = nested_files[reference_name]
            else:
                clean_name = ldraw_paths.clean_path(reference_name)
                file_path = LDRAW_FILES.get(
                        clean_name, None)
                if file_path is not None:
                    reference_type = ldraw_paths.get_reference_type(
                            file_path, settings.paths['ldraw'])
                    reference_contents = open(file_path)
                else:
                    raise LDrawReferenceNotFoundError(reference_name)
            
            if reference_type in recursion_types:
                reference_lines.extend(parse_ldraw(
                        reference_contents,
                        nested_files = nested_files,
                        recursion_types = recursion_types,
                        content_types = content_types,
                        include_shadow = include_shadow))
            if include_shadow:
                reference_lines.extend(ldcad.import_shadow(reference_name))
            
            '''
            reference_data['name'] = reference_name
            reference_data['color'] = color
            reference_data['transform'] = transform
            reference_data['reference_type'] = reference_type
            result[reference_type].append(reference_data)
            '''
            
        resolved_commands.append((line, reference_lines))
        
    return resolved_commands
