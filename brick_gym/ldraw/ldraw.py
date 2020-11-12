import os

import numpy

import brick_gym.config as config
import brick_gym.ldraw.paths as ldraw_paths
import brick_gym.ldraw.ldcad as ldcad

'''
EXTERNAL_REFERENCE_TYPES = ('models', 'parts', 'p')
INTERNAL_REFERENCE_TYPES = ('files',)
ALL_REFERENCE_TYPES = EXTERNAL_REFERENCE_TYPES + INTERNAL_REFERENCE_TYPES

LDRAW_FILES = {}
SHADOW_FILES = {}
for path_name, FILES in (('ldraw',LDRAW_FILES), ('shadow_ldraw',SHADOW_FILES)):
    for reference_type in EXTERNAL_REFERENCE_TYPES:
        FILES[reference_type] = {}
        reference_directory = os.path.join(
                config.paths[path_name], reference_type)
        for root, dirs, files in os.walk(reference_directory):
            local_root = paths.clean_path(root.replace(reference_directory, ''))
            if len(local_root) and local_root[0] == '/':
                local_root = local_root[1:]
            FILES[reference_type].update({
                    os.path.join(local_root, f.lower()) :
                    os.path.join(root, f)
                    for f in files})
'''

ALL_CONTENT_TYPES = ('comments', 'draw')
LDRAW_FILES = ldraw_paths.get_ldraw_part_paths(config.paths['ldraw'])

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
        content_types = ALL_CONTENT_TYPES,
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
        content_types = ALL_CONTENT_TYPES,
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
    
    result = {
        'files' : [],
    }
    for reference_type in ldraw_paths.ALL_REFERENCE_TYPES:
        result[reference_type] = []
    for content_type in content_types:
        result[content_type] = []
    
    for line in ldraw_lines:
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            continue
        
        command, arguments = line_contents
        if command == '0':
            if 'comments' in content_types:
                result['comments'].append(line)
        elif command == '1':
            color, *matrix_elements, reference_name = arguments.split(None, 13)
            transform = matrix_ldraw_to_list(matrix_elements)
            reference_name = reference_name.strip()
            
            if reference_name in nested_files:
                reference_type = 'files'
                reference_contents = nested_files[reference_name]
            else:
                #clean_name = reference_name.lower().replace('\\', '/')
                clean_name = ldraw_paths.clean_path(reference_name)
                '''
                for reference_type in EXTERNAL_REFERENCE_TYPES:
                    file_name = LDRAW_FILES[reference_type].get(
                            clean_name, None)
                    if file_name is not None:
                        reference_contents = open(file_name)
                        if include_shadow:
                            shadow_file_name = SHADOW_FILES[reference_type].get(
                                    clean_name, None)
                            if shadow_file_name is not None:
                                shadow_contents = open(shadow_file_name)
                                reference_contents = (
                                        reference_contents.readlines() +
                                        shadow_contents.readlines())
                        break
                
                else:
                    raise LDrawReferenceNotFoundError(reference_name)
                '''
                file_path = LDRAW_FILES.get(
                        clean_name, None)
                if file_path is not None:
                    print(file_path)
                    reference_type = ldraw_paths.get_reference_type(
                            file_path, config.paths['ldraw'])
                    print(reference_type)
                    reference_contents = open(file_path)
                    if include_shadow:
                        pass
                else:
                    raise LDrawReferenceNotFoundError(reference_name)
            
            if reference_type in recursion_types:
                reference_data = parse_ldraw(
                        reference_contents,
                        nested_files = nested_files,
                        recursion_types = recursion_types,
                        content_types = content_types,
                        include_shadow = include_shadow)
            else:
                reference_data = {}
            
            reference_data['name'] = reference_name
            reference_data['color'] = color
            reference_data['transform'] = transform
            reference_data['reference_type'] = reference_type
            result[reference_type].append(reference_data)
            
        elif command in ('2', '3', '4', '5') and 'draw' in content_types:
            result['draw'].append(line)
        
    return result

def import_shadow_contents(reference_type, clean_name):
    shadow_file_name = SHADOW_FILES[reference_type].get(
            clean_name, None)
    if shadow_file_name is None:
        return []
    
    shadow_lines = open(shadow_file_name).readlines()
    resolved_lines = []
    for line in shadow_lines:
        line_parts = line.split(None, 2)
        if len(line_parts) != 2:
            resolved_lines.append(line)
            continue
        
        command, arguments = line_parts
        argument_parts = arguments.split(None, 3)
        if len(argument_parts) != 3:
            resolved_lines.append(line)
            continue
        
        LDCAD, ldcad_command, ldcad_arguments = argument_parts
        if LDCAD != '!LDCAD' or ldcad_command != 'SNAP_INCL':
            resolved_lines.append(line)
            continue
        
        path = None
