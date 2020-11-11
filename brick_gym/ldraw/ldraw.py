import os

import numpy

import brick_gym.config as config

EXTERNAL_REFERENCE_TYPES = ('models', 'parts', 'p')
INTERNAL_REFERENCE_TYPES = ('files',)
ALL_REFERENCE_TYPES = EXTERNAL_REFERENCE_TYPES + INTERNAL_REFERENCE_TYPES
ALL_CONTENT_TYPES = ('comments', 'draw')

LDRAW_FILES = {}
SHADOW_FILES = {}
for path_name, FILES in (('ldraw',LDRAW_FILES), ('shadow_ldraw',SHADOW_FILES)):
    for reference_type in EXTERNAL_REFERENCE_TYPES:
        FILES[reference_type] = {}
        reference_directory = os.path.join(
                config.paths[path_name], reference_type)
        for root, dirs, files in os.walk(reference_directory):
            local_root = root.replace(reference_directory, '').lower()
            if len(local_root) and local_root[0] == '/':
                local_root = local_root[1:]
            FILES[reference_type].update({
                    os.path.join(local_root, f.lower()) :
                    os.path.join(root, f)
                    for f in files})
        
    '''   
    LDRAW_FILES[reference_type] = {}
    reference_ldraw_directory = os.path.join(
            config.paths['ldraw'], reference_type)
    for root, dirs, files in os.walk(reference_ldraw_directory):
        local_root = root.replace(reference_ldraw_directory, '').lower()
        if len(local_root) and local_root[0] == '/':
            local_root = local_root[1:]
        LDRAW_FILES[reference_type].update({
                os.path.join(local_root, f.lower()) :
                os.path.join(reference_ldraw_directory, root, f)
                for f in files})
    '''
    '''
    reference_shadow_directory = os.path.join(
            config_paths['shadow_ldraw'], reference_type)
    for root, dirs, files in os.walk(reference_shadow_directory):
        local_root = root.replace(reference_shadow_directory, '').lower()
        if len(local_root) and local_root[0] == '/':
            local_root = local_root[1:]
        SHADOW_FILES[reference_type].update({
                os.path.join(
    '''

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
        recursion_types = ALL_REFERENCE_TYPES,
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
            nested_files,
            recursion_types,
            content_types)
    
    return ldraw_data

def parse_ldraw(
        ldraw_data,
        nested_files = None,
        recursion_types = ALL_REFERENCE_TYPES,
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
    for reference_type in ALL_REFERENCE_TYPES:
        result[reference_type] = []
    for content_type in content_types:
        result[content_type] = []
    
    for line in ldraw_lines:
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            continue
        
        command, arguments = line_contents
        if command == '0' and 'comments' in content_types:
            result['comments'].append(line)
        elif command == '1':
            color, *matrix_elements, reference_name = arguments.split(None, 13)
            transform = matrix_ldraw_to_list(matrix_elements)
            reference_name = reference_name.strip()
            
            if reference_name in nested_files:
                reference_type = 'files'
                reference_contents = nested_files[reference_name]
            else:
                clean_name = reference_name.lower().replace('\\', '/')
                for reference_type in EXTERNAL_REFERENCE_TYPES:
                    file_name = LDRAW_FILES[reference_type].get(
                            clean_name, None)
                    if file_name is not None:
                        reference_contents = open(file_name)
                        break
                
                else:
                    raise LDrawReferenceNotFoundError(reference_name)
            
            if reference_type in recursion_types:
                reference_data = parse_ldraw(
                        reference_contents,
                        nested_files,
                        recursion_types,
                        content_types)
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
