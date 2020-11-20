import os

import numpy

import brick_gym.config as config

EXTERNAL_REFERENCE_TYPES = ('models', 'parts', 'p')
INTERNAL_REFERENCE_TYPES = ('files',)
ALL_REFERENCE_TYPES = EXTERNAL_REFERENCE_TYPES + INTERNAL_REFERENCE_TYPES
ALL_CONTENT_TYPES = ('comments', 'draw')

LDRAW_FILES = {}
SHADOW_FILES = {}
for reference_type in EXTERNAL_REFERENCE_TYPES:
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
ldraw_directory = config.paths['ldraw']
reference_directories = {
    reference_type : os.path.join(ldraw_directory, reference_type)
    for reference_type in EXTERNAL_REFERENCE_TYPES
}

reference_files = {}
for reference_type, directory in reference_directories.items():
    reference_files[reference_type] = {}
    for root, dirs, files in os.walk(directory):
        local_root = root.replace(directory, '').lower()
        if len(local_root) and local_root[0] == '/':
            local_root = local_root[1:]
        reference_files[reference_type].update({
                os.path.join(local_root, f.lower()) :
                os.path.join(directory, root, f) for f in files})

shadow_ldraw_directory = config.paths['shadow_ldraw']
shadow_reference_directories = {}
for reference_type in reference_directories:
    shadow_reference_directory = os.path.join(
            shadow_ldraw_directory, reference_type)
    if os.path.isdir(shadow_reference_directory):
        shadow_reference_directories[reference_type] = (
            shadow_reference_directory)
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
        #reference_types = ALL_REFERENCE_TYPES,
        recursion_types = ALL_REFERENCE_TYPES,
        content_types = ALL_CONTENT_TYPES):
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
            #reference_types,
            recursion_types,
            content_types)
    
    return ldraw_data

def parse_ldraw(
        ldraw_data,
        nested_files = None,
        #reference_types = ('models', 'parts', 'p'),
        recursion_types = ALL_REFERENCE_TYPES,
        content_types = ALL_CONTENT_TYPES):
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
    for reference_type in ALL_REFERENCE_TYPES: #reference_types:
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
                recursion_data = nested_files[reference_name]
                '''
                reference_data = parse_ldraw(
                        nested_files[reference_name],
                        nested_files,
                        #reference_types,
                        recursion_types,
                        content_types)
                '''
            else:
                for reference_type in EXTERNAL_REFERENCE_TYPES:#reference_types:
                    #directory = reference_directories[reference_type]
                    try:
                        clean_name = reference_name.lower().replace('\\', '/')
                        file_name = (
                            LDRAW_FILES[reference_type][clean_name])
                    except KeyError:
                        continue
                    
                    '''
                    if reference_type in recursion_types:
                        file_path = os.path.join(directory, file_name)
                        with open(file_path, 'r') as f:
                            reference_data = parse_ldraw(
                                    f,
                                    nested_files,
                                    #reference_types,
                                    recursion_types,
                                    content_types)
                    else:
                        reference_data = {}
                    '''
                    #recursion_path = os.path.join(directory, file_name)
                    print(file_name)
                    recursion_data = open(file_name)
                    
                    break
                
                else:
                    raise LDrawReferenceNotFoundError(reference_name)
            
            if reference_type in recursion_types:
                reference_data = parse_ldraw(
                        recursion_data,
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
