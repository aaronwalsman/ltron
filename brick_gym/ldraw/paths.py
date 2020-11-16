import os
import brick_gym.config as config

from brick_gym.ldraw.exceptions import *

EXTERNAL_REFERENCE_TYPES = ('models', 'parts', 'p')
INTERNAL_REFERENCE_TYPES = ('files',)
ALL_REFERENCE_TYPES = EXTERNAL_REFERENCE_TYPES + INTERNAL_REFERENCE_TYPES

def clean_name(path):
    '''
    LDraw reference paths are not case-insensitive, and sometimes contain
    backslashes for directory separators.  This function takes either
    an actual system file path or a reference from an LDraw file and produce
    a consistent string that be used to tell if they match.
    '''
    return path.lower().replace('\\', '/')

def get_reference_type(file_path, root_path):
    relative_path = os.path.relpath(file_path, root_path)
    return relative_path.split(os.sep)[0]

def get_part_paths(root_directory):
    part_paths = {}
    for root, dirs, files in os.walk(root_directory):
        local_root = clean_name(os.path.relpath(root, start = root_directory))
        if local_root == '.':
            local_root = ''
        part_paths.update({
                os.path.join(local_root, f.lower()) : os.path.join(root, f)
                for f in files})
    
    return part_paths

def get_ldraw_part_paths(
        root_directory,
        reference_types = EXTERNAL_REFERENCE_TYPES):
    part_paths = {}
    for reference_type in reference_types:
        reference_directory = os.path.join(root_directory, reference_type)
        part_paths.update(get_part_paths(reference_directory))
    
    return part_paths

LDRAW_FILES = get_ldraw_part_paths(config.paths['ldraw'])
SHADOW_FILES = get_ldraw_part_paths(config.paths['shadow_ldraw'])

class LDrawMissingPath(LDrawException):
    pass

def resolve_path(file_path, FILES, allow_existing = True):
    if allow_existing and os.path.exists(file_path):
        return file_path
    
    clean_file_path = clean_name(file_path)
    try:
        return FILES[clean_file_path]
    except KeyError:
        raise LDrawMissingPath('Cannot resolve path: %s'%file_path)

def resolve_ldraw_path(file_path):
    return resolve_path(file_path, LDRAW_FILES, allow_existing = True)

def resolve_shadow_path(file_path):
    return resolve_path(file_path, SHADOW_FILES, allow_existing = False)
