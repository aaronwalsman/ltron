import os
import ltron.settings as settings

from ltron.ldraw.exceptions import LDrawException

ldraw_subdirectories = ('models', 'parts', 'p')

raise Exception('Deprecated')

def get_reference_name(path):
    '''
    LDraw reference paths are not case-sensitive, and sometimes contain
    backslashes for directory separators.  This function takes either
    an actual system file path or a reference from an LDraw file and produce
    a consistent string that be used to tell if they match.
    
    If an explicit file path was given, this also checks if it is a known ldraw
    part, and if so returns the reference name rather than the file path.
    '''
    reference_name = path.lower().replace('\\', '/')
    
    # check if the path exists under the ldraw or shadow directories,
    # if so, convert it to the name that another file would use to import it
    abs_path = os.path.abspath(reference_name)
    for root_path in settings.paths['ldraw'], settings.paths['shadow']:
        root_path = root_path.lower()
        for reference_subdirectory in ldraw_subdirectories:
            reference_path = os.path.join(root_path, reference_subdirectory)
            if abs_path.startswith(reference_path):
                reference_name = os.path.relpath(abs_path, start=reference_path)
                return reference_name
    
    return reference_name

def get_reference_paths(root_directory):
    reference_paths = {}
    for root, dirs, files in os.walk(root_directory):
        local_root = get_reference_name(
            os.path.relpath(root, start=root_directory))
        if local_root == '.':
            local_root = ''
        reference_paths.update({
                os.path.join(local_root, f.lower()) : os.path.join(root, f)
                for f in files})
    
    return reference_paths

def get_ldraw_reference_paths(
        root_directory,
        reference_subdirectories=ldraw_subdirectories):
    reference_paths = {}
    for reference_subdirectory in reference_subdirectories:
        reference_directory = os.path.join(
                root_directory, reference_subdirectory)
        reference_paths.update(get_reference_paths(reference_directory))
    
    return reference_paths

# each of these maps a reference_name to an absolute file path on disk
LDRAW_FILES = get_ldraw_reference_paths(settings.paths['ldraw'])
SHADOW_FILES = get_ldraw_reference_paths(settings.paths['shadow_ldraw'])

# a set containing all ldraw parts
ldraw_parts_directory = os.path.join(settings.paths['ldraw'], 'parts')
LDRAW_PARTS = set(
        get_reference_name(os.path.join(ldraw_parts_directory, file_name))
        for file_name in os.listdir(ldraw_parts_directory)
        if os.path.splitext(file_name)[-1] in ('.dat', '.DAT'))

class LDrawMissingPath(LDrawException):
    pass

def resolve_path(file_path, FILES, allow_existing = True):
    expanded_path = os.path.expanduser(file_path)
    if allow_existing and os.path.exists(expanded_path):
        return expanded_path
    
    reference_name = get_reference_name(file_path)
    try:
        return FILES[reference_name]
    except KeyError:
        raise LDrawMissingPath('Cannot resolve path: %s'%file_path)

def resolve_ldraw_path(file_path):
    return resolve_path(file_path, LDRAW_FILES, allow_existing = True)

def resolve_shadow_path(file_path):
    return resolve_path(file_path, SHADOW_FILES, allow_existing = False)
