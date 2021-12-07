import io
import os
import zipfile

from ltron.exceptions import LtronException
from ltron.home import get_ltron_home
import ltron.settings as settings
from ltron.ldraw.exceptions import LtronException

'''
external requirements:
- LDRAW_PARTS:
    a set of all parts that are loadable as a BrickType
- get_reference_name:
    convert a file path into a consistent reference name
- resolve_shadow_path:
    Given a reference name, return a path to the shadow file.
    Currently used by ldraw/documents.py to find paths to read.
    Shouldn't be necessary when reading from zip file.
    Be careful here though: file names in the zip may be capitalized.
- resolve_ldraw_path:
    Given a reference name, return a path to the ldraw file.
    Currently used by ldraw/documents.py to find paths to read.
    Shouldn't be necessary when reading from zip file.
    Be careful here though: file names in the zip may be capitalized.
- SHADOW_FILES:
    A mapping from reference names to file paths.
    Currently used by ldraw/documents.py to test if a reference path has a
    corresponding shadow file.
    
    
'''

def get_reference_name(path):
    reference_name = path.lower().replace('\\', '/')
    return reference_name

ldraw_zip_path = os.path.join(get_ltron_home(), 'complete.zip')
ldraw_zip = zipfile.ZipFile(ldraw_zip_path, 'r')
LDRAW_PARTS = set()
LDRAW_PARTS_S = set()
LDRAW_P = set()
LDRAW_MODELS = set()
LDRAW = {}
for info in ldraw_zip.infolist():
    zip_path = info.filename
    zip_folder, zip_filename = os.path.split(zip_path)
    extension = os.path.splitext(zip_filename)[1].lower()
    if extension == '.dat' or extension == '.ldr':
        # only files that are directly in the ldraw/parts directory
        # (not including subfolders) belong in LDRAW_PARTS
        if zip_folder == 'ldraw/parts':
            relpath = zip_path.replace('ldraw/parts/', '', 1)
            partition = LDRAW_PARTS
        elif zip_path.startswith('ldraw/parts/'):
            relpath = zip_path.replace('ldraw/parts/', '', 1)
            partition = LDRAW_PARTS_S
        elif zip_path.startswith('ldraw/p/'):
            relpath = zip_path.replace('ldraw/p/', '', 1)
            partition = LDRAW_P
        elif zip_path.startswith('ldraw/models/'):
            relpath = zip_path.replace('ldraw/models/', '', 1)
            partition = LDRAW_MODELS
        reference_name = get_reference_name(relpath)
        partition.add(reference_name)
        LDRAW[reference_name] = zip_path

shadow_zip_path = os.path.join(settings.paths['ldcad'], 'seeds', 'shadow.sf')
shadow_zip = zipfile.ZipFile(shadow_zip_path, 'r')
offlib_csl_path = 'offLib/offLibShadow.csl'
offlib_csl = zipfile.ZipFile(
    io.BytesIO(shadow_zip.open(offlib_csl_path).read()))
SHADOW = {}
for info in offlib_csl.infolist():
    zip_path = info.filename
    zip_folder, zip_filename = os.path.split(zip_path)
    extension = os.path.splitext(zip_filename)[1].lower()
    if extension == '.dat':
        if zip_path.startswith('parts/'):
            relpath = zip_path.replace('parts/', '', 1)
        elif zip_path.startswith('p/'):
            relpath = zip_path.replace('p/', '', 1)
        reference_name = get_reference_name(relpath)
        SHADOW[reference_name] = zip_path

class LtronReferenceException(LtronException):
    pass

def get_reference_path(path, shadow):
    if shadow:
        if path in SHADOW:
            return SHADOW[path]
        else:
            raise LtronReferenceException('Shadow path not found: %s'%path)
    else:
        if path in LDRAW:
            return LDRAW[path]
        else:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                return path
            else:
                raise LtronReferenceException('Ldraw path not found: %s'%path)

