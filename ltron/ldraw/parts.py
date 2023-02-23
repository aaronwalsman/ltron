import io
import os
import zipfile
import json

from splendor.assets import AssetLibrary

from ltron.exceptions import LtronException
from ltron.home import get_ltron_home
import ltron.settings as settings
import ltron.constants as constants
from ltron.ldraw.exceptions import LtronException

def get_reference_name(path):
    reference_name = path.lower().replace('\\', '/')
    return reference_name

blacklist_path = os.path.join(get_ltron_home(), 'blacklist.json')
with open(blacklist_path, 'r') as f:
    blacklist_data = json.load(f)

LDRAW_BLACKLIST_ALL = set(blacklist_data['all'])

ldraw_zip_path = os.path.join(get_ltron_home(), 'complete.zip')
ldraw_zip = zipfile.ZipFile(ldraw_zip_path, 'r')

ltron_splendor_assets = AssetLibrary(asset_packages='ltron_assets')

LDRAW_PARTS = set()
LDRAW_PARTS_S = set()
LDRAW_P = set()
LDRAW_MODELS = set()
LDRAW_PATHS = {}
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
        LDRAW_PATHS[reference_name] = zip_path

LDRAW_PARTS_WITHOUT_MESHES = set(
    part for part in LDRAW_PARTS
    if part.replace('.dat', '.obj').replace('.ldr', '.obj') not in
    ltron_splendor_assets['meshes']
)

LDRAW_BLACKLIST_ALL |= LDRAW_PARTS_WITHOUT_MESHES

def regenerate_shape_class_labels():
    shape_class_labels = {
        part : i+1 for i, part in
        enumerate(sorted(LDRAW_PARTS - LDRAW_BLACKLIST_ALL))
    }
    class_labels = json.load(open(settings.PATHS['class_labels']))
    class_labels['shape'] = shape_class_labels
    with open(settings.PATHS['class_labels'], 'w') as f:
        json.dump(class_labels, f, indent=2)
    constants.reload_class_labels()

shadow_zip_path = os.path.join(settings.PATHS['ldcad'], 'seeds', 'shadow.sf')
shadow_zip = zipfile.ZipFile(shadow_zip_path, 'r')
offlib_csl_path = 'offLib/offLibShadow.csl'
offlib_csl = zipfile.ZipFile(
    io.BytesIO(shadow_zip.open(offlib_csl_path).read()))
SHADOW_PATHS = {}
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
        SHADOW_PATHS[reference_name] = zip_path

class LtronReferenceException(LtronException):
    pass

def get_reference_path(path, shadow):
    if shadow:
        if path in SHADOW_PATHS:
            return SHADOW_PATHS[path]
        else:
            raise LtronReferenceException('Shadow path not found: %s'%path)
    else:
        if path in LDRAW_PATHS:
            return LDRAW_PATHS[path]
        else:
            path = os.path.expanduser(path)
            return path
            # don't require existence because it may be in a tar file
            #if os.path.exists(path):
            #    return path
            #else:
            #    raise LtronReferenceException('Ldraw path not found: %s'%path)

