import os
import shutil
import requests
import zipfile
import bz2
import tarfile

from splendor.home import get_splendor_home
from splendor.assets import install_assets
from splendor.download import download, agree_to_zip_licenses

import ltron.settings as settings
from ltron.home import get_ltron_home, make_ltron_home
from ltron.license import ldcad_license_text

ltron_home = get_ltron_home()

def install_ldraw(overwrite=False):
    print('='*80)
    print('Installing LDraw')
    make_ltron_home()
    print('-'*80)
    complete_zip_path = os.path.join(ltron_home, 'complete.zip')
    downloaded_path = download(
        settings.urls['ldraw'],
        complete_zip_path,
        overwrite=overwrite,
    )
    
    if downloaded_path is not None:
        print('-'*80)
        print('Checking for Licenses')
        if agree_to_zip_licenses(downloaded_path):
            print('Extracting Contents To: %s'%ltron_home)
            with zipfile.ZipFile(downloaded_path, 'r') as z:
                z.extractall(ltron_home)
        else:
            print('Must agree to all licensing.  Aborting LDraw install.')

def ldcad_license_agreement():
    print('LDCad is a necessary component of LTRON '
        'and is provided under the following license:')
    print(ldcad_license_text)
    print('Agree? (y/n)')
    yn = input()
    return yn in 'yY'

def install_ldcad(overwrite=True):
    print('='*80)
    print('Installing LDCad')
    make_ltron_home()
    print('-'*80)
    
    # download
    ldcad_url = settings.urls['ldcad']
    ldcad_bz2_filename = ldcad_url.split('/')[-1]
    ldcad_bz2_path = os.path.join(ltron_home, ldcad_bz2_filename)
    download(ldcad_url, ldcad_bz2_path, overwrite=overwrite)
    
    print('-'*80)
    if not ldcad_license_agreement():
        print('Must agree to all licensing.  Aborting LDCad intall.')
        return False
    
    # unbz2
    ldcad_tar_path = ldcad_bz2_path.replace('.bz2', '')
    print('-'*80)
    print('Extracting bz2 archive to: %s'%ldcad_tar_path)
    with open(ldcad_bz2_path, 'rb') as f_in:
        data = bz2.decompress(f_in.read())
        with open(ldcad_tar_path, 'wb') as f_out:
            f_out.write(data)
    
    # untar
    ldcad_path = ldcad_tar_path.replace('.tar', '')
    print('-'*80)
    print('Extracting tar archive to: %s'%ldcad_path)
    with tarfile.open(ldcad_tar_path, 'r:') as f:
        f.extractall(ltron_home)
    
    # unzip shadow
    print('-'*80)
    print('Unzipping shadow')
    shadow_seed_path = os.path.join(ldcad_path, 'seeds', 'shadow.sf')
    ldcad_shadow_path = os.path.join(ldcad_path, 'shadow')
    if not os.path.exists(ldcad_shadow_path):
        os.makedirs(ldcad_shadow_path)
    
        with zipfile.ZipFile(shadow_seed_path, 'r') as z:
            z.extractall(ldcad_shadow_path)
    
    # unzip offLib
    print('-'*80)
    print('Unzipping offLibShadow')
    ldcad_offlibshadow_csl_path = os.path.join(
        ldcad_shadow_path, 'offLib', 'offLibShadow.csl')
    ldcad_offlibshadow_path = os.path.join(
        ldcad_shadow_path, 'offLib', 'offLibShadow')
    if not os.path.exists(ldcad_offlibshadow_path):
        os.makedirs(ldcad_offlibshadow_path)
        with zipfile.ZipFile(ldcad_offlibshadow_csl_path, 'r') as z:
            z.extractall(ldcad_offlibshadow_path)

def install_collection(name, overwrite=False):
    print('='*80)
    print('Installing %s Data Collection'%name)
    
    print('-'*80)
    zip_path = os.path.join(settings.paths['collections'], '%s.zip'%name)
    download(settings.urls[name], zip_path, overwrite=overwrite)
    
    print('-'*80)
    print('Extracting collection %s'%name)
    extract_path = os.path.join(settings.paths['collections'], name)
    if not os.path.exists(extract_path) or overwrite:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(settings.paths['collections'])
    else:
        print('Already extracted.')

def install_splendor_meshes(resolution, overwrite=False):
    print('='*80)
    print('Installing Splendor Meshes (%s)'%resolution)
    print('-'*80)
    asset_name = 'ltron_assets_%s'%resolution
    install_assets(settings.urls[asset_name], asset_name, overwrite=overwrite)
    splendor_home = get_splendor_home()
    resolution_path = os.path.join(splendor_home, asset_name)
    resolution_cfg_path = resolution_path + '.cfg'
    generic_cfg_path = os.path.join(splendor_home, 'ltron_assets.cfg')
    if os.path.exists(generic_cfg_path):
        os.unlink(generic_cfg_path)
    os.symlink(resolution_cfg_path, generic_cfg_path)
    #generic_path = os.path.join(splendor_home, 'ltron_assets')
    #if os.path.exists(generic_path):
    #    os.unlink(generic_path)
    #os.symlink(resolution_path, generic_path)

default_settings_cfg = '''
[DEFAULT]
datasets = {HOME}/datasets
collections = {HOME}/collections

[paths]
ldraw = {HOME}/ldraw
ldcad = {HOME}/LDCad-1-6d-Linux
shadow = %(ldcad)s/shadow
shadow_ldraw = %(shadow)s/offLib/offLibShadow

[datasets]
random_six = %(collections)s/random_six/random_six.json
#snap_one = %(collections)s/snap_one/snap_one.json
#snap_one_frames = %(collections)s/snap_one/snap_one_frames.json
#snap_four = %(collections)s/snap_four/snap_four.json
#snap_four_frames = %(collections)s/snap_four/snap_four_frames.json
#conditional_snap_two = %(collections)s/conditional_snap_two/conditional_snap_two.json
#conditional_snap_two_frames = %(collections)s/conditional_snap_two/conditional_snap_two_frames.json

[collections]
omr = %(collections)s/omr
random_six = %(collections)s/random_six
#snap_one = %(collections)s/snap_one
#snap_four = %(collections)s/snap_four
#conditional_snap_two = %(collections)s/conditional_snap_two

[urls]
ltron = https://github.com/aaronwalsman/ltron
ldraw = http://www.ldraw.org/library/updates/complete.zip
ldcad = http://www.melkert.net/action/download/LDCad-1-6d-Linux.tar.bz2
ldcad_home = http://www.melkert.net/LDCad
omr_ldraw = https://omr.ldraw.org
omr = https://drive.google.com/uc?id=1nr3uut3QK2qCzRm3VjYKc4HNgsum8hLf
random_six = https://drive.google.com/uc?id=11K6Zu59aU7EXRcsY_ALcOJG1S2aXcVXz
ltron_assets_low = https://drive.google.com/uc?id=11p_vyeL_B_BK7gupI8_JvGGbffJ2kXiG
ltron_assets_high = https://drive.google.com/uc?id=1wIw-0YXx9QkQ9Kjpcvv5XsZFqdZrGj6U
'''

def make_settings_cfg(overwrite=False):
    settings_path = os.path.join(ltron_home, 'settings.cfg')
    if not os.path.exists(settings_path) or overwrite:
        print('Writing default settings file to: %s'%settings_path)
        with open(settings_path, 'w') as f:
            f.write(default_settings_cfg)
    else:
        print('Settings file already exists: %s'%settings_path)
