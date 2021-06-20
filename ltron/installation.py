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

ltron_home = get_ltron_home()

'''
def make_data_paths():
    print('='*80)
    print('Making paths')
    print('-'*80)
    if not os.path.isdir(settings.paths['data']):
        print('Making data directory: %s'%settings.paths['data'])
        os.makedirs(settings.paths['data'])
    else:
        print('Data directory already exists: %s'%settings.paths['data'])

    print('-'*80)
    if not os.path.isdir(settings.paths['collections']):
        print('Making collections directory: %s'%settings.paths['collections'])
        os.makedirs(settings.paths['collections'])
    else:
        print(
            'Collections directory already exists: %s'%
            settings.paths['collections']
        )

def download(name, destination, url, overwrite=False, mode='request'):
    if os.path.exists(destination):
        if overwrite:
            print('Removing existing %s: %s'%(name, destination))
            os.remove(destination)
        else:
            print('%s already downloaded: %s'%(name, destination))
    if not os.path.exists(destination):
        print('Downloading %s to: %s'%(name, destination))
        if mode == 'request':
            r = requests.get(str(url), allow_redirects=True)
            open(destination, 'wb').write(r.content)
        elif mode == 'gdown':
            gdown.cached_download(url, destination, quiet=False)
'''

'''
def extract(
    name,
    source_path,
    extract_name,
    destination,
    overwrite=False,
    mode='zip'
):
    if os.path.exists(destination):
        if overwrite:
            print('Removing existing %s: %s'%(name, destination))
            shutil.rmtree(destination)
        else:
            print('%s already exists: %s'%(name, destination))
    if not os.path.exists(destination):
        source_dir = os.path.dirname(source_path)
        extract_path = os.path.join(source_dir, extract_name)
        print('Extracting %s contents to: %s'%(name, extract_path))
        if mode == 'zip':
            with zipfile.ZipFile(source_path, 'r') as z:
                z.extractall(source_dir)
        elif mode == 'bz2':
            with open(source_path, 'rb') as f_in:
                data = bz2.decompress(f_in.read())
                with open(extract_path, 'wb') as f_out:
                    f_out.write(data)
        elif mode == 'tar':
            with tarfile.open(source_path, 'r:') as t:
                t.extractall(source_dir)
        else:
            raise NotImplementedError
        if destination != extract_path:
            destination_directory = os.path.dirname(destination)
            if not os.path.exists(destination_directory):
                print('Making directory nescessary for %s location: %s'%(
                    name, destination_directory))
                os.makedirs(destination_directory)
            print('Moving %s to: %s'%(name, destination))
            os.rename(extract_path, destination)
'''

def install_ldraw(overwrite=False):
    print('='*80)
    print('Installing LDraw')
    make_ltron_home()
    print('-'*80)
    #complete_zip_path = os.path.join(settings.paths['data'], 'complete.zip')
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

'''
def install_shadow(overwrite=False):
    raise Exception('Needs overhaul, pull directly from Melkert')
    print('='*80)
    print('Installing LDCad Shadow Files')
    
    print('-'*80)
    shadow_zip_path = os.path.join(settings.paths['data'], 'shadow.zip')
    download(
        'Shadow',
        shadow_zip_path,
        settings.urls['shadow'],
        overwrite=overwrite,
    )
    
    print('-'*80)
    extract(
        'Shadow',
        shadow_zip_path,
        'shadow',
        settings.paths['shadow'],
        overwrite=overwrite,
        mode='zip',
    )
'''

def ldcad_license_agreement():
    ldcad_license_text = '''LDCad license agreement (V3)

LDCad and its configuration files, the software from here on, are free for personal (non commercial) and educational use. The software might be used in this manner, free of charge, by anyone as far local law permits.

The author (Roland Melkert) does not guarantee perfect operation of the software, nor can he be held responsible for any damage and / or loss resulting from the use of the software in anyway.

Using the software to (help) create digital material (including but not limited to instruction booklets) to be sold later on is permitted as long a single copy of the material is donated to the author free of charge.

It is permitted to customize and repackage the software as long this is done without modifying the main executable or adding third party software (including but not limited to 'adware' and 'spyware').

(re)Distribution of the software in any form is only allowed when done so free of charge and a reference to the original software's website (www.melkert.net/LDCad) is included. If it concerns a customized version this must be clearly stated and in such cases the package must (also) be made available through a public accessible website.

Permission is granted to post, display and/or distribute screenshots (including videoclips) of the software for use on social media and or promotional material.


By using the software you agree with the contents of this document and therefore agree with the license.

For questions, contact- or additional information visit: www.melkert.net/LDCad
'''
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
    
    print('-'*80)
    print('Unzipping shadow')
    shadow_seed_path = os.path.join(ldcad_path, 'seeds', 'shadow.sf')
    ldcad_shadow_path = os.path.join(ldcad_path, 'shadow')
    if not os.path.exists(ldcad_shadow_path):
        os.makedirs(ldcad_shadow_path)
    
        with zipfile.ZipFile(shadow_seed_path, 'r') as z:
            z.extractall(ldcad_shadow_path)
    
    '''
    print('-'*80)
    ldcad_tar = os.path.join(settings.paths['data'], '%s.tar'%ldcad_name)
    extract(
        'LDCad tar',
        ldcad_bz2,
        '%s.tar'%ldcad_name,
        ldcad_tar,
        overwrite=overwrite,
        mode='bz2',
    )
    '''
    '''
    print('-'*80)
    ldcad_path = os.path.join(settings.paths['data'], ldcad_name)
    extract(
        'LDCad',
        ldcad_tar,
        ldcad_name,
        settings.paths['ldcad'],
        overwrite=overwrite,
        mode='tar',
    )
    '''

def install_collection(name, overwrite=False):
    print('='*80)
    print('Installing %s Data Collection'%name)
    
    print('-'*80)
    zip_path = os.path.join(settings.paths['collections'], '%s.zip'%name)
    download(name, zip_path, settings.urls[name], overwrite, mode='gdown')
    '''
    if os.path.exists(zip_path):
        if overwrite:
            print('Removing existing zip file: %s'%zip_path)
            os.remove(zip_path)
        else:
            print('Zip file already downloaded: %s'%zip_path)
    if not os.path.exists(zip_path):
        print('Downloading %s.zip to: %s'%(name, zip_path))
        url = settings.urls[name]
        gdown.cached_download(url, zip_path, quiet=False)
    '''
    print('-'*80)
    extract(
        name,
        zip_path,
        name,
        settings.collections[name],
        overwrite=overwrite,
    )
    
    '''
    if os.path.exists(settings.collections[name]):
        if overwrite:
            print('Removing existing collection directory: %s'%
                settings.collections[name])
            shutil.rmtree(settings.collections[name])
        else:
            print('%s.zip already extracted: %s'%(
                name, settings.collections[name]))
    if not os.path.exists(settings.collections[name]):
        default_path = os.path.join(settings.paths['collections'], name)
        print('Extracting %s.zip contents to: %s'%(name, default_path))
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(settings.paths['collections'])
        if settings.collections[name] != default_path:
            directory = os.path.dirname(settings.collections[name])
            if not os.path.exists(directory):
                print('Making directory necessary for %s location: %s'%(
                    name, directory))
                os.makedirs(directory)
            print('Moving %s to: %s'%(name, settings.collections[name]))
            os.rename(default_path, settings.collections[name])
    '''

def install_splendor_meshes(resolution):
    print('='*80)
    print('Installing Splendor Meshes (%s)'%resolution)
    print('-'*80)
    asset_name = 'ltron_assets_%s'%resolution
    install_assets(settings.urls[asset_name])
    splendor_home = get_splendor_home()
    resolution_path = os.path.join(splendor_home, asset_name)
    resolution_cfg_path = resolution_path + '.cfg'
    generic_path = os.path.join(splendor_home, 'ltron_assets')
    generic_cfg_path = generic_path + '.cfg'
    if os.path.exists(generic_path):
        os.unlink(generic_path)
    if os.path.exists(generic_cfg_path):
        os.unlink(generic_cfg_path)
    os.symlink(resolution_path, generic_path)
    os.symlink(resolution_cfg_path, generic_cfg_path)
    '''
    zip_path = os.path.join(
        settings.paths['data'], 'splendor_%s.zip'%resolution)
    if os.path.exists(zip_path):
        if overwrite:
            print('Removing existing zip file: %s'%zip_path)
            os.remove(zip_path)
        else:
            print('Zip file already downloaded: %s'%zip_path)
    if not os.path.exists(zip_path):
        print('Downloading splendor_%s.zip to: %s'%(resolution, zip_path))
        url = settings.urls['splendor_%s'%resolution]
        gdown.cached_download(url, zip_path, quiet=False)
    
    print('-'*80)
    resolution_path = os.path.join(
        settings.paths['data'], 'splendor_%s'%resolution)
    if os.path.exists(resolution_path):
        if overwrite:
            print('Removing existing splendor directory: %s'%resolution_path)
            shutil.rmtree(resolution_path)
        else:
            print('splendor_%s.zip already extracted to: %s'%(
                resolution, resolution_path))
    if not os.path.exists(resolution_path):
        print('Extracting splendor_%s.zip contents to: %s'%(
            resolution, resolution_path))
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(settings.paths['data'])
    print('-'*80)
    print('Linking %s to %s'%(settings.splendor['assets'], resolution_path))
    if os.path.exists(settings.splendor['assets']):
        os.unlink(settings.splendor['assets'])
    os.symlink(resolution_path, settings.splendor['assets'])
    '''
