import os
import shutil
import requests
import zipfile
import bz2
import tarfile
import json

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
    
    print('-'*80)
    print('Checking for Licenses')
    if agree_to_zip_licenses(complete_zip_path):
        print('Extracting Contents To: %s'%ltron_home)
        with zipfile.ZipFile(complete_zip_path, 'r') as z:
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
        print('Must agree to all licensing.  Aborting LDCad install.')
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

def install_extras(overwrite=False):
    print('='*80)
    print('Installing Extras (blacklist, symmetry table, font)')
    
    print('-'*80)
    zip_path = os.path.join(ltron_home, 'extras.zip')
    download(settings.urls['extras'], zip_path, overwrite=overwrite)
    
    print('-'*80)
    print('Extracting extras')
    if not any((os.path.exists(os.path.join(ltron_home, f)) for f in
        ('blacklist.json', 'font', 'symmetry_table.json'))) or overwrite:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ltron_home)
    else:
        print('Already extracted.')

def install_episodes(collection, episode_name, overwrite=False):
    episode_path = os.path.join(
        settings.paths['collections'], collection, episode_name + '.zip')
    download(
        settings.urls[episode_name],
        episode_path,
        overwrite=overwrite,
    )

def install_pretrained_lstm_weights(overwrite=False):
    zip_path = os.path.join(ltron_home, 'eccv_pretrain_lstms.zip')
    download(
        settings.urls['pretrained_lstm_weights'],
        zip_path,
        overwrite=overwrite,
    )
    
    print('-'*80)
    print('Extracting pretrained LSTM weights')
    if not os.path.exists(
        os.path.join(ltron_home, 'eccv_pretrain_lstms')) or overwrite:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ltron_home)
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

ldcad_version = 'LDCad-1-6d2-Linux'
default_settings_cfg = '''
[DEFAULT]
collections = {HOME}/collections

[paths]
ldraw = {HOME}/ldraw
ldcad = {HOME}/LDCad-1-6d2-Linux
shadow = %(ldcad)s/shadow
shadow_ldraw = %(shadow)s/offLib/offLibShadow
font = {HOME}/font/RobotoCondensed-Regular.ttf

[datasets]
random_construction_6_6 = %(collections)s/random_construction_6_6/rc_6_6.json
omr_clean = %(collections)s/omr_clean/omr_clean.json

[collections]
random_construction_6_6 = %(collections)s/random_construction_6_6
omr = %(collections)s/omr
omr_clean = %(collections)s/omr_clean

[urls]
ltron = https://github.com/aaronwalsman/ltron
ldraw = http://www.ldraw.org/library/updates/complete.zip
ldcad = http://www.melkert.net/action/download/LDCad-1-6d2-Linux.tar.bz2
ldcad_home = http://www.melkert.net/LDCad
omr_ldraw = https://omr.ldraw.org
omr = https://drive.google.com/uc?id=1nr3uut3QK2qCzRm3VjYKc4HNgsum8hLf
omr_clean = https://drive.google.com/uc?id=15-z6YWvzakWsE4WNqXcvg1n72VNWGnbq
random_construction_6_6 = https://drive.google.com/uc?id=1uLaEWykyDWv3qv7q-8ppgT0GXdh6G9D3
extras = https://drive.google.com/uc?id=1SklSBjDC57p1rlPrJCM7TY-niGDlGIJy
random_construction_6_6_episodes_2 = https://drive.google.com/uc?id=1NkUzbalQB7DVmvLHuM1o7EVKjENmC9Gj
random_construction_6_6_episodes_4 = https://drive.google.com/uc?id=1kGboSReHCpzo3_mXd3LAIkssmKfa0Vke
random_construction_6_6_episodes_8 = https://drive.google.com/uc?id=1H1u3y1xH1YzACQ3Mm78Z-rvr-1O57ZXH
omr_episodes_2 = https://drive.google.com/uc?id=1HuJ5L-dZKA57HwGK5ZSZJTKx-Pd5oBTl
omr_episodes_4 = https://drive.google.com/uc?id=1xKTgKmQc1EKUeVi723-xuBr7kM_dhY_f
omr_episodes_8 = https://drive.google.com/uc?id=1d6Sesme2KGcf-PnsyEtcz3yR2xoE5jBL
pretrained_lstm_weights = https://drive.google.com/uc?id=1_aohXQsKhKwYicQPZcasn4Jw3mtX5kIB
ltron_assets_low = https://drive.google.com/uc?id=11p_vyeL_B_BK7gupI8_JvGGbffJ2kXiG
ltron_assets_high = https://drive.google.com/uc?id=1wIw-0YXx9QkQ9Kjpcvv5XsZFqdZrGj6U

[render]
color_scheme = ldraw
'''

default_blacklist_data = {
  "all": [
    "30520.dat",
    "3626.dat",
    "3625.dat",
    "3624.dat",
    "41879.dat",
    "3820.dat",
    "3819.dat",
    "3818.dat",
    "10048.dat",
    "973.dat",
    "3817.dat",
    "3816.dat",
    "3815.dat",
    "92198.dat",
    "92250.dat",
    "2599.dat",
    "93352.dat",
    "92245.dat",
    "92244.dat",
    "92248.dat",
    "92257.dat",
    "92251.dat",
    "92256.dat",
    "92241.dat",
    "92252.dat",
    "92258.dat",
    "92255.dat",
    "93352.dat",
    "92259.dat",
    "92243.dat",
    "62810.dat",
    "92240.dat",
    "92438.dat",
    "92242.dat",
    "92247.dat",
    "92251.dat",
    "87991.dat",
    "u9201.dat",
    "95227.dat"
  ],
  "large_400": [
    "44343p02.dat",
    "44343p02.dat",
    "36069a.dat",
    "21833-f1.dat",
    "785.dat",
    "22461.dat",
    "18912.dat",
    "36069b.dat",
    "30477.dat",
    "4186p01.dat",
    "4109601.dat",
    "85651c02.dat",
    "425p01.dat",
    "866c04.dat",
    "50384.dat",
    "21833-f2.dat",
    "u9232c00.dat",
    "915p01.dat",
    "75924.dat",
    "2886.dat",
    "2538.dat",
    "3811p03.dat",
    "2678.dat",
    "80548.dat",
    "948b.dat",
    "610.dat",
    "54093.dat",
    "u9494.dat",
    "2359p02.dat",
    "23221.dat",
    "10p0b.dat",
    "4093c.dat",
    "949ac02.dat",
    "30644.dat",
    "949a.dat",
    "309p04.dat",
    "21830-f2.dat",
    "u9058.dat",
    "2296.dat",
    "81022.dat",
    "2296p01.dat",
    "80602.dat",
    "6099p06.dat",
    "u9221.dat",
    "10p09.dat",
    "30030p01.dat",
    "210.dat",
    "125c01.dat",
    "30072.dat",
    "126.dat",
    "6851.dat",
    "92709c01.dat",
    "u9058c01.dat",
    "608p33.dat",
    "6490.dat",
    "96891-f2.dat",
    "2868.dat",
    "2360.dat",
    "92710c03.dat",
    "u9494p01c02.dat",
    "606p33.dat",
    "309p01.dat",
    "30184.dat",
    "608p01.dat",
    "3811p01.dat",
    "6092.dat",
    "u9310c07.dat",
    "76283.dat",
    "21835-f1.dat",
    "44336p01.dat",
    "96890-f1.dat",
    "47116.dat",
    "26438-f2.dat",
    "2840.dat",
    "3645p02.dat",
    "30473.dat",
    "53400c04.dat",
    "2359.dat",
    "27965.dat",
    "3497.dat",
    "76254.dat",
    "2372c01.dat",
    "6100p05.dat",
    "10p03.dat",
    "15265.dat",
    "2687.dat",
    "2372c01d01.dat",
    "2373.dat",
    "74780-f2.dat",
    "21830-f1.dat",
    "3947.dat",
    "6584.dat",
    "2541.dat",
    "u9494p01c01.dat",
    "57046c.dat",
    "2358p04.dat",
    "2881a.dat",
    "73696c04.dat",
    "44343.dat",
    "73697c01.dat",
    "u9494c01.dat",
    "4478.dat",
    "611p01.dat",
    "6099p01.dat",
    "613.dat",
    "80547.dat",
    "73697c03.dat",
    "80672.dat",
    "u9328.dat",
    "61898b.dat",
    "4478p04.dat",
    "2972.dat",
    "6100.dat",
    "40918-f2.dat",
    "54101.dat",
    "57046b.dat",
    "2552p02.dat",
    "608p02.dat",
    "26439-f1.dat",
    "70978.dat",
    "6099p05.dat",
    "u9234c00.dat",
    "61898cc01.dat",
    "51560-f1.dat",
    "26440-f1.dat",
    "2842c02.dat",
    "911.dat",
    "374p04.dat",
    "47996.dat",
    "609.dat",
    "31043.dat",
    "99013p01.dat",
    "20033c01.dat",
    "455.dat",
    "33177.dat",
    "u9233c02.dat",
    "92710.dat",
    "73696c02.dat",
    "455p01.dat",
    "61898e.dat",
    "u572p01.dat",
    "u9310c08.dat",
    "u9233.dat",
    "2842.dat",
    "51542.dat",
    "2880a.dat",
    "u9232c02.dat",
    "3811p05.dat",
    "48002a.dat",
    "u9310c05.dat",
    "66821-f1.dat",
    "2552p05.dat",
    "92709c02.dat",
    "u9495c01.dat",
    "610p01.dat",
    "608p03.dat",
    "4109600.dat",
    "309.dat",
    "71958.dat",
    "33763.dat",
    "6100p02.dat",
    "607.dat",
    "26440-f2.dat",
    "80546.dat",
    "6024p02.dat",
    "4093.dat",
    "87949-f2.dat",
    "2360p02.dat",
    "4187.dat",
    "6024.dat",
    "3334.dat",
    "44336p04.dat",
    "3811.dat",
    "73696.dat",
    "71958c01.dat",
    "44343p01.dat",
    "3857.dat",
    "u9233c01.dat",
    "61898c.dat",
    "54100.dat",
    "33080.dat",
    "949ac01.dat",
    "96890-f2.dat",
    "949b.dat",
    "211.dat",
    "u9231c02.dat",
    "607p01.dat",
    "2538a.dat",
    "0903.dat",
    "2748.dat",
    "2845.dat",
    "374p01.dat",
    "2372.dat",
    "92710c01.dat",
    "u334.dat",
    "85651c01.dat",
    "74781-f2.dat",
    "6099.dat",
    "10p01.dat",
    "30271.dat",
    "u9495c02.dat",
    "92710c02.dat",
    "3857p02.dat",
    "309p02.dat",
    "21825-f1.dat",
    "21835-f2.dat",
    "14661-f2.dat",
    "81043.dat",
    "6100p04.dat",
    "26438-f1.dat",
    "309p03.dat",
    "6099p03.dat",
    "93541.dat",
    "76252.dat",
    "52258.dat",
    "6099p04.dat",
    "64991c02.dat",
    "4268.dat",
    "65418.dat",
    "10p07.dat",
    "86501.dat",
    "2869c02.dat",
    "606p02.dat",
    "64991.dat",
    "85651.dat",
    "u9234c02.dat",
    "30255.dat",
    "92340.dat",
    "3026.dat",
    "4093a.dat",
    "55767.dat",
    "809.dat",
    "609p01.dat",
    "u9231c00.dat",
    "92709.dat",
    "606.dat",
    "44556.dat",
    "94318.dat",
    "20033c02.dat",
    "4478p02.dat",
    "915.dat",
    "u9232.dat",
    "u9495p01c02.dat",
    "2361p01.dat",
    "608.dat",
    "125.dat",
    "65417.dat",
    "22296.dat",
    "92711.dat",
    "51943.dat",
    "95195.dat",
    "73697c04.dat",
    "3229ac04.dat",
    "61898d.dat",
    "2361p03.dat",
    "87949-f1.dat",
    "u9234.dat",
    "280.dat",
    "2672.dat",
    "44336.dat",
    "23397.dat",
    "2358.dat",
    "262.dat",
    "21825-f2.dat",
    "612p01.dat",
    "76277.dat",
    "96891-f1.dat",
    "10.dat",
    "374.dat",
    "385.dat",
    "2552.dat",
    "66645b.dat",
    "6100p03.dat",
    "2359p03.dat",
    "26439-f2.dat",
    "6261.dat",
    "425.dat",
    "u9058c02.dat",
    "0904.dat",
    "30225b.dat",
    "6475.dat",
    "u9494c02.dat",
    "648.dat",
    "4186.dat",
    "70688.dat",
    "303.dat",
    "51353.dat",
    "3857p01.dat",
    "425p02.dat",
    "572c02.dat",
    "2842c01.dat",
    "96889-f2.dat",
    "6099p02.dat",
    "53184-f1.dat",
    "47978.dat",
    "30402.dat",
    "44079-f1.dat",
    "92088.dat",
    "57781.dat",
    "54779c01.dat",
    "0902.dat",
    "2671.dat",
    "6161.dat",
    "44342p02.dat",
    "51560-f2.dat",
    "31074.dat",
    "608p04.dat",
    "2869c01.dat",
    "u9220.dat",
    "73696c03.dat",
    "54779c02.dat",
    "606p01.dat",
    "572c01.dat",
    "4109603.dat",
    "2677.dat",
    "10p08.dat",
    "u9231c01.dat",
    "948a.dat",
    "2358p02.dat",
    "6181.dat",
    "41881.dat",
    "2552p06.dat",
    "48002.dat",
    "303c01.dat",
    "36069bp01.dat",
    "92709c04.dat",
    "76281.dat",
    "2891.dat",
    "87950-f2.dat",
    "80256.dat",
    "62812.dat",
    "64991c01.dat",
    "u9310c06.dat",
    "30401.dat",
    "4478p03.dat",
    "374p02.dat",
    "26.dat",
    "864.dat",
    "44343p03.dat",
    "948c.dat",
    "948ac01.dat",
    "36069ap01.dat",
    "42688.dat",
    "44342p01.dat",
    "6100p01.dat",
    "2892.dat",
    "61898bc01.dat",
    "3811p04.dat",
    "73697c02.dat",
    "4109602.dat",
    "92339.dat",
    "u9058c03.dat",
    "74781-f1.dat",
    "47122.dat",
    "76389.dat",
    "48002b.dat",
    "44342.dat",
    "53184-f2.dat",
    "2359p04.dat",
    "53475.dat",
    "61898dc01.dat",
    "74780-f1.dat",
    "73697.dat",
    "u1023a.dat",
    "18601.dat",
    "26436-f1.dat",
    "6099p07.dat",
    "73696c00.dat",
    "948ac02.dat",
    "865.dat",
    "76282.dat",
    "76280.dat",
    "915p04.dat",
    "u9494p01.dat",
    "57786.dat",
    "4478p01.dat",
    "49059.dat",
    "14661-f1.dat",
    "782.dat",
    "2552p01.dat",
    "64991c03.dat",
    "44079-f2.dat",
    "30225bp1.dat",
    "6940.dat",
    "u9233c00.dat",
    "915p02.dat",
    "4093b.dat",
    "50450.dat",
    "2361.dat",
    "u9038.dat",
    "33086.dat",
    "613p01.dat",
    "95.dat",
    "23421.dat",
    "87058.dat",
    "2841.dat",
    "73696c01.dat",
    "10p02.dat",
    "66645ap01.dat",
    "71735c01.dat",
    "2361p02.dat",
    "4616992.dat",
    "66645a.dat",
    "u9495p01c01.dat",
    "30030.dat",
    "43086.dat",
    "44336p03.dat",
    "3645.dat",
    "2358p03.dat",
    "44341p01.dat",
    "85976c02.dat",
    "262p01.dat",
    "10p05.dat",
    "10p06.dat",
    "99013.dat",
    "44341p02.dat",
    "10p04.dat",
    "92709c03.dat",
    "54779.dat",
    "10a.dat",
    "2538b.dat",
    "949c.dat",
    "3645p03.dat",
    "6525.dat",
    "811.dat",
    "3241ac04.dat",
    "u9234c01.dat",
    "353.dat",
    "374p03.dat",
    "44336p02.dat",
    "26436-f2.dat",
    "20033.dat",
    "10p0a.dat",
    "51542dq0.dat",
    "367.dat",
    "71122.dat",
    "6923.dat",
    "96889-f1.dat",
    "3229bc04.dat",
    "2360p01.dat",
    "2869.dat",
    "85976.dat",
    "44341.dat",
    "46305.dat",
    "51352.dat",
    "10b.dat",
    "3811p02.dat",
    "80671.dat",
    "80549.dat",
    "4196.dat",
    "611.dat",
    "57274.dat",
    "u9231.dat",
    "57915.dat",
    "2885.dat",
    "280c01.dat",
    "66645bp01.dat",
    "0901.dat",
    "73697c00.dat",
    "2359p01.dat",
    "2358p01.dat",
    "612.dat",
    "3645p04.dat",
    "u572p02.dat",
    "u9232c01.dat",
    "u1193.dat",
    "455p02.dat",
    "18913.dat"
  ]
}

def make_settings_cfg(overwrite=False):
    settings_path = os.path.join(ltron_home, 'settings.cfg')
    if not os.path.exists(settings_path) or overwrite:
        print('Writing default settings file to: %s'%settings_path)
        with open(settings_path, 'w') as f:
            f.write(default_settings_cfg)
    else:
        print('Settings file already exists: %s'%settings_path)

#def make_blacklist_json(overwrite=False):
#    blacklist_path = os.path.join(ltron_home, 'blacklist.json')
#    if not os.path.exists(blacklist_path) or overwrite:
#        print('Writing default blacklist file to: %s'%blacklist_path)
#        with open(blacklist_path, 'w') as f:
#            json.dump(default_blacklist_data, f)
