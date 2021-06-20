#!/usr/bin/env python
import os
import requests
import zipfile

import ltron.settings as settings

if not os.path.isdir(settings.paths['data']):
    print('making data directory: %s'%settings.paths['data'])
    os.makedirs(settings.paths['data'])
else:
    print('data directory already exists: %s'%settings.paths['data'])

complete_zip_path = os.path.join(settings.paths['download'], 'complete.zip')
if os.path.exists(complete_zip_path):
    print('ldraw complete.zip already downloaded: %s'%complete_zip_path)
else:
    print('downloading ldraw complete.zip to: %s'%complete_zip_path)
    ldraw_url = str(settings.urls['ldraw_complete_zip'])
    r = requests.get(ldraw_url, allow_redirects=True)
    open(complete_zip_path, 'wb').write(r.content)

if os.path.exists(settings.paths['ldraw']):
    print('ldraw already extracted: %s'%settings.paths['ldraw'])
else:
    print('extracting LDraw contents to: %s'%settings.paths['ldraw'])
    with zipfile.ZipFile(complete_zip_path, 'r') as z:
        z.extractall(settings.paths['download'])
