import os
import configparser
import json

from ltron.home import get_ltron_home

home_path = get_ltron_home()
settings_cfg_path = os.path.join(home_path, 'settings.cfg')

def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(path.format(HOME=home_path))

PATHS = {}
DATASETS = {}
SHARDS = {}
URLS = {}

def reload_settings():
    if os.path.exists(settings_cfg_path):
        settings_parser = configparser.ConfigParser()
        settings_parser.read(settings_cfg_path)
        
        PATHS.clear()
        PATHS.update({
            key : resolve_path(value)
            for key, value in dict(settings_parser['paths']).items()
        })

        DATASETS.clear()
        datasets_path = resolve_path(settings_parser['paths']['datasets'])
        DATASETS.update({
            fname.replace('.json', '') : os.path.join(datasets_path, fname)
            for fname in os.listdir(datasets_path)
            if fname.endswith('.json')
        })
        
        SHARDS.clear()
        shards_path = resolve_path(settings_parser['paths']['shards'])
        SHARDS.update({
            os.path.splitext(fname)[0] : os.path.join(shards_path, fname)
            for fname in os.listdir(shards_path)
            if '.tar' in fname
        })
        
        URLS.clear()
        URLS.update({
            key : url
            for key, url in dict(settings_parser['urls']).items()
            if key not in settings_parser['DEFAULT']
        })

reload_settings()
