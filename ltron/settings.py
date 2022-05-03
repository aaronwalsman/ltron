import os
import configparser

from ltron.home import get_ltron_home

home_path = get_ltron_home()
settings_cfg_path = os.path.join(home_path, 'settings.cfg')

def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(path.format(HOME=home_path))

paths = {}
datasets = {}
collections = {}
urls = {}
render = {}

def reload_settings():
    if os.path.exists(settings_cfg_path):
        settings_parser = configparser.ConfigParser()
        settings_parser.read(settings_cfg_path)
        
        paths.clear()
        paths.update({
            key : resolve_path(value)
            for key, value in dict(settings_parser['paths']).items()
        })

        datasets.clear()
        datasets_path = resolve_path(settings_parser['paths']['datasets'])
        datasets.update({
            fname.replace('.json', '') : os.path.join(datasets_path, fname)
            for fname in os.listdir(datasets_path)
            if fname.endswith('.json')
        })
        
        collections.clear()
        collections_path = resolve_path(settings_parser['paths']['collections'])
        collections.update({
            fname.replace('.tar', '') : os.path.join(collections_path, fname)
            for fname in os.listdir(collections_path)
            if fname.endswith('.tar')
        })

        urls.clear()
        urls.update({
            key : url
            for key, url in dict(settings_parser['urls']).items()
            if key not in settings_parser['DEFAULT']
        })
        
        render.clear()
        render.update({
            key : value
            for key, value in dict(settings_parser['render']).items()
            if key not in settings_parser['DEFAULT']
        })

reload_settings()
