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
        setup_parser = configparser.ConfigParser()
        setup_parser.read(settings_cfg_path)
        
        paths.clear()
        paths.update({
            key : resolve_path(value)
            for key, value in dict(setup_parser['paths']).items()
        })

        datasets.clear()
        datasets.update({
            key : resolve_path(value)
            for key, value in dict(setup_parser['datasets']).items()
            if key not in setup_parser['DEFAULT']
        })

        collections.clear()
        collections.update({
            key : resolve_path(value)
            for key, value in dict(setup_parser['collections']).items()
            if key not in setup_parser['DEFAULT']
        })

        urls.clear()
        urls.update({
            key : url
            for key, url in dict(setup_parser['urls']).items()
            if key not in setup_parser['DEFAULT']
        })
        
        render.clear()
        render.update({
            key : value
            for key, value in dict(setup_parser['render']).items()
            if key not in setup_parser['DEFAULT']
        })

reload_settings()
