import os
import configparser

root_path = os.path.join(os.path.dirname(__file__), '..')
setup_cfg_path = os.path.join(root_path, 'settings.cfg')

parser = configparser.ConfigParser()
parser.read(setup_cfg_path)

def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(root_path, path))

paths = {
    key : resolve_path(value)
    for key, value in dict(parser['paths']).items()
}

datasets = {
    key : resolve_path(value)
    for key, value in dict(parser['datasets']).items()
    if key not in parser['DEFAULT']
}

collections = {
    key : resolve_path(value)
    for key, value in dict(parser['collections']).items()
    if key not in parser['DEFAULT']
}

urls = {
    key : url
    for key, url in dict(parser['urls']).items()
    if key not in parser['DEFAULT']
}
