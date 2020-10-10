import os
import configparser

root_path = os.path.join(os.path.dirname(__file__), '..')
setup_cfg_path = os.path.join(root_path, 'setup.cfg')

print(setup_cfg_path)

parser = configparser.ConfigParser()
parser.read(setup_cfg_path)
print(parser.sections())

def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(root_path, path))

paths = {key : resolve_path(value)
        for key, value in dict(parser['paths']).items()}
