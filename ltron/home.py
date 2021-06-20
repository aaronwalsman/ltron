import os

def get_ltron_home():
    default_home = os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'))
    default_home = os.path.join(default_home, 'ltron')
    ltron_home = os.path.expanduser(os.getenv('LTRON_HOME', default_home))
    return ltron_home

def make_ltron_home():
    home = get_ltron_home()
    print('HOME')
    print(home)
    print(os.path.exists(home))
    if not os.path.exists(home):
        os.makedirs(home)
