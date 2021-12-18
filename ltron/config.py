import os
from configparser import ConfigParser
import argparse
import json

class Config:
    '''
    A class for high-level constant configurations.
    To use this, create a subclass with class-level configuration values.
    Thsee can then be automatically parsed from the command line and from
    configuration (.cfg) files.
    
    Supports multiple inheritance, so that one config class can inherit values
    from many others with typical resolution order.  See example 2.
    
    Additionally supports a "set_dependents" argument that will be
    automatically called on initialization which can generate constants
    that depend on the others.  See example2.py below.
    
    This class supports bool, int, float, str and nested-json values for
    constants, although json is somewhat clunky to specify on the command line.
    
    example1.py:
    from ltron_torch.config import Config
    class MyTrainingConfig(Config):
        learning_rate = 3e-4
        momentum = 0.9
    
    config = MyTrainingConfig.from_commandline()
    print('learning_rate:', config.learning_rate)
    print('momentum:', config.momentum)
    
    command line:
    python example1.py --learning-rate 0.001
    learning_rate: 0.001
    momentum: 0.9
    
    example2.py:
    from ltron_torch.config import Config
    class MyTrainingConfig(Config):
        learning_rate = 3e-4
        momentum = 0.9
    
    class MyNetworkConfig(Config):
        image_height = 224
        image_width = 224
        
        def set_dependents(self):
            self.num_pixels = self.image_height * self.image_width
    
    class MyScriptConfig(MyTrainingConfig, MyNetworkConfig):
        pass
    
    config = MyScriptConfig.from_commandline()
    print('learning_rate:', config.learning_rate)
    print('momentum:', config.momentum)
    print('image_height:', config.image_height)
    print('image_width:', config.image_width)
    print('num_pixels:', config.num_pixels)
    
    example2.cfg
    [CONFIG]
    image_height = 28
    image_width = 28
    
    python example2.py --config example2.cfg
    learning_rate: 0.0003
    momentum: 0.9
    image_height: 28
    image_width: 28
    num_pixels: 784
    '''
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Config Argument: %s'%key
            setattr(self, key, value)
        
        inheritance = list(reversed(self.__class__.mro()))
        inheritance.append(self.__class__)
        applied_methods = set()
        for BaseClass in inheritance:
            try:
                method = getattr(BaseClass, 'set_dependents')
            except AttributeError:
                continue
            
            if method not in applied_methods:
                method(self)
                applied_methods.add(method)
        
        self.kwargs = kwargs
    
    def set_dependents(self):
        pass
    
    @classmethod
    def primary_attrs(cls):
        return [d for d in dir(cls) if
            d[:2] != '__' and
            not callable(getattr(cls, d))
        ]
    
    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=None)
        parser.add_argument('--config-section', type=str, default='CONFIG')
        primary_attrs = cls.primary_attrs()
        for primary_attr in primary_attrs:
            default_value = getattr(cls, primary_attr)
            argtype = type(default_value)
            if argtype not in (str, int, float):
                argtype = str
            parser.add_argument(
                '--' + primary_attr.replace('_', '-'),
                type=argtype,
                default=None,
                help="default: %s"%(default_value,))
        
        args = parser.parse_args()
        config_path = args.config
        config_section = args.config_section
        
        parser = ConfigParser()
        if config_path is not None:
            parser.read_file(open(os.path.expanduser(config_path)))
        
        if not parser.has_section(config_section):
            parser[config_section] = {}
        
        for key, value in vars(args).items():
            if key in ('config', 'config_section'):
                continue
            if value is not None:
                parser[config_section][key] = str(value)
        
        return cls.load_config(parser, section=config_section)
    
    @classmethod
    def translate(cls, other, **kwargs):
        args = {}
        for primary_attr in cls.primary_attrs():
            if primary_attr in kwargs:
                source_name = kwargs[primary_attr]
            else:
                source_name = primary_attr
            if hasattr(other, source_name):
                args[primary_attr] = getattr(other, source_name)
        return cls(**args)
    
    @classmethod
    def load_config(cls, cfg, section='CONFIG'):
        args = {}
        if isinstance(cfg, ConfigParser):
            parser = cfg
        else:
            parser = ConfigParser()
            parser.read_file(open(os.path.expanduser(cfg)))
        for name in parser[section]:
            try:
                args[name] = parser[section].getint(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getboolean(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getfloat(name)
                continue
            except ValueError:
                pass
            
            value = parser[section][name]
            try:
                value = json.loads(value)
            except(json.decoder.JSONDecodeError):
                pass
            args[name] = value
            
            '''
            value_string = parser[section][name]
            if ',' in value_string:
                for remove in '[]()':
                    value_string = value_string.replace(remove, '')
                def convert_value(value):
                    try:
                        v = float(value)
                        if v.is_integer():
                            return int(v)
                        else:
                            return v
                    except ValueError:
                        return value
                values = tuple(
                    convert_value(v) for v in value_string.split(','))
                args[name] = values
                continue
            
            args[name] = value_string
            '''
        
        return cls(**args)
    
    def write_config(self, file_path, section='CONFIG'):
        file_path = os.path.expanduser(file_path)
        parser = ConfigParser()
        try:
            parser.read_file(open(file_path))
        except FileNotFoundError:
            pass
        parser[section] = self.kwargs
        with open(file_path, 'w') as f:
            parser.write(f)
'''
class OldConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Config Argument: %s'%key
            setattr(self, key, value)
        self.set_dependents()
        self.kwargs = kwargs
    
    def set_dependents(self):
        pass
    
    @classmethod
    def load_config(cls, file_path, section='CONFIG'):
        args = {}
        parser = ConfigParser()
        parser.read_file(open(os.path.expanduser(file_path)))
        for name in parser[section]:
            try:
                args[name] = parser[section].getint(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getboolean(name)
                continue
            except ValueError:
                pass
            
            try:
                args[name] = parser[section].getfloat(name)
                continue
            except ValueError:
                pass
            
            value_string = parser[section][name]
            if ',' in value_string:
                for remove in '[]()':
                    value_string = value_string.replace(remove, '')
                def convert_value(value):
                    try:
                        v = float(value)
                        if v.is_integer():
                            return int(v)
                        else:
                            return v
                    except ValueError:
                        return value
                values = tuple(
                    convert_value(v) for v in value_string.split(','))
                args[name] = values
                continue
            
            args[name] = value_string
        
        return cls(**args)
    
    def write_config(self, file_path, section='CONFIG'):
        file_path = os.path.expanduser(file_path)
        parser = ConfigParser()
        try:
            parser.read_file(open(file_path))
        except FileNotFoundError:
            pass
        parser[section] = self.kwargs
        with open(file_path, 'w') as f:
            parser.write(f)

class CompositeConfig(Config):
    MemberClasses = ()
    def __init__(self, **kwargs):
        member_args = [{} for MemberClass in self.MemberClasses]
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                found_key = True
            else:
                found_key = False
            for args, MemberClass in zip(member_args, self.MemberClasses):
                if hasattr(MemberClass, key):
                    args[key] = value
                    found_key = True
            assert found_key, 'Invalid Config Argument: %s'%key
        self.set_dependents()
        self.kwargs = kwargs
        

def CompositeConfig(ConfigClasses):
    class CustomCompositeConfig(Config):
        def __init__(self, **kwargs):
            #self.configs = [ConfigClass() for ConfigClass in ConfigClasses]
            config_args = [{} for ConfigClass in ConfigClasses]
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    found_key = True
                else:
                    found_key = False
                for ConfigClass in ConfigClasses:
                    if hasattr(config, key):
                        setattr(config, key, value)
                        found_key = True
                assert found_key, 'Invalid Config Argument: %s'%key
            self.set_dependents()
            self.kwargs = kwargs
        
        def __getattr__(self, attr):
            for config in self.configs:
                if hasattr(config, attr):
                    return getattr(config, attr)
            
            raise AttributeError(attr)
        
        def __setattr__(self, attr, value):
            for config in self.configs:
                if hasattr(config, attr):
                    setattr(config, attr, value)
            
            raise AttributeError(attr)
        
        def set_dependents(self):
            for config in self.configs:
                config.set_dependents()
    
    return CustomCompositeConfig
'''
