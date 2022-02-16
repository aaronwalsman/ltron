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
            assert hasattr(self.__class__, key), (
                'Invalid Config Argument: %s'%key)
            setattr(self, key, value)
        
        self.set_all_dependents()
    
    def set_all_dependents(self):
        inheritance = list(reversed(self.__class__.mro()))
        inheritance.append(self.__class__)
        applied_methods = set()
        for BaseClass in inheritance:
            if hasattr(BaseClass, 'set_dependents'):
                method = getattr(BaseClass, 'set_dependents')
                if method not in applied_methods:
                    method(self)
                    applied_methods.add(method)
    
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
        
        parser = ConfigParser(allow_no_value=True)
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
        for kwarg in kwargs:
            if not hasattr(cls, kwarg):
                raise AttributeError('The attribute "%s" does not exist'%kwarg)
        return cls(**args)
    
    @classmethod
    def load_config(cls, cfg, section='CONFIG'):
        args = {}
        if isinstance(cfg, ConfigParser):
            parser = cfg
        else:
            parser = ConfigParser(allow_no_value=True)
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
        
        return cls(**args)
    
    def as_dict(self):
        return {attr : getattr(self, attr) for attr in self.primary_attrs()}
    
    def write_config(self, file_path, section='CONFIG'):
        file_path = os.path.expanduser(file_path)
        parser = ConfigParser(allow_no_value=True)
        try:
            parser.read_file(open(file_path))
        except FileNotFoundError:
            pass
        parser[section] = self.as_dict()
        with open(file_path, 'w') as f:
            parser.write(f)
