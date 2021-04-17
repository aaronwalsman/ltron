import collections

import numpy

import ltron.ldraw.colors as ldraw_colors

class BrickColorLibrary(collections.abc.MutableMapping):
    def __init__(self, colors = None):
        if colors is None:
            colors = {}
        self.colors = colors
    
    def load_from_instances(self, instances):
        '''
        new_colors = []
        for brick_instance in instances:
            color = brick_instance.color
            if color not in self:
                brick_color = BrickColor(color)
                self.colors[color] = BrickColor(color)
                new_colors.append(brick_color)
        
        return new_colors
        '''
        colors = [instance.color for instance in instances]
        return self.load_colors(colors)
    
    def load_colors(self, colors):
        new_colors = []
        for color in colors:
            if color not in self:
                brick_color = BrickColor(color)
                self.colors[color] = brick_color
                new_colors.append(brick_color)
        
        return new_colors
    
    def __getitem__(self, key):
        return self.colors[str(key)]
    
    def __setitem__(self, key, value):
        assert isinstance(value, BrickColor)
        self.colors[str(key)] = value
    
    def __delitem__(self, key):
        del(self.colors[str(key)])
    
    def __iter__(self):
        return iter(self.colors)
    
    def __len__(self):
        return len(self.colors)
    
class BrickColor:
    def __init__(self, color):
        self.color = color
        self.material_name = str(color)
        self.color_byte = ldraw_colors.color_index_to_alt_rgb.get(
                int(self.color), (128,128,128))
    
    def renderpy_material_args(self):
        material_args = {
            'flat_color' : numpy.array(self.color_byte)/255.,
            'ka' : 1.0,
            'kd' : 0.0,
            'ks' : 0.0,
            'shine' : 1.0,
            'image_light_kd' : 0.90,
            'image_light_ks' : 0.10,
            'image_light_blur_reflection' : 4.0
        }
        
        return material_args
