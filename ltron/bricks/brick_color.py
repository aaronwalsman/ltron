import collections

import numpy

import ltron.ldraw.colors as ldraw_colors
#import ltron.ldraw.paths as ldraw_paths
from ltron.ldraw.parts import LDRAW_PARTS
from ltron.ldraw.commands import LDrawImportCommand
from ltron.ldraw.documents import (
    LDrawMPDMainFile,
    LDrawMPDInternalFile,
    LDrawLDR,
    LDrawDAT,
)

class BrickColorLibrary(collections.abc.MutableMapping):
    def __init__(self, colors = None):
        if colors is None:
            colors = {}
        self.colors = colors
    
    def import_document(self, document):
        new_colors = []
        for command in document.commands:
            if isinstance(command, LDrawImportCommand):
                reference_name = command.reference_name
                reference_document = (
                        document.reference_table['ldraw'][reference_name])
                if isinstance(reference_document, LDrawDAT):
                    if reference_name in LDRAW_PARTS:
                        new_colors.extend(self.load_colors([command.color]))
                elif isinstance(
                    reference_document,
                    (LDrawMPDMainFile, LDrawMPDInternalFile, LDrawLDR),
                ):
                    new_colors.extend(self.import_document(reference_document))
        
        return new_colors
    
    def load_colors(self, colors):
        new_colors = []
        for color in colors:
            #color = str(color)
            if color not in self:
                brick_color = BrickColor(color)
                self.colors[brick_color.color_name] = brick_color
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
        self.color_index = int(color)
        self.color_name = str(color)
        self.color_byte = ldraw_colors.get_color_rgb(
            self.color_index, (128,128,128))
    
    def splendor_material_args(self):
        material_args = {
            'flat_color' : numpy.array(self.color_byte)/255.,
            'ambient' : 1.0,
            'metal' : 0.0,
            'rough' : 0.5,
            'base_reflect' : 0.04,
        }
        
        return material_args
    
    def __str__(self):
        return self.color_name
    
    def __int__(self):
        return self.color_index
