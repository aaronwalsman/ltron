import sys
import os
import json

from ltron.settings import PATHS

# revisit this once we have all snaps accounted for
MAX_SNAPS_PER_BRICK = 256 #4096
MAX_INSTANCES_PER_SCENE = 256#16384
#MAX_BRICKS_PER_SCENE = 8000 
MAX_EDGES_PER_SCENE = 4000#64000 

# default bounds on the scene in LDU
# 20 LDU = 1 stud + brick width, so 6000LDU = 300 studs ~= 8 feet
# TODO: scan all files and make sure they all fit inside this box
DEFAULT_WORLD_BBOX = ((-3000,-3000,-3000),(3000,3000,3000))

SHAPE_CLASS_LABELS = {}
SHAPE_CLASS_NAMES = {}
COLOR_CLASS_LABELS = {}
COLOR_CLASS_NAMES = {}

def reload_class_labels(): 
    class_labels = json.load(open(PATHS['class_labels'])) 
    SHAPE_CLASS_LABELS.clear()
    SHAPE_CLASS_LABELS.update(class_labels['shape']) 
    COLOR_CLASS_LABELS.clear()
    COLOR_CLASS_LABELS.update(class_labels['color'])
    num_shape_labels = max(SHAPE_CLASS_LABELS.values())+1
    num_color_labels = max(COLOR_CLASS_LABELS.values())+1
    setattr(sys.modules[__name__], 'NUM_SHAPE_CLASSES', num_shape_labels)
    setattr(sys.modules[__name__], 'NUM_COLOR_CLASSES', num_color_labels)
    SHAPE_CLASS_NAMES.clear()
    SHAPE_CLASS_NAMES.update(
        {value:key for key, value in SHAPE_CLASS_LABELS.items()})
    COLOR_CLASS_NAMES.clear()
    COLOR_CLASS_NAMES.update(
        {value:key for key, value in COLOR_CLASS_LABELS.items()})
reload_class_labels()
