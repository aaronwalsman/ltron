import os
import json

import numpy

import ltron.settings as settings
import ltron.constants as constants

# faster that PIL.ImageColor.getrgb
def hex_to_rgb(rgb):
    if rgb[0] == '#':
        rgb = rgb[1:]
    elif rgb[:2] == '0x':
        rgb = rgb[2:]
    return (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))

def rgb_to_hex(rgb):
    return '#' + ''.join(['0'*(c <= 16) + hex(c)[2:] for c in rgb]).upper()

ldsettings_path = os.path.join(settings.PATHS['ldraw'], 'LDConfig.ldr')
COLOR_NAME_TO_INDEX = {}
COLOR_INDEX_TO_NAME = {}
COLOR_INDEX_TO_RGB = {}
COLOR_INDEX_TO_EDGE_RGB = {}
COLOR_INDEX_TO_HEX = {}
COLOR_INDEX_TO_EDGE_HEX = {}
with open(ldsettings_path, 'r') as f:
    for line in f.readlines():
        line_parts = line.split()
        if len(line_parts) < 2:
            continue
        if line_parts[1] == '!COLOUR':
            name = line_parts[2]
            index = int(line_parts[4])
            COLOR_NAME_TO_INDEX[name] = index
            COLOR_INDEX_TO_NAME[index] = name
            color_hex = line_parts[6]
            color_rgb = hex_to_rgb(color_hex)
            edge_hex = line_parts[8]
            edge_rgb = hex_to_rgb(edge_hex)
            
            COLOR_INDEX_TO_RGB[index] = color_rgb
            COLOR_INDEX_TO_HEX[index] = color_hex
            COLOR_INDEX_TO_EDGE_RGB[index] = edge_rgb
            COLOR_INDEX_TO_EDGE_HEX[index] = edge_hex

def regenerate_color_class_labels():
    color_class_labels = {
        index : i+1 for i, index in enumerate(sorted(COLOR_INDEX_TO_RGB.keys()))
    }
    class_labels = json.load(open(settings.PATHS['class_labels']))
    class_labels['color'] = color_class_labels
    with open(settings.PATHS['class_labels'], 'w') as f:
        json.dump(class_labels, f, indent=2)
    constants.reload_class_labels()
