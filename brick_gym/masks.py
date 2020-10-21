import numpy

def halton_sequence(index, base):
    f = 1
    r = 0

    while index > 0:
        f = f/base
        r = r + f * (index % base)
        index = index // base

    return r

def halton_mask_color(index):
    index += 1
    r = halton_sequence(index, 2)
    #r = round(r * 255) / 255
    g = halton_sequence(index, 3)
    #g = round(g * 255) / 255
    b = halton_sequence(index, 5)
    #b = round(b * 255) / 255

    return r,g,b

def color_float_to_int(f):
    return round(f*255)

def color_floats_to_ints(fs):
    return tuple(color_float_to_int(c) for c in fs)

def neighboring_colors(color):
    neighbors = []
    for r in -1, 0, 1:
        for g in -1, 0, 1:
            for b in -1, 0, 1:
                neighbors.append((color[0]+r, color[1]+g, color[2]+b))
    return neighbors

halton_index = [0]
index_mask_colors = {}
mask_color_indices = {}

def index_to_mask_color(index):
    if index not in index_mask_colors:
        while True:
            color_floats = halton_mask_color(halton_index[0])
            halton_index[0] += 1
            color_ints = color_floats_to_ints(color_floats)
            neighbors = neighboring_colors(color_ints)
            if any(neighbor in mask_color_indices for neighbor in neighbors):
                continue

            index_mask_colors[index] = color_floats
            for neighbor in neighbors:
                mask_color_indices[neighbor] = index
            break

    return index_mask_colors[index]

def memoize_indices(indices):
    for index in indices:
        _ = index_to_mask_color(index)

memoize_indices(10000)

def get_mask(mask, mask_color):
    difference = numpy.abs(mask - numpy.array([[mask_color]]))
    close_enough = difference <= 1
    mask = numpy.all(close_enough, axis=2)
    return mask

def segmentation_target(mask):
    
