import numpy

import renderpy.masks as masks

def make_image_strips(num_strips, concatenate=False, **kwargs):
    image_strips = [make_image_strip(
            **{key:value[i] for key, value in kwargs.items()})
            for i in range(num_strips)]
    if concatenate:
        image_strips = [image_strip.reshape(1, *image_strip.shape)
                for image_string in image_strips]
        image_strips = numpy.concatenate(image_strips, axis=0)
    
    return image_strips

def make_image_strip(
        color_image = None,
        dense_score = None,
        dense_visibility = None,
        dense_class_labels = None,
        mask_segmentation = None,
        instance_id = None,
        step_size = None,
        step_segment_ids = None,
        step_step_match = None,
        step_step_edge = None,
        state_size = None,
        state_segment_ids = None,
        step_state_match = None,
        step_state_edge = None):
    
    max_height = 0
    strip_components = []
    
    for raw, converter in (
            (color_image, lambda x : x),
            (dense_score, single_float_to_three_bytes),
            (dense_visibility, single_float_to_three_bytes),
            (dense_class_labels, masks.color_index_to_byte),
            (mask_segmentation, masks.color_index_to_byte),
            (instance_id, masks.color_index_to_byte)):
        
        if raw is not None:
            image = converter(raw)
            strip_components.append(image)
            max_height = max(max_height, image.shape[0])
    
    if step_segment_ids is not None:
        if max_height == 0:
            max_height = step_step_segment_id.shape[0] * 2
        
        step_step_match_image = cross_product_image(
                max_height//2,
                step_size,
                step_size,
                step_step_match,
                step_segment_ids,
                step_segment_ids)
        step_step_edge_image = cross_product_image(
                max_height//2,
                step_size,
                step_size,
                step_step_edge,
                step_segment_ids,
                step_segment_ids)
        
        step_height = step_step_match_image.shape[0]
        
        step_step_image = numpy.zeros(
                (max_height, step_step_match_image.shape[1], 3),
                dtype = numpy.uint8)
        step_step_image[0:step_height] = step_step_match_image
        step_step_image[max_height//2:step_height+max_height//2] = (
                step_step_edge_image)
        strip_components.append(step_step_image)
    
    if state_segment_ids is not None:
        if max_height == 0:
            max_height = step_step_segment_id.shape[0] * 2
        
        step_state_match_image = cross_product_image(
                max_height//2,
                step_size,
                state_size,
                step_state_match,
                step_segment_ids,
                state_segment_ids)
        step_state_edge_image = cross_product_image(
                max_height//2,
                step_size,
                state_size,
                step_state_edge,
                step_segment_ids,
                state_segment_ids)
        
        step_height = step_state_match_image.shape[0]
        
        step_state_image = numpy.zeros(
                (max_height, step_state_match_image.shape[1], 3),
                dtype = numpy.uint8)
        step_state_image[0:step_height] = step_state_match_image
        step_state_image[max_height//2:step_height+max_height//2] = (
                step_state_edge_image)
        strip_components.append(step_state_image)
    
    strip_components = [pad_to_height(component, max_height)
            for component in strip_components]
    
    return numpy.concatenate(strip_components, axis=1)

def single_float_to_three_bytes(raw):
    h, w = raw.shape[-2:]
    out = (raw * 255).astype(numpy.uint8)
    out = numpy.repeat(out.reshape(h,w,1), 3, axis=2)
    return out

def pad_to_height(image, max_height):
    if image.shape[0] == max_height:
        return image
    
    else:
        out = numpy.ones(
                (max_height, image.shape[1], 3), dtype=numpy.uint8) * 64
        out[:image.shape[0], :] = image
        return out

def cross_product_image(
        max_height, cells_y, cells_x, matrix, segment_id_y, segment_id_x):
    if cells_y is None:
        cells_y = matrix.shape[0] + 1
    else:
        cells_y = cells_y + 1
    if cells_x is None:
        cells_x = matrix.shape[1] + 1
    else:
        cells_x = cells_x + 1
    cell_height = max_height // cells_y
    
    small_image = numpy.zeros((cells_y, cells_x, 3), dtype=numpy.uint8)
    
    segment_id_y = segment_id_y.reshape(segment_id_y.shape[0])
    segment_id_x = segment_id_x.reshape(segment_id_x.shape[0])
    segment_colors_y = masks.color_index_to_byte(segment_id_y)
    segment_colors_x = masks.color_index_to_byte(segment_id_x)
    
    small_image[0, 1:matrix.shape[1]+1] = segment_colors_x
    small_image[1:matrix.shape[0]+1, 0] = segment_colors_y
    
    small_image[1:matrix.shape[0]+1, 1:matrix.shape[1]+1, :] = (
            matrix.reshape(*matrix.shape, 1) * 255)
    
    large_image = numpy.repeat(small_image, cell_height, axis=0)
    large_image = numpy.repeat(large_image, cell_height, axis=1)
    
    return large_image
