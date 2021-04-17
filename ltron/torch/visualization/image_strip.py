import numpy

from PIL import Image, ImageDraw

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
        center_voting_offsets = None,
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
    
    target_height = color_image.shape[0]
    
    if center_voting_offsets is not None:
        vector_image = offset_image(
                instance_id,
                center_voting_offsets,
                8)
        strip_components.append(vector_image)
        max_height = vector_image.shape[0]
    
    for raw, converter in (
            (color_image, lambda x : x),
            (dense_score, single_float_to_three_bytes),
            (dense_visibility, single_float_to_three_bytes),
            (dense_class_labels, masks.color_index_to_byte),
            (mask_segmentation, masks.color_index_to_byte),
            (instance_id, masks.color_index_to_byte)):
        
        if raw is not None:
            image = converter(raw)
            while image.shape[0] <= target_height // 2:
                image = numpy.repeat(image, 2, axis=0)
                image = numpy.repeat(image, 2, axis=1)
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
    cell_height = min(cell_height, 16)
    
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

def offset_image(
        segmentation, offsets, cell_height):
    
    small_image = masks.color_index_to_byte(segmentation)
    large_image = numpy.repeat(small_image, cell_height, axis=0)
    large_image = numpy.repeat(large_image, cell_height, axis=1)
    
    #large_image = Image.fromarray(large_image)
    #draw = ImageDraw.Draw(large_image)
    
    from skimage.draw import line
    
    h, w = offsets.shape[-2:]
    for y in range(h):
        for x in range(w):
            if segmentation[y,x] == 0:
                continue
            start_x = round(x * cell_height + 0.5 * cell_height)
            start_y = round(y * cell_height + 0.5 * cell_height)
            end_x = round((round(x + offsets[1,y,x]) + 0.5) * cell_height)
            end_y = round((round(y + offsets[0,y,x]) + 0.5) * cell_height)
            #end_x = round(
            #        (round(start_x + offsets[1,y,x]) + 0.5) * cell_height)
            #end_y = round(
            #        (round(start_y + offsets[0,y,x]) + 0.5) * cell_height)
            #end_x = round(start_x + offsets[1,y,x]) * cell_height)
            #end_y = round(start_y + offsets[0,y,x]) * cell_height)
            
            if start_x < 0 or start_x > 511:
                continue
            if start_y < 0 or start_y > 511:
                continue
            if end_x < 0 or end_x > 511:
                continue
            if end_y < 0 or end_y > 511:
                continue
            #draw.line((start_x, start_y, end_x, end_y), fill=(0,128,0))
            rr, cc = line(start_y, start_x, end_y, end_x)
            
            segment_id = segmentation[y,x]
            color = masks.color_index_to_byte(segment_id+1)
            
            large_image[rr, cc] = color
    
    #large_image = numpy.array(large_image)
    return large_image

