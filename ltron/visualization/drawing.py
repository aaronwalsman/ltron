import numpy

from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.draw import line
    skimage_available = True
except ImportError:
    skimage_available = False

from splendor.masks import color_index_to_byte

import ltron.settings as settings

def draw_box(image, min_x, min_y, max_x, max_y, color):
    # expects numpy image hxwx3
    h, w = image.shape[:2]
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, w)
    max_y = min(max_y, h)
    image[min_y, min_x:max_x+1] = color
    image[max_y, min_x:max_x+1] = color
    image[min_y:max_y+1, min_x] = color
    image[min_y:max_y+1, max_x] = color

def draw_vector_field(image, vector_field, weight, color):
    assert skimage_available
    image_height, image_width = image.shape[:2]
    field_height, field_width = vector_field.shape[:2]
    
    scale_y = image_height / field_height
    scale_x = image_width / field_width
    
    color = numpy.array(color)
    
    for y in range(field_height):
        start_y = round(y * scale_y + scale_y/2)
        for x in range(field_width):
            start_x = round(x * scale_x + scale_x/2)
            
            dest_y = start_y + round(vector_field[y,x,0] * scale_y)
            dest_x = start_x + round(vector_field[y,x,1] * scale_x)
            
            if (dest_y < 0 or
                dest_x < 0 or
                dest_y >= image_height or
                dest_x >= image_width
            ):
                continue
            
            yy, xx = line(start_y, start_x, dest_y, dest_x)
            image[yy, xx] = color * weight[y,x] + image[yy,xx] * (1-weight[y,x])

def block_upscale_image(image, target_width, target_height):
    scale_y = target_height // image.shape[0]
    scale_x = target_width // image.shape[1]
    image = numpy.repeat(image, scale_y, axis=0)
    image = numpy.repeat(image, scale_x, axis=1)
    return image

def clamp(x, min_x, max_x):
    if x < min_x:
        return min_x
    elif x > max_x:
        return max_x
    else:
        return x

def draw_crosshairs(image, x, y, size, color):
    
    start_x = clamp(round(x-size), 0, image.shape[1]-1)
    center_x = clamp(round(x), 0, image.shape[1]-1)
    end_x = clamp(round(x+size+1), 0, image.shape[1]-1)
    
    start_y = clamp(round(y-size), 0, image.shape[0]-1)
    center_y = clamp(round(y), 0, image.shape[0]-1)
    end_y = clamp(round(y+size+1), 0, image.shape[0]-1)
    
    image[center_y, start_x:end_x] = color
    image[start_y:end_y, center_x] = color

def write_text(
    image,
    text,
    location=(10,10),
    font='Roboto-Regular',
    size=10,
    color=(0,0,0),
):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_path = settings.paths['font']
    font = ImageFont.truetype(font_path, size)
    #color = 'rgb(%i, %i, %i)'%color
    draw.text(location, text, color, font)
    
    return numpy.array(image)

def map_overlay(image, overlay, opacity, convert_mask_colors=False):
    h, w = image.shape[:2]
    if convert_mask_colors:
        overlay = color_index_to_byte(overlay)
    upsampled_overlay = block_upscale_image(overlay, w, h)
    upsampled_opacity = block_upscale_image(opacity, w, h)
    return (
        image * (1. - upsampled_opacity) +
        upsampled_overlay * upsampled_opacity).astype(numpy.uint8)

def stack_images_horizontal(images, align='top', background_color=(0,0,0)):
    max_h = max(image.shape[0] for image in images)
    sum_w = sum(image.shape[1] for image in images)
    out = numpy.zeros((max_h, sum_w, 3), dtype=numpy.uint8)
    out[:,:] = background_color
    start_x = 0
    for image in images:
        h, w = image.shape[:2]
        end_x = start_x + w
        if align=='top':
            start_y = 0
            end_y = h
        elif align=='bottom':
            start_y = max_h-h
            end_y = max_h
        out[start_y:end_y, start_x:end_x] = image
        start_x = end_x
    return out

def stack_images_vertical(images, align='left', background_color=(0,0,0)):
    sum_h = sum(image.shape[0] for image in images)
    max_w = max(image.shape[1] for image in images)
    out = numpy.zeros((sum_h, max_w, 3), dtype=numpy.uint8)
    out[:,:] = background_color
    start_y = 0
    for image in images:
        h, w = image.shape[:2]
        end_y = start_y + h
        if align=='left':
            start_x = 0
            end_x = w
        elif align=='right':
            start_x = max_w-w
            end_x = max_w
        out[start_y:end_y, start_x:end_x] = image
        start_y = end_y
    return out
