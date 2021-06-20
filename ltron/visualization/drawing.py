import numpy

import PIL.Image as Image

from skimage.draw import line

def draw_box(image, min_x, min_y, max_x, max_y, color):
    # expects numpy image hxwx3
    image[min_y, min_x:max_x+1] = color
    image[max_y, min_x:max_x+1] = color
    image[min_y:max_y+1, min_x] = color
    image[min_y:max_y+1, max_x] = color

def test(path):
    image = numpy.array(Image.open(path))
    draw_box(image, 10, 10, 20, 40, (255, 0, 0))
    Image.fromarray(image).save('box.png')

def draw_vector_field(image, vector_field, weight, color):
    
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
