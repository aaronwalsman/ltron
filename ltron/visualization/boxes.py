import numpy

import PIL.Image as Image

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
