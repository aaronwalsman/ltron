import numpy

def segment_weight_image(segmentation, segment_weights, segment_ids):
    h, w = segmentation.shape
    image = numpy.zeros((h, w))
    
    num_segments = segment_weights.shape[0]
    for i in range(num_segments):
        image = (image +
                (segmentation == segment_ids[i]) * 255 * segment_weights[i])
    
    return image.astype(numpy.uint8)
