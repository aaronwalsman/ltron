from collections import OrderedDict

import numpy

class NameSpan:
    def __init__(self, **name_shapes):
        self.spans = OrderedDict()
        self.total = 0
        self.add_names(**name_shapes)
    
    def add_names(self, **name_shapes):
        for name, shape in name_shapes.items():
            if name in self.spans:
                assert ValueError('Name %s already exists'%name)
            
            if isinstance(shape, int):
                shape = (shape,)
            
            if isinstance(shape, NameSpan):
                num_items = shape.total
            else:
                num_items = 1
                for s in shape:
                    num_items *= s
            
            self.spans[name] = {
                'start':self.total,
                'end':self.total+num_items,
                'shape':shape,
            }
            self.total += num_items
    
    def keys(self):
        return self.spans.keys()
    
    def name_range(self, name):
        return self.spans[name]['start'], self.spans[name]['end']
    
    def get_shape(self, name):
        return self.spans[name]['shape']
    
    def unravel(self, i):
        for name, span in self.spans.items():
            if i >= span['start'] and i < span['end']:
                i -= span['start']
                if isinstance(span['shape'], NameSpan):
                    ijk = span['shape'].unravel(i)
                else:
                    ijk = numpy.unravel_index(i, span['shape'])
                return name, *ijk
        raise IndexError
    
    def ravel(self, name, *ijk):
        if isinstance(self.spans[name]['shape'], NameSpan):
            i = self.spans[name]['shape'].ravel(*ijk)
        else:
            i = numpy.ravel_multi_index(ijk, self.spans[name]['shape'])
        return self.spans[name]['start'] + i
    
    def unravel_vector(self, v, dim=0):
        result = {}
        for name, span in self.spans.items():
            #index_tuple = tuple(
            #    slice(None) if i != dim else slice(span['start'], span['end'])
            #    for i, s in enumerate(v.shape)
            #)
            index = [slice(None) for _ in v.shape]
            index[dim] = slice(span['start'], span['end'])
            index = tuple(index)
            chunk = v[index]
            if isinstance(span['shape'], NameSpan):
                result[name] = span['shape'].unravel_vector(chunk, dim=dim)
            else:
                reshape = v.shape[:dim] + span['shape'] + v.shape[dim+1:]
                result[name] = chunk.reshape(reshape)
        
        return result
    
    def ravel_vector(self, v, dim=0, out=None):
        result = []
        for name in self.keys():
            start, end = self.name_range(name)
            if out is not None:
                index = [slice(None) for _ in out.shape]
                index[dim] = slice(start, end)
                index = tuple(index)
                out[index] = v[name]
            else:
                result.append(v[name])
        
        if out is not None:
            return out
        
        else:
            return numpy.cat(result, dim=dim)
