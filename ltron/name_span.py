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
    
    def subspace(self, subspace):
        return NameSpan(**{
            n:span['shape'] for n,span in self.spans.items() if n in subspace
        })
    
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
    
    def unravel_all(self, i):
        def recurse(ns, ii):
            result = {}
            for name, span in ns.spans.items():
                chunk = ii[span['start']:span['end']]
                if isinstance(span['shape'], NameSpan):
                    result[name] = recurse(span['shape'], chunk)
                else:
                    result[name] = chunk
            return result
        return recurse(self, i)
    
    def ravel(self, name, *ijk):
        if isinstance(self.spans[name]['shape'], NameSpan):
            i = self.spans[name]['shape'].ravel(*ijk)
        else:
            i = numpy.ravel_multi_index(ijk, self.spans[name]['shape'])
        return self.spans[name]['start'] + i
